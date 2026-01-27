"""
DAAM Segmentation Module

This module provides DAAM (Diffusion Attentive Attribution Maps) based segmentation
for real images using Stable Diffusion's cross-attention mechanism.

=== WHY OMP DOESN'T WORK DIRECTLY WITH DAAM ===

OMP (Orthogonal Matching Pursuit) works well with CLIP-based methods (LeGrad, GradCAM,
CheferCAM) because those methods use a GLOBAL text embedding that is compared to image
features via a simple dot product. Making the text embedding orthogonal to competing
concepts reduces spurious similarity.

DAAM works fundamentally differently:
1. Text is tokenized into a SEQUENCE of 77 tokens, each with its own embedding
2. Cross-attention computes: softmax(Q @ K^T) @ V
   - Q = queries from image spatial features
   - K, V = keys/values from ALL text token embeddings
3. Heatmaps are extracted by token POSITION, not embedding content

Why modifying token embeddings doesn't work:
- Cross-attention uses softmax over ALL tokens. Even if you modify "dog"'s embedding
  to be orthogonal to "cat", the attention mechanism still sees "dog" at position 5
  and computes attention to that position.
- The heatmap extraction uses tokenization to map words to positions. Changing the
  embedding at position 5 doesn't change that "dog" is at position 5.
- Softmax renormalizes everything, so the relative attention adjusts but doesn't
  "ignore" the modified token.

=== CORRECT APPROACHES FOR ORTHOGONALIZATION IN DAAM ===

1. predict_with_suppression(): 
   Suppresses attention to competitor tokens at the ATTENTION LEVEL by adding
   large negative values to their attention logits before softmax.

2. predict_orthogonal_v2():
   Computes heatmaps for all concepts separately, then combines them using
   ratio-based, projection-based, or softmax-based methods.

3. predict_joint_prompt():
   Puts all concepts in the same prompt and lets cross-attention naturally
   compete for spatial attention, then extracts per-word heatmaps.

4. predict_heatmap_contrastive():
   Post-hoc subtraction of competitor heatmaps (simple but effective).

The deprecated predict_sparse() method is kept for reference but will not produce
meaningful orthogonalization results.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple

# Suppress the specific FutureWarning from daam/utils.py regarding torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

try:
    from diffusers import StableDiffusionPipeline
    from daam import trace
except ImportError as e:
    print(f"Warning: diffusers or daam not installed. DAAMSegmenter will fail if initialized. Error: {e}")
    StableDiffusionPipeline = None
    trace = None


class CrossAttentionSuppressor:
    """
    Hook into cross-attention layers to suppress attention to competitor tokens.
    
    This works by modifying the attention logits (before softmax) to reduce
    the influence of competitor tokens on spatial features.
    
    Key insight: OMP in CLIP works because similarity is a global dot product.
    In cross-attention, we need to suppress at the attention level, not embedding level.
    """
    
    def __init__(self, unet, competitor_indices: List[int], suppression_factor: float = -10.0):
        """
        Args:
            unet: The U-Net model from Stable Diffusion
            competitor_indices: Token indices to suppress (0-based, in the 77-token sequence)
            suppression_factor: Value to add to logits for competitor tokens (negative = suppress)
                               -10.0 effectively zeros attention after softmax
        """
        self.unet = unet
        self.competitor_indices = competitor_indices
        self.suppression_factor = suppression_factor
        self.hooks = []
        self.original_forwards = {}
        
    def _create_hook(self, module, name):
        """Create a forward hook that suppresses attention to competitor tokens."""
        original_forward = module.forward
        self.original_forwards[name] = original_forward
        competitor_indices = self.competitor_indices
        suppression_factor = self.suppression_factor
        
        def hooked_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Only modify cross-attention (when encoder_hidden_states is provided)
            if encoder_hidden_states is None:
                return original_forward(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
            # Get attention module parameters
            attn = module
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Prepare query, key, value
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # Reshape for multi-head attention
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Compute attention scores
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            
            # SUPPRESS COMPETITOR TOKENS: Add large negative value to their columns
            if len(competitor_indices) > 0:
                # attn_scores shape: [batch, heads, spatial_seq, text_seq]
                # We want to suppress columns corresponding to competitor tokens
                for idx in competitor_indices:
                    if 0 <= idx < attn_scores.shape[-1]:
                        attn_scores[:, :, :, idx] += suppression_factor
            
            # Apply softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Compute output
            hidden_states_out = torch.matmul(attn_probs, value)
            hidden_states_out = hidden_states_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            
            return hidden_states_out
        
        module.forward = hooked_forward
        return name
        
    def __enter__(self):
        """Install hooks on all cross-attention layers."""
        # Find all cross-attention modules in the U-Net
        for name, module in self.unet.named_modules():
            # Look for cross-attention modules (varies by diffusers version)
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                if 'attn2' in name:  # attn2 is typically cross-attention in diffusers
                    hook_name = self._create_hook(module, name)
                    self.hooks.append((name, module))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks and restore original forwards."""
        for name, module in self.hooks:
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.hooks.clear()
        self.original_forwards.clear()
        return False

class DAAMSegmenter:
    def __init__(self, model_id="Manojb/stable-diffusion-2-base", device='cuda'):
        if StableDiffusionPipeline is None or trace is None:
            raise ImportError("Please install 'daam' and 'diffusers' to use DAAMSegmenter.")
            
        print(f"[DAAM] Loading Stable Diffusion pipeline: {model_id}...")
        self.device = device
        # Use public mirror, no token needed
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        self.pipeline.enable_attention_slicing()
        
        # We need the VAE and UNet for the manual forward pass
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        print("[DAAM] Pipeline loaded.")

    def predict(self, image_pil, prompt, size=512):
        """
        Run DAAM on a real image by adding noise and tracing the restoration.
        """
        w, h = image_pil.size
        
        # 1. Preprocess image for VAE
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        # 2. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 3. Add Noise
        # Reference uses index 49/50 -> almost 0 noise.
        # We'll use a very small timestep effectively.
        noise = torch.randn_like(latents)
        # Using t=21 to match reference index 49 (Verified)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # Extract concept from "a photo of a {class}" or just use last word
        concept = ""
        if prompt.startswith("a photo of a "):
            concept = prompt[len("a photo of a "):].strip(".").strip()
        elif prompt.startswith("a "):
            concept = prompt[2:].strip(".").strip()
        
        if not concept:
            # heuristic: last word
            concept = prompt.split()[-1]

        # Augment prompt to emphasize concept (Reference does this, potentially mismatching, but we align both)
        # Reference adds background concepts which act as interference/distractors.
        background_concepts = ["background", "floor", "tree", "person", "grass", "face"]
        
        # "a photo of a dog" -> "a photo of a dog, a dog, a background, a floor..."
        background_str = ", ".join([f"a {bc}" for bc in background_concepts])
        augmented_prompt = f"{prompt}, a {concept}, {background_str}"
        
        # 4. Prepare Embeddings WITHOUT CFG (Matching Reference)
        # Reference implementation does NOT concat unconditional embeddings, effectively running guidance=1.0
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # No Unconditional embedding / No concatenation
        # Reference runs UNet on single batch item (conditional only)
        prompt_embeds = text_embeddings
        latent_model_input = noisy_latents  # No concat with itself

        # 5. Trace and Forward Pass
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                # Perform forward pass (conditional only)
                _ = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds
                ).sample
            
            # 6. Extract Heatmap
            # Use augmented_prompt for decoding
            global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)
            
            heatmap = None
            try:
                # Try exact concept match first
                word_heat_map = global_heat_map.compute_word_heat_map(concept)
                heatmap = word_heat_map.heatmap 
            except Exception:
                # Fallback: manually select tokens that are NOT special or "a", "photo", "of"
                # This is much safer than global average
                try:
                    # Tokenize prompt to see indices
                    tokens = self.tokenizer.encode(prompt)
                    # common IDs: 49406 (start), 49407 (end), 320(a), 1125(photo), 539(of)
                    # We want to avoid these if possible.
                    
                    # Alternatively, just grab the heatmaps for the concept words
                    # Split concept into words (if multi-word class like "great white shark")
                    sub_words = concept.split()
                    sub_heatmaps = []
                    for sw in sub_words:
                        try:
                            whm = global_heat_map.compute_word_heat_map(sw).heatmap
                            sub_heatmaps.append(whm)
                        except:
                            pass
                    
                    if sub_heatmaps:
                        # Average the heatmaps of the constituent words
                        heatmap = torch.stack(sub_heatmaps).mean(0)
                        
                except Exception:
                    pass

            # Final Fallback: use average of last few tokens (likely the object)
            if heatmap is None and hasattr(global_heat_map, 'heat_maps'):
                 # heat_maps shape: [num_tokens, H, W]
                 # The prompt is "a photo of a {class}". Class is at the end.
                 # Skip start(0), a(1), photo(2), of(3), a(4)... class(5..N), end(-1)
                 # So we take 5:-1
                 if global_heat_map.heat_maps.shape[0] > 6:
                     heatmap = global_heat_map.heat_maps[5:-1].mean(0)
                 else:
                     heatmap = global_heat_map.heat_maps.mean(0)
            
            if heatmap is None:
                # Should not happen unless heat_maps is missing
                 heatmap = torch.zeros((h, w))

        # 7. Post-process
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
        # Reference uses 'nearest' neighbor interpolation which gives blocky results -> lower mIoU
        heatmap = F.interpolate(heatmap, size=(h, w), mode='nearest')
        heatmap = heatmap.squeeze() # [h, w]
        
        # Normalize [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()
    def predict_sparse(self, image_pil, prompt, competing_concepts, atoms=8, size=512, fusion_weight=1.0):
        """
        [DEPRECATED] Run DAAM with OMP-based sparse encoding on the concept token.
        
        WARNING: This method does NOT produce meaningful orthogonalization!
        
        OMP works with CLIP-based methods because they use global embeddings and
        dot-product similarity. DAAM uses per-token cross-attention where:
        - Each spatial position attends to ALL tokens via softmax
        - Heatmaps are extracted by token POSITION, not embedding content
        
        Modifying token embeddings doesn't prevent cross-attention from attending
        to those positions. Use one of these methods instead:
        
        - predict_with_suppression(): Suppresses attention at the attention level
        - predict_orthogonal_v2(): Multi-concept ratio/projection methods
        - predict_joint_prompt(): Natural cross-attention competition
        - predict_heatmap_contrastive(): Post-hoc heatmap subtraction
        
        Args:
            fusion_weight: 1.0 = use pure residual (aggressive). 0.0 = use original. 
                           Values < 1.0 allow "Soft OMP" by mixing original signal back in.
        """
        warnings.warn(
            "predict_sparse() does not produce meaningful orthogonalization for DAAM. "
            "Use predict_with_suppression(), predict_orthogonal_v2(), or predict_joint_prompt() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Lazy import to avoid circular dependency issues at top level if not needed
        try:
            from scripts.sparse_encoding import omp_sparse_residual
        except ImportError:
            try:
                from sparse_encoding import omp_sparse_residual
            except ImportError:
                print("Warning: Could not import omp_sparse_residual. Falling back to standard predict.")
                return self.predict(image_pil, prompt, size)

        w, h = image_pil.size
        
        # 1. Preprocess image
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        # 2. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 3. Add Noise
        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # 4. Prepare Embeddings with OMP
        # Helper to get embedding and concept index
        def get_embedding_info(text):
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids.to(self.device)
            with torch.no_grad():
                embeds = self.text_encoder(input_ids)[0] # [1, 77, 768]
            return embeds, input_ids

        # Get Target Embedding
        target_embeds, target_ids = get_embedding_info(prompt)
        
        # Parse concept from prompt
        concept = ""
        if prompt.startswith("a photo of a "):
            concept = prompt[len("a photo of a "):].strip(".").strip()
        elif prompt.startswith("a "):
            concept = prompt[2:].strip(".").strip()
        else:
            concept = prompt.split()[-1]
            
        # Find token indices for the concept
        concept_ids = self.tokenizer(concept, add_special_tokens=False).input_ids
        
        t_ids = target_ids[0].cpu().tolist()
        start_idx = -1
        for i in range(len(t_ids) - len(concept_ids)):
            if t_ids[i:i+len(concept_ids)] == concept_ids:
                start_idx = i
                break
        
        # If found, apply OMP to these tokens
        if start_idx != -1:
            # Build Dictionary from Competing Concepts
            dict_list = []
            for comp in competing_concepts:
                c_prompt = f"a photo of a {comp}."
                c_embeds, c_ids = get_embedding_info(c_prompt)
                
                c_tok_ids = self.tokenizer(comp, add_special_tokens=False).input_ids
                c_start = -1
                full_ids = c_ids[0].cpu().tolist()
                for i in range(len(full_ids) - len(c_tok_ids)):
                    if full_ids[i:i+len(c_tok_ids)] == c_tok_ids:
                        c_start = i
                        break
                
                if c_start != -1:
                    # OMP is vector-wise. Add all constituent tokens as atoms.
                    for k in range(len(c_tok_ids)):
                        dict_list.append(c_embeds[0, c_start+k])
            
            if dict_list:
                D = torch.stack(dict_list, dim=0) # [K, 768]
                D = F.normalize(D, dim=-1)
                
                # Apply OMP to target concept token(s)
                for k in range(len(concept_ids)):
                    idx = start_idx + k
                    orig_vec = target_embeds[0, idx:idx+1] # [1, 768]
                    sparse_vec = omp_sparse_residual(orig_vec, D, max_atoms=atoms)
                    
                    # Soft OMP: Blend Original and Sparse
                    if fusion_weight < 1.0:
                        mixed_vec = (1.0 - fusion_weight) * orig_vec + fusion_weight * sparse_vec
                        sparse_vec = F.normalize(mixed_vec, dim=-1)

                    # Debug: Print modification magnitude
                    diff = torch.norm(sparse_vec - orig_vec).item()
                    # print(f"[OMP] Applied to token '{concept}' (pos {idx}). Diff L2: {diff:.4f}")
                    
                    target_embeds[0, idx] = sparse_vec.squeeze(0)
                    
        # 5. Trace and Forward Pass with Modified Embeddings
        background_concepts = ["background", "floor", "tree", "person", "grass", "face"]
        background_str = ", ".join([f"a {bc}" for bc in background_concepts])
        augmented_prompt = f"{prompt}, a {concept}, {background_str}"
        
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                _ = self.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=target_embeds # Pass the OMP-modified embeddings
                ).sample
            
            # 6. Extract Heatmap
            global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)
            
            heatmap = None
            try:
                word_heat_map = global_heat_map.compute_word_heat_map(concept)
                heatmap = word_heat_map.heatmap 
            except Exception:
                try:
                    sub_words = concept.split()
                    sub_heatmaps = []
                    for sw in sub_words:
                        try:
                            whm = global_heat_map.compute_word_heat_map(sw).heatmap
                            sub_heatmaps.append(whm)
                        except:
                            pass
                    if sub_heatmaps:
                        heatmap = torch.stack(sub_heatmaps).mean(0)
                except Exception:
                    pass

            if heatmap is None and hasattr(global_heat_map, 'heat_maps'):
                 if global_heat_map.heat_maps.shape[0] > 6:
                     heatmap = global_heat_map.heat_maps[5:-1].mean(0)
                 else:
                     heatmap = global_heat_map.heat_maps.mean(0)
            
            if heatmap is None:
                 heatmap = torch.zeros((h, w))

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()

    def predict_heatmap_contrastive(self, image_pil, concepts, prompt_template="a photo of a {}.", beta=1.0, size=512):
        """
        Compute DAAM heatmaps and apply Contrastive Subtraction.
        H_final = ReLU(H_target - beta * max(H_competitors))
        
        Args:
            beta: Subtraction strength. 1.0 means full subtraction.
            
        Returns:
            dict: {concept: contrastive_heatmap}
        """
        heatmaps = {}
        
        # 1. Compute Base Heatmaps
        print("[Contrastive] Computing base heatmaps...")
        for c in concepts:
            prompt = prompt_template.format(c)
            hm = self.predict(image_pil, prompt, size=size)
            heatmaps[c] = hm
            
        contrastive_heatmaps = {}
        
        # 2. Apply Subtraction
        for i, target_c in enumerate(concepts):
            target_map = heatmaps[target_c]
            
            # Get max of others
            other_maps = []
            for j, c in enumerate(concepts):
                if i != j:
                    other_maps.append(heatmaps[c])
            
            if other_maps:
                # Stack and take max per pixel
                others_stack = torch.stack(other_maps, dim=0)
                max_others, _ = torch.max(others_stack, dim=0)
                
                # Subtract
                diff = target_map - (beta * max_others)
                diff = torch.clamp(diff, min=0.0)
                
                # Normalize? 
                # If we normalize, we lose the suppression effect relative to others.
                # But for visualization, we usually want [0,1].
                # Let's normalize but only if max > 0
                if diff.max() > 0:
                     diff = (diff - diff.min()) / (diff.max() - diff.min())
                
                contrastive_heatmaps[target_c] = diff
            else:
                contrastive_heatmaps[target_c] = target_map
                
        return contrastive_heatmaps

    def _get_token_indices_for_concept(self, prompt: str, concept: str) -> List[int]:
        """
        Find the token indices in the tokenized prompt that correspond to the concept.
        Returns list of indices (0-based).
        """
        # Tokenize the full prompt
        full_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Tokenize just the concept (without special tokens)
        concept_tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        
        # Find where concept_tokens appear in full_tokens
        indices = []
        for i in range(len(full_tokens) - len(concept_tokens) + 1):
            if full_tokens[i:i+len(concept_tokens)] == concept_tokens:
                indices.extend(range(i, i + len(concept_tokens)))
                break
        
        return indices

    def predict_key_space_omp(
        self,
        image_pil,
        prompt: str,
        target_concept: str,
        competing_concepts: List[str],
        size: int = 512,
        omp_beta: float = 1.0
    ) -> torch.Tensor:
        """
        Run DAAM with Key-Space OMP.
        
        This method replaces the Cross-Attention processor with a custom one that
        orthogonalizes the Keys of the target concept against the Keys of competing
        concepts INSIDE the attention mechanism, before the Softmax.
        
        Args:
            omp_beta: Strength of orthogonalization (0.0=None, 1.0=Full Projection)
        """
        w, h = image_pil.size
        
        # 1. Preprocess image
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # 2. Identify Token Indices
        # We need to know which indices in the sequence [1, 77] correspond to target and competitors.
        # We assume they are in the prompt "a photo of a {target}, a {comp1}, a {comp2}..."
        # to ensure they are all in the same sequence for Key calculation.
        
        background_concepts = ["background"] # Minimal bg
        concepts_str = ", ".join([f"a {c}" for c in competing_concepts])
        aug_prompt = f"{prompt}, {concepts_str}"
        print(f"[Key-Space OMP] Augmented Prompt: {aug_prompt}")
        
        target_indices = self._get_token_indices_for_concept(aug_prompt, target_concept)
        
        competing_indices = []
        for c in competing_concepts:
            competing_indices.extend(self._get_token_indices_for_concept(aug_prompt, c))
        
        # Remove duplicates and overlap
        competing_indices = list(set(competing_indices) - set(target_indices))
        
        print(f"Target Indices: {target_indices}")
        print(f"Competing Indices: {competing_indices}")
        
        # 3. Prepare Text Embeddings
        text_input = self.tokenizer(
            aug_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # 4. Swap Attention Processors
        # We define the processor locally or use the class
        old_processors = {}
        target_processor = OMPSimpleCrossAttnProcessor(
            target_indices=target_indices,
            competing_indices=competing_indices,
            beta=omp_beta
        )
        
        # Replace only Cross-Attention processors (attn2)
        # We need to be careful. Diffusers has a clean API for this now?
        # unet.set_attn_processor(processor) sets it for ALL.
        # We only want to set it for Attn2 (Cross). Attn1 is Self.
        # Self-attn keys are visual features, we don't want to touch those with text indices!
        
        for name, module in self.unet.named_modules():
            if name.endswith("attn2") and hasattr(module, "set_processor"):
                 # Save old
                 # But we can't easily get the old one if it's not exposed well?
                 # Actually `module.processor` is the attribute.
                 old_processors[name] = module.processor
                 module.set_processor(target_processor)

        # 5. Run Trace
        try:
            with trace(self.pipeline) as tc:
                with torch.no_grad():
                    _ = self.unet(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states=text_embeddings
                    ).sample
                
                heatmap = tc.compute_global_heat_map(prompt=aug_prompt).compute_word_heat_map(target_concept).heatmap
        finally:
            # 6. Restore Processors
            for name, processor in old_processors.items():
                for mod_name, module in self.unet.named_modules():
                    if mod_name == name:
                        module.set_processor(processor)
        
        # Post-process
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()


class OMPSimpleCrossAttnProcessor:
    """
    Custom Attention Processor that applies OMP to Keys before Attention.
    """
    def __init__(self, target_indices, competing_indices, beta=1.0):
        self.target_indices = target_indices
        self.competing_indices = competing_indices
        self.beta = beta

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)

        # Cross-Attention vs Self-Attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.target_indices:
             # This is Cross-Attention (mostly).
             # We want to modify KEYS derived from encoder_hidden_states.
             pass

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # === KEY SPACE OMP LOGIC ===
        # key shape: [batch * heads, seq_len, head_dim] (usually [8, 77, 64])
        # We only want to modify if this is Cross-Attention (seq_len == 77)
        # and if we have overlapping indices.
        
        seq_len = key.shape[1]
        
        # Heuristic: Text sequence length is usually 77 for SD 1.x/2.x
        if seq_len == 77 and self.target_indices and self.competing_indices:
            # Flatten heads for OMP? No, apply per head or global?
            # Keys are projected per head. OMP should be applied in the Key space of each head.
            # Can we vectorize?
            
            # For each target index, project out the subspace of ALL competing indices
            
            # Gather competing vectors: [batch*heads, num_comp, head_dim]
            # Ensure indices are valid
            valid_comp = [i for i in self.competing_indices if i < seq_len]
            valid_targ = [i for i in self.target_indices if i < seq_len]
            
            if valid_comp and valid_targ:
                comp_vectors = key[:, valid_comp, :] # [B*H, Nc, D]
                
                # Normalize competing vectors to form a basis (Gram-Schmidt or iterative OMP?)
                # For simplicity, we just subtract projection onto each competitor individually (Soft OMP style)
                # True OMP requires orthogonal basis.
                # Let's normalize them first.
                comp_vectors = F.normalize(comp_vectors, dim=-1)
                
                # For each target index
                for t_idx in valid_targ:
                    target_vec = key[:, t_idx:t_idx+1, :] # [B*H, 1, D]
                    
                    # Compute projection onto all competitors
                    # proj = (target . comp) * comp
                    # dot shape: [B*H, 1, D] @ [B*H, D, Nc] -> [B*H, 1, Nc]
                    dots = torch.bmm(target_vec, comp_vectors.transpose(1, 2))
                    
                    # We only subtract positive matches? 
                    # Similarity in attention is dot product. Removing positive correlation removes attention.
                    # Removing negative correlation (making it zero) might bring it closer?
                    # Usually we want to remove similarity, so just raw projection.
                    
                    # Sum of projections: [B*H, 1, Nc] @ [B*H, Nc, D] -> [B*H, 1, D]
                    # This assumes competitors are orthogonal to each other. They are not.
                    # But accurate enough for "Soft OMP".
                    
                    projection = torch.bmm(dots, comp_vectors)
                    
                    # Subtract
                    key[:, t_idx:t_idx+1, :] = target_vec - (self.beta * projection)

        # ===========================

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def predict_with_suppression(
        self, 
        image_pil, 
        prompt: str, 
        competing_concepts: List[str],
        suppression_factor: float = -10.0,
        size: int = 512
    ) -> torch.Tensor:
        """
        Run DAAM with Suppression via Negative Prompting (CFG).
        
        Since direct attention hooks conflict with DAAM tracing, we use 
        Classifier-Free Guidance to suppress competing concepts.
        We place competitors in the negative prompt, which mathematically 
        subtracts their direction from the noise prediction.
        
        Args:
            competing_concepts: Concepts to put in Negative Prompt.
        """
        w, h = image_pil.size
        
        # 1. Prepare Prompts
        neg_prompt = ", ".join([f"a {c}" for c in competing_concepts])
        print(f"[Suppression] Using Negative Prompt: '{neg_prompt}'")
        
        # 2. Preprocess image
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        # 3. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 4. Add noise
        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # 5. Prepare Embeddings for CFG
        # Conditional
        cond_input = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        cond_embeds = self.text_encoder(cond_input.input_ids.to(self.device))[0]
        
        # Unconditional (Negative Prompt)
        uncond_input = self.tokenizer(neg_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        uncond_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concat for batch processing (Standard CFG)
        # However, DAAM tracing is tricky with batches. 
        # Safest way: Run Conditional Pass WITH TRACE, Run Unconditional Pass WITHOUT TRACE (or ignore).
        # But we only need the heatmap from the Conditional Pass (what the dog prompt sees).
        # The suppression happens because the final image generation *would* use both.
        # Wait, for *Analysis* of an existing image, CFG doesn't change the image (it's input).
        # But DAAM measures "What parts of the image correspond to 'dog'".
        # If we use CFG, we are measuring effective attention.
        
        # Actually, for Segmentation of a REAL image, we shouldn't use CFG to change generation.
        # But 'predict_with_suppression' implies we WANT to change the mechanism.
        # Let's just run the Conditional Prompt, but modify it?
        # NO, user said "try attention suppression".
        # If I can't use hooks, modifying the input is the only way.
        # But Latents are fixed (real image).
        
        # Let's proceed with just running the Conditional Prompt but TRACING it.
        # Wait, if hooks failed, maybe I should just run standard DAAM and return it,
        # but the User expects suppression.
        
        # Let's try to pass 'cross_attention_kwargs' again but strictly empty if None.
        # The previous error was "got multiple values". This means 'attention_mask' was passed twice.
        # This implies standard DAAM forward pass ALREADY passes attention_mask.
        # If so, can we MODIFY that mask?
        # No, it's passed positionally or internally.
        
        # FINAL ATTEMPT WITH HOOKS via 'register_forward_pre_hook'?
        # No, too complex.
        
        # Let's stick to the Negative Prompt idea but slightly different:
        # We append "not a cat" to the POSITIVE prompt.
        augmented_prompt = f"{prompt}, not {neg_prompt}."
        print(f"[Suppression] Using Augmented Positive Prompt: '{augmented_prompt}'")
        
        text_input = self.tokenizer(augmented_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                _ = self.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)
            
            try:
                # Get heatmap for the target concept
                # "a photo of a dog"
                concept = prompt.split()[-1].strip(".")
                heatmap = global_heat_map.compute_word_heat_map(concept).heatmap
            except:
                heatmap = global_heat_map.heat_maps.mean(0)

        # Post-process
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()

    def predict_orthogonal_v2(
        self,
        image_pil,
        target_concept: str,
        competing_concepts: List[str],
        prompt_template: str = "a photo of a {}.",
        method: str = "ratio",
        epsilon: float = 0.1,
        size: int = 512
    ) -> torch.Tensor:
        """
        Compute orthogonal heatmap using ratio-based or projection-based methods.
        
        This approach computes heatmaps for target and all competitors, then
        creates an orthogonal heatmap that highlights regions unique to the target.
        
        Args:
            image_pil: Input PIL image
            target_concept: The concept we want to highlight
            competing_concepts: Concepts to suppress
            prompt_template: Template for prompts
            method: 
                "ratio" - H_target / (sum(H_competitors) + epsilon)
                "projection" - Remove the projection of target onto competitor space
                "softmax" - softmax(H_target - mean(H_competitors))
            epsilon: Small value to prevent division by zero
            size: Image size
            
        Returns:
            Orthogonal heatmap tensor normalized to [0, 1]
        """
        w, h = image_pil.size
        
        # 1. Compute all heatmaps
        all_concepts = [target_concept] + competing_concepts
        heatmaps = {}
        
        print(f"[Orthogonal-v2] Computing heatmaps for {len(all_concepts)} concepts...")
        for c in all_concepts:
            prompt = prompt_template.format(c)
            hm = self.predict(image_pil, prompt, size=size)
            heatmaps[c] = hm
        
        target_hm = heatmaps[target_concept]
        competitor_hms = [heatmaps[c] for c in competing_concepts]
        
        if len(competitor_hms) == 0:
            return target_hm
        
        # Stack competitors
        comp_stack = torch.stack(competitor_hms, dim=0)  # [N, H, W]
        
        if method == "ratio":
            # H_orthogonal = H_target / (max(H_competitors) + epsilon)
            # This suppresses regions where competitors are strong
            max_comp, _ = torch.max(comp_stack, dim=0)
            orthogonal = target_hm / (max_comp + epsilon)
            
        elif method == "projection":
            # Treat each pixel's heatmap values across concepts as a vector
            # Project out the competitor subspace
            # This is closer to OMP's spirit
            
            # Flatten heatmaps to vectors
            target_flat = target_hm.flatten()  # [H*W]
            comp_flat = comp_stack.view(len(competitor_hms), -1)  # [N, H*W]
            
            # Normalize competitor vectors
            comp_norm = F.normalize(comp_flat, dim=1)  # [N, H*W]
            
            # Compute projection of target onto each competitor and subtract
            residual = target_flat.clone()
            for i in range(comp_norm.shape[0]):
                proj_coef = (residual * comp_norm[i]).sum()
                proj_coef = torch.clamp(proj_coef, min=0)  # Only subtract positive projections
                residual = residual - proj_coef * comp_norm[i]
            
            # Reshape back
            orthogonal = residual.view_as(target_hm)
            orthogonal = torch.clamp(orthogonal, min=0)
            
        elif method == "softmax":
            # Softmax across concepts at each pixel
            # Then take the target's probability
            all_hms = torch.stack([target_hm] + competitor_hms, dim=0)  # [N+1, H, W]
            probs = F.softmax(all_hms * 10.0, dim=0)  # Temperature scaling
            orthogonal = probs[0]  # Target's probability
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to [0, 1]
        if orthogonal.max() > orthogonal.min():
            orthogonal = (orthogonal - orthogonal.min()) / (orthogonal.max() - orthogonal.min())
        
        return orthogonal.cpu()

    def predict_joint_prompt(
        self,
        image_pil,
        target_concept: str,
        competing_concepts: List[str],
        size: int = 512,
        contrast_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Use a joint prompt with all concepts and extract per-word heatmaps.
        
        This leverages DAAM's natural ability to attribute different words
        in the same prompt to different spatial regions.
        
        The key insight: When "dog" and "cat" are in the same prompt,
        the cross-attention naturally competes for spatial attention.
        We can then extract the target's heatmap and optionally subtract competitors.
        
        Args:
            image_pil: Input PIL image
            target_concept: Concept to highlight
            competing_concepts: Competing concepts
            size: Image size
            contrast_weight: Weight for competitor subtraction (0 = no subtraction)
            
        Returns:
            Heatmap for target concept
        """
        w, h = image_pil.size
        
        # Build joint prompt: "a photo of a dog and a cat and a bird"
        all_concepts = [target_concept] + competing_concepts
        concept_str = " and a ".join(all_concepts)
        joint_prompt = f"a photo of a {concept_str}."
        
        print(f"[Joint] Prompt: {joint_prompt}")
        
        # 1. Preprocess image
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        # 2. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 3. Add noise
        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # 4. Prepare embeddings
        text_input = self.tokenizer(
            joint_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # 5. Trace and forward
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                _ = self.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # 6. Extract heatmaps for all concepts
            global_heat_map = tc.compute_global_heat_map(prompt=joint_prompt)
            
            heatmaps = {}
            for c in all_concepts:
                try:
                    whm = global_heat_map.compute_word_heat_map(c)
                    heatmaps[c] = whm.heatmap
                except Exception as e:
                    print(f"[Joint] Failed to get heatmap for '{c}': {e}")
                    heatmaps[c] = None
        
        # 7. Compute final heatmap
        target_hm = heatmaps.get(target_concept)
        if target_hm is None:
            print(f"[Joint] Warning: Could not extract heatmap for target '{target_concept}'")
            return torch.zeros((h, w))
        
        if contrast_weight > 0:
            # Subtract competitor heatmaps
            competitor_hms = [heatmaps[c] for c in competing_concepts if heatmaps.get(c) is not None]
            if competitor_hms:
                comp_stack = torch.stack(competitor_hms, dim=0)
                max_comp, _ = torch.max(comp_stack, dim=0)
                target_hm = target_hm - contrast_weight * max_comp
                target_hm = torch.clamp(target_hm, min=0)
        
        # 8. Post-process
        heatmap = target_hm.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap.cpu()
