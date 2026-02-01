"""
DAAM Segmentation Module

This module provides DAAM (Diffusion Attentive Attribution Maps) based segmentation
for real images using Stable Diffusion's cross-attention mechanism.

Key method: predict_key_space_omp() - Orthogonalizes the Keys of the target concept
against the Keys of competing concepts INSIDE the attention mechanism, before Softmax.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
from typing import List

# Suppress the specific FutureWarning from daam/utils.py regarding torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

try:
    from diffusers import StableDiffusionPipeline
    from daam import trace
except ImportError as e:
    print(f"Warning: diffusers or daam not installed. DAAMSegmenter will fail if initialized. Error: {e}")
    StableDiffusionPipeline = None
    trace = None


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
