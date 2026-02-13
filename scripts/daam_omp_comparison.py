#!/usr/bin/env python3
"""
DAAM OMP Comparison: True Key-Space OMP vs Post-Hoc Heatmap OMP

Compares two orthogonalization strategies for DAAM:
1. Post-Hoc Heatmap OMP: Current approach - compute heatmaps independently, then 
   orthogonalize flattened heatmap vectors via Gram-Schmidt.
2. True Key-Space OMP: New approach - modify the cross-attention K matrix inside the 
   UNet so that the target token's key vector is orthogonalized against distractor 
   token keys BEFORE the softmax attention computation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import sys
import math
import warnings
import copy
from typing import List, Dict, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from daam import trace as daam_trace
from daam.trace import DiffusionHeatMapHooker, UNetCrossAttentionHooker
from daam.hook import UNetCrossAttentionLocator
from daam.heatmap import RawHeatMapCollection
from daam.utils import auto_autocast

# Add scripts dir to path for daam_segmentation
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
from daam_segmentation import DAAMSegmenter

DATA_DIR = os.path.join(scripts_dir, "data")
OUTPUT_DIR = os.path.join(scripts_dir, "..", "docs", "daam_omp_comparison")


class Float32DAAMSegmenter(DAAMSegmenter):
    """DAAMSegmenter that uses float32 to avoid NaN on MPS."""
    def __init__(self, model_id="Manojb/stable-diffusion-2-base", device='mps'):
        if StableDiffusionPipeline is None:
            raise ImportError("Please install 'daam' and 'diffusers'")
        print(f"[DAAM-F32] Loading Stable Diffusion pipeline in float32: {model_id}...")
        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device)
        self.pipeline.enable_attention_slicing()
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        print("[DAAM-F32] Pipeline loaded (float32).")

    def predict(self, image_pil, prompt, size=512):
        """Override predict to use float32 tensors."""
        if self.device == 'cuda' or self.device == 'mps':
            torch.mps.empty_cache() if self.device == 'mps' else torch.cuda.empty_cache()

        w, h = image_pil.size
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        # Use float32 instead of half() to avoid NaN on MPS
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        self.scheduler.set_timesteps(50, device=self.device)
        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        concept = ""
        if prompt.startswith("a photo of a "):
            concept = prompt[len("a photo of a "):].strip(".").strip()
        elif prompt.startswith("a "):
            concept = prompt[2:].strip(".").strip()
        if not concept:
            concept = prompt.split()[-1]

        background_concepts = ["background", "floor", "tree", "person", "grass", "face"]
        background_str = ", ".join([f"a {bc}" for bc in background_concepts])
        augmented_prompt = f"{prompt}, a {concept}, {background_str}"

        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        prompt_embeds = text_embeddings
        latent_model_input = noisy_latents

        from daam import trace
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                _ = self.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
            global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)

            heatmap = None
            try:
                word_heat_map = global_heat_map.compute_word_heat_map(concept)
                heatmap = word_heat_map.heatmap
            except Exception:
                sub_words = concept.split()
                sub_heatmaps = []
                for sw in sub_words:
                    try:
                        whm = global_heat_map.compute_word_heat_map(sw).heatmap
                        sub_heatmaps.append(whm)
                    except Exception:
                        pass
                if sub_heatmaps:
                    heatmap = torch.stack(sub_heatmaps).mean(0)

            if heatmap is None and hasattr(global_heat_map, 'heat_maps'):
                if global_heat_map.heat_maps.shape[0] > 6:
                    heatmap = global_heat_map.heat_maps[5:-1].mean(0)
                else:
                    heatmap = global_heat_map.heat_maps.mean(0)

            if heatmap is None:
                heatmap = torch.zeros((h, w))

        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='nearest')
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.cpu()


# ==============================================================================
# True Key-Space OMP: Custom attention processor that orthogonalizes keys
# ==============================================================================

class KeySpaceOMPProcessor:
    """
    Custom attention processor that orthogonalizes the target token's key vector
    against distractor token keys in the cross-attention layers of the UNet.
    
    In cross-attention:
        Q = W_Q · spatial_latents   (queries from image features)
        K = W_K · text_embeddings   (keys from text)
        V = W_V · text_embeddings   (values from text)
        
    The attention logits are: A = softmax(Q K^T / sqrt(d_k))
    
    Key-Space OMP modifies K by orthogonalizing the target token's key(s) 
    against the distractor tokens' keys, so that the attention logits for 
    the target no longer respond to the shared components.
    """
    
    def __init__(
        self, 
        target_token_indices: List[int],
        distractor_token_indices: List[List[int]],
        beta: float = 1.0,
        heat_maps: RawHeatMapCollection = None,
        layer_idx: int = 0,
        latent_hw: int = 4096,
        context_size: int = 77,
        parent_trace=None,
    ):
        """
        Args:
            target_token_indices: Token indices of the target concept in the prompt
            distractor_token_indices: List of lists of token indices for each distractor
            beta: Orthogonalization strength (0=none, 1=full, >1=aggressive)
            heat_maps: RawHeatMapCollection to store attention maps
            layer_idx: Layer index for heat map storage
            latent_hw: Latent height*width for factor computation
            context_size: Context size (77 for SD)
            parent_trace: Parent DiffusionHeatMapHooker for gen_idx tracking
        """
        self.target_token_indices = target_token_indices
        self.distractor_token_indices = distractor_token_indices
        self.beta = beta
        self.heat_maps = heat_maps
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw
        self.context_size = context_size
        self.parent_trace = parent_trace
    
    def _orthogonalize_keys(self, key: torch.Tensor, n_heads: int) -> torch.Tensor:
        """
        Orthogonalize target token keys against distractor token keys.
        
        Args:
            key: Key tensor [batch*heads, seq_len, head_dim]
            n_heads: Number of attention heads
            
        Returns:
            Modified key tensor with orthogonalized target token keys
        """
        key = key.clone()
        
        # For each target token index
        for target_idx in self.target_token_indices:
            # Get the target key vector: [batch*heads, head_dim]
            target_key = key[:, target_idx, :]
            
            # Collect all distractor key vectors
            for dist_indices in self.distractor_token_indices:
                for dist_idx in dist_indices:
                    dist_key = key[:, dist_idx, :]  # [batch*heads, head_dim]
                    
                    # Normalize distractor key
                    dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # Project target onto distractor and subtract
                    # projection: [batch*heads, head_dim]
                    projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                    target_key = target_key - self.beta * projection
            
            # Write back orthogonalized key
            key[:, target_idx, :] = target_key
        
        return key
    
    @torch.no_grad()
    def _unravel_attn(self, x):
        """Unravel attention maps (from DAAM source)."""
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)
        
        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)
        
        maps = torch.stack(maps, 0)
        return maps.permute(1, 0, 2, 3).contiguous()
    
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        """Custom attention forward with key-space OMP."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # >>> KEY-SPACE OMP: Orthogonalize target keys against distractor keys <<<
        # Only apply to cross-attention (encoder_hidden_states != hidden_states)
        is_cross_attention = (key.shape[1] == self.context_size)
        if is_cross_attention and self.beta > 0:
            key = self._orthogonalize_keys(key, attn.heads)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store heat maps (same logic as DAAM's UNetCrossAttentionHooker)
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1])) if attention_probs.shape[1] > 0 else 8
        if self.parent_trace is not None:
            self.parent_trace._gen_idx += 1
        
        if self.heat_maps is not None and attention_probs.shape[-1] == self.context_size and factor != 8:
            maps = self._unravel_attn(attention_probs)
            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


def get_token_indices(tokenizer, prompt: str, concept: str) -> List[int]:
    """Get token indices for a concept within a prompt."""
    # Tokenize the full prompt
    tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)
    
    indices = []
    for i in range(len(tokens)):
        if tokens[i:i+len(concept_tokens)] == concept_tokens:
            # +1 because of the SOS token
            indices.extend(range(i + 1, i + 1 + len(concept_tokens)))
            break
    
    if not indices:
        # Fallback: just find any matching sub-token
        for i, tok in enumerate(tokens):
            for ct in concept_tokens:
                if ct in tok or tok in ct:
                    indices.append(i + 1)  # +1 for SOS
        
    return indices


def run_daam_with_key_space_omp(
    segmenter: DAAMSegmenter,
    image_pil: Image.Image,
    prompt: str,
    target_concept: str,
    competing_concepts: List[str],
    beta: float = 1.0,
    size: int = 512,
) -> torch.Tensor:
    """
    Run DAAM with true key-space OMP intervention.
    
    Instead of computing heatmaps independently and orthogonalizing them,
    this modifies the cross-attention K matrix inside the UNet.
    """
    pipeline = segmenter.pipeline
    tokenizer = segmenter.tokenizer
    text_encoder = segmenter.text_encoder
    vae = segmenter.vae
    unet = segmenter.unet
    scheduler = segmenter.scheduler
    device = segmenter.device
    
    w, h = image_pil.size
    
    # 1. Preprocess image
    img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = img_arr * 2.0 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    # 2. Encode to latents
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * 0.18215
    
    # 3. Build the combined prompt with all concepts
    # We need a prompt that contains BOTH target and distractor concepts
    # so that all token keys are projected into the same attention space
    all_concepts = [target_concept] + competing_concepts
    combined_prompt = f"a photo of a {', a '.join(all_concepts)}."
    
    # Get token indices for target and each distractor in the combined prompt
    target_indices = get_token_indices(tokenizer, combined_prompt, target_concept)
    distractor_indices = [
        get_token_indices(tokenizer, combined_prompt, comp)
        for comp in competing_concepts
    ]
    
    # Debug info
    tokens_debug = tokenizer.tokenize(combined_prompt)
    print(f"    Combined prompt: '{combined_prompt}'")
    print(f"    Tokens: {tokens_debug}")
    print(f"    Target '{target_concept}' indices: {target_indices}")
    for comp, idx in zip(competing_concepts, distractor_indices):
        print(f"    Distractor '{comp}' indices: {idx}")
    
    if not target_indices:
        print(f"    WARNING: Could not find target token indices, falling back to basic DAAM")
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    # 4. Prepare text embeddings
    text_input = tokenizer(
        combined_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    # 5. Add noise
    scheduler.set_timesteps(50, device=device)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([21], device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    
    # 6. Create heat map collection and set up key-space OMP processors
    heat_maps = RawHeatMapCollection()
    locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
    cross_attn_modules = locator.locate(unet)
    
    latent_hw = 4096  # 64x64 for SD 2.0
    
    # Create a simple object to track _gen_idx
    class GenIdxTracker:
        def __init__(self):
            self._gen_idx = 0
    
    tracker = GenIdxTracker()
    
    # Save original processors and install key-space OMP processors
    original_processors = {}
    for idx, module in enumerate(cross_attn_modules):
        original_processors[idx] = module.processor
        module.set_processor(KeySpaceOMPProcessor(
            target_token_indices=target_indices,
            distractor_token_indices=distractor_indices,
            beta=beta,
            heat_maps=heat_maps,
            layer_idx=idx,
            latent_hw=latent_hw,
            context_size=77,
            parent_trace=tracker,
        ))
    
    # 7. Forward pass
    try:
        with torch.no_grad():
            _ = unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=text_embeddings
            ).sample
    finally:
        # Restore original processors
        for idx, module in enumerate(cross_attn_modules):
            module.set_processor(original_processors[idx])
    
    # 8. Extract heatmap for the target concept tokens
    x = int(np.sqrt(latent_hw))
    factors = {0, 1, 2, 4, 8, 16, 32, 64}
    
    all_merges = []
    with auto_autocast(dtype=torch.float32):
        for (factor, layer, head), heat_map in heat_maps:
            if factor in factors and factor != 8:
                heat_map = heat_map.unsqueeze(1)
                all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))
    
    if not all_merges:
        print("    WARNING: No heat maps captured")
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    maps = torch.stack(all_merges, dim=0)
    maps = maps.mean(0)[:, 0]  # Average across layers/heads: [num_tokens, H, W]
    
    # Extract target token heatmaps and average them
    target_maps = []
    for tidx in target_indices:
        if tidx < maps.shape[0]:
            target_maps.append(maps[tidx])
    
    if not target_maps:
        print("    WARNING: Target token index out of range")
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    heatmap = torch.stack(target_maps).mean(0)  # [H, W]
    
    # Resize to original image dimensions
    heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap, size=(h, w), mode='nearest')
    heatmap = heatmap.squeeze()
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap.cpu()


def run_comparison(
    segmenter: DAAMSegmenter,
    image_path: str,
    target_concept: str,
    competing_concepts: List[str],
    name: str,
    beta: float = 1.0,
):
    """Run both methods and generate comparison visualization."""
    
    image_pil = Image.open(image_path).convert("RGB")
    prompt = f"a photo of a {target_concept}."
    
    print(f"\n{'=' * 60}")
    print(f"  '{target_concept}' vs {competing_concepts}")
    print(f"{'=' * 60}")
    
    # 1. Baseline DAAM (no OMP)
    print("  [1/3] Running baseline DAAM...")
    baseline_heatmap = segmenter.predict(image_pil, prompt, size=512)
    
    # 2. Post-hoc Heatmap OMP (current approach)
    print("  [2/3] Running post-hoc heatmap OMP...")
    posthoc_heatmap = segmenter.predict_key_space_omp(
        image_pil,
        prompt=prompt,
        target_concept=target_concept,
        competing_concepts=competing_concepts,
        omp_beta=beta,
    )
    
    # 3. True Key-Space OMP (new approach)
    print("  [3/3] Running true key-space OMP...")
    keyspace_heatmap = run_daam_with_key_space_omp(
        segmenter,
        image_pil,
        prompt=prompt,
        target_concept=target_concept,
        competing_concepts=competing_concepts,
        beta=beta,
        size=512,
    )
    
    # Generate visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Baseline DAAM
    axes[1].imshow(image_pil)
    hm = baseline_heatmap.numpy() if hasattr(baseline_heatmap, 'numpy') else baseline_heatmap
    axes[1].imshow(hm, cmap='jet', alpha=0.6)
    axes[1].set_title(f"DAAM Baseline\n'{target_concept}'", fontsize=11)
    axes[1].axis('off')
    
    # Post-hoc OMP
    axes[2].imshow(image_pil)
    hm_post = posthoc_heatmap.numpy() if hasattr(posthoc_heatmap, 'numpy') else posthoc_heatmap
    axes[2].imshow(hm_post, cmap='jet', alpha=0.6)
    axes[2].set_title(f"Post-Hoc Heatmap OMP\n'{target_concept}' ⊥ {competing_concepts}", fontsize=10)
    axes[2].axis('off')
    
    # Key-Space OMP
    axes[3].imshow(image_pil)
    hm_key = keyspace_heatmap.numpy() if hasattr(keyspace_heatmap, 'numpy') else keyspace_heatmap
    axes[3].imshow(hm_key, cmap='jet', alpha=0.6)
    axes[3].set_title(f"True Key-Space OMP\n'{target_concept}' ⊥ {competing_concepts}", fontsize=10)
    axes[3].axis('off')
    
    plt.suptitle(f"DAAM OMP Comparison: {name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Compute similarity metrics between methods
    baseline_flat = np.nan_to_num(hm.flatten(), nan=0.0)
    posthoc_flat = np.nan_to_num(hm_post.flatten(), nan=0.0)
    keyspace_flat = np.nan_to_num(hm_key.flatten(), nan=0.0)
    
    # Cosine similarity 
    cos_baseline_posthoc = np.dot(baseline_flat, posthoc_flat) / (np.linalg.norm(baseline_flat) * np.linalg.norm(posthoc_flat) + 1e-8)
    cos_baseline_keyspace = np.dot(baseline_flat, keyspace_flat) / (np.linalg.norm(baseline_flat) * np.linalg.norm(keyspace_flat) + 1e-8)
    cos_posthoc_keyspace = np.dot(posthoc_flat, keyspace_flat) / (np.linalg.norm(posthoc_flat) * np.linalg.norm(keyspace_flat) + 1e-8)
    
    print(f"\n  Cosine similarities:")
    print(f"    Baseline ↔ Post-Hoc: {cos_baseline_posthoc:.4f}")
    print(f"    Baseline ↔ Key-Space: {cos_baseline_keyspace:.4f}")
    print(f"    Post-Hoc ↔ Key-Space: {cos_posthoc_keyspace:.4f}")
    
    # Mean activation (lower = more suppression for wrong prompts)
    print(f"\n  Mean activation:")
    print(f"    Baseline:  {baseline_flat.mean():.4f}")
    print(f"    Post-Hoc:  {posthoc_flat.mean():.4f}")
    print(f"    Key-Space: {keyspace_flat.mean():.4f}")
    
    return {
        'baseline': baseline_heatmap,
        'posthoc': posthoc_heatmap,
        'keyspace': keyspace_heatmap,
        'cos_baseline_posthoc': cos_baseline_posthoc,
        'cos_baseline_keyspace': cos_baseline_keyspace,
        'cos_posthoc_keyspace': cos_posthoc_keyspace,
    }


def main():
    """Run the full comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize segmenter with float32 to avoid NaN on MPS
    print("Initializing Float32 DAAMSegmenter...")
    segmenter = Float32DAAMSegmenter(device=device)
    
    # Define comparison examples
    examples = [
        # Correct prompt: target IS in the image
        {
            "image": os.path.join(DATA_DIR, "cat_and_dog.jpeg"),
            "target": "dog",
            "competing": ["cat"],
            "name": "catdog_target_dog",
        },
        {
            "image": os.path.join(DATA_DIR, "cat_and_dog.jpeg"),
            "target": "cat",
            "competing": ["dog"],
            "name": "catdog_target_cat",
        },
        # Wrong/hallucination prompt: target is NOT in the image
        {
            "image": os.path.join(DATA_DIR, "shepherd_dog.png"),
            "target": "cat",
            "competing": ["dog"],
            "name": "shepherd_hallucinate_cat",
        },
        {
            "image": os.path.join(DATA_DIR, "bird.png"),
            "target": "dog",
            "competing": ["bird"],
            "name": "bird_hallucinate_dog",
        },
        # Multi-distractor
        {
            "image": os.path.join(DATA_DIR, "test_dog_and_car.jpg"),
            "target": "dog",
            "competing": ["car", "cat"],
            "name": "dogcar_target_dog",
        },
    ]
    
    all_results = {}
    for ex in examples:
        if not os.path.exists(ex["image"]):
            print(f"[SKIP] {ex['image']} not found")
            continue
        
        result = run_comparison(
            segmenter,
            image_path=ex["image"],
            target_concept=ex["target"],
            competing_concepts=ex["competing"],
            name=ex["name"],
            beta=1.0,
        )
        all_results[ex["name"]] = result
    
    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Example':<30} {'Base↔Post':<12} {'Base↔Key':<12} {'Post↔Key':<12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    for name, res in all_results.items():
        print(f"  {name:<30} {res['cos_baseline_posthoc']:.4f}       {res['cos_baseline_keyspace']:.4f}       {res['cos_posthoc_keyspace']:.4f}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
