#!/usr/bin/env python3
"""
GAP-based hyperparameter optimization for anti-hallucination segmentation.

Goal: Find hyperparameters that MAXIMIZE the GAP between correct and wrong metrics.

The key insight is that we want:
1. CORRECT prompts: High/improved segmentation metrics  
2. WRONG prompts: Low/degraded segmentation metrics (no hallucination on wrong class)
3. The GAP (correct - wrong) should INCREASE compared to baseline

This is different from just maximizing correct, because if BOTH correct and wrong
improve by the same amount, the GAP stays the same (no anti-hallucination benefit).

Objective Function (Default):
    composite = (correct_miou + correct_acc + correct_map) - (wrong_miou + wrong_acc + wrong_map)

    This objective directly maximizes the gap between correct and wrong prompt performance.
    
    Legacy Objective (if --composite_lambda changed):
    gap_baseline = baseline_correct - baseline_wrong
    gap_improvement = gap_current - gap_baseline
    wrong_degradation = baseline_wrong - current_wrong
    composite = gap_improvement + λ × wrong_degradation

Ideal outcomes:
    - correct improves, wrong degrades → gap_improvement >> 0 AND wrong_degradation > 0 (excellent!)
    - correct stays same, wrong degrades → gap_improvement > 0, wrong_degradation > 0 (good!)
    - correct improves, wrong also improves equally → gap_improvement ≈ 0, wrong_degradation < 0 (bad)
    - correct improves less than wrong → gap_improvement < 0, wrong_degradation < 0 (very bad)

Optimization Approaches:
1. GAP-based composite score (default, recommended)
2. Multi-objective Pareto optimization (--multi_objective)

Usage:
    # GAP-based optimization (recommended)
    python scripts/optimize_anti_hallucination.py --n_trials 100 --limit 100

    # Multi-objective Pareto optimization
    python scripts/optimize_anti_hallucination.py --n_trials 100 --multi_objective

    # Adjust secondary weight for correct improvement
    python scripts/optimize_anti_hallucination.py --n_trials 50 --composite_lambda 1.0
"""

import time
import sys
import os
import math
import argparse
import json
import random
from typing import List
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

from legrad import LeWrapper, LePreprocess
import open_clip

# Import functions from benchmark_segmentation
# Import functions from benchmark_segmentation_v2
from benchmark_segmentation_v2 import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    batch_intersection_union,
    get_ap_scores,
    batch_pix_accuracy,
)

# Import sparse functions from sparse_encoding
from sparse_encoding import (
    omp_sparse_residual,
    wordnet_neighbors_configured,
    compute_map_for_embedding,
)

# Import DAAM segmenter
try:
    from daam_segmentation import DAAMSegmenter
except ImportError:
    DAAMSegmenter = None

# DAAM/Diffusers imports for true key-space OMP
try:
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention_processor import Attention
    from daam.hook import UNetCrossAttentionLocator
    from daam.heatmap import RawHeatMapCollection
    from daam.utils import auto_autocast
except ImportError:
    StableDiffusionPipeline = None


# ==============================================================================
# True Key-Space OMP Components (from benchmark_segmentation_v2.py)
# ==============================================================================

class KeySpaceOMPProcessor:
    """Custom attention processor that orthogonalizes the target token's key vector
    against distractor token keys in the cross-attention layers of the UNet."""

    def __init__(
        self, 
        target_token_indices: List[int],
        distractor_token_indices: List[List[int]],
        beta: float = 1.0,
        heat_maps: 'RawHeatMapCollection' = None,
        layer_idx: int = 0,
        latent_hw: int = 4096,
        context_size: int = 77,
        parent_trace=None,
    ):
        self.target_token_indices = target_token_indices
        self.distractor_token_indices = distractor_token_indices
        self.beta = beta
        self.heat_maps = heat_maps
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw
        self.context_size = context_size
        self.parent_trace = parent_trace

    def _orthogonalize_keys(self, key: torch.Tensor, n_heads: int) -> torch.Tensor:
        key = key.clone()
        
        for target_idx in self.target_token_indices:
            target_key = key[:, target_idx, :]
            
            for dist_indices in self.distractor_token_indices:
                for dist_idx in dist_indices:
                    dist_key = key[:, dist_idx, :]
                    dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                    projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                    target_key = target_key - self.beta * projection
            
            key[:, target_idx, :] = target_key
        
        return key

    def _unravel_attn(self, x):
        factor = int(math.sqrt(self.latent_hw // x.shape[1]))
        if factor == 0:
            factor = 1
        hw = int(math.sqrt(x.shape[1]))
        maps = x.reshape(x.shape[0], hw, hw, x.shape[2])
        maps = maps.permute(0, 3, 1, 2)
        return maps

    def __call__(
        self,
        attn: 'Attention',
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
        
        # KEY-SPACE OMP
        is_cross_attention = (key.shape[1] == self.context_size)
        if is_cross_attention and self.beta > 0:
            key = self._orthogonalize_keys(key, attn.heads)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store heat maps
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
    tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)
    
    indices = []
    concept_len = len(concept_tokens)
    for i in range(len(tokens) - concept_len + 1):
        if tokens[i:i + concept_len] == concept_tokens:
            # Token indices are offset by 1 due to BOS token
            indices.extend(range(i + 1, i + 1 + concept_len))
    
    if not indices:
        # Fallback: try single-token matching
        for i, tok in enumerate(tokens):
            if concept.lower() in tok.lower().replace('</w>', ''):
                indices.append(i + 1)
    
    return indices


def run_daam_with_key_space_omp(
    segmenter: 'DAAMSegmenter',
    image_pil: Image.Image,
    target_concept: str,
    competing_concepts: List[str],
    beta: float = 1.0,
    size: int = 512,
) -> torch.Tensor:
    """Run DAAM with true key-space OMP intervention."""
    pipeline = segmenter.pipeline
    tokenizer = segmenter.tokenizer
    text_encoder = segmenter.text_encoder
    vae = segmenter.vae
    unet = segmenter.unet
    scheduler = segmenter.scheduler
    device = segmenter.device
    
    w, h = image_pil.size
    
    # Match the pipeline's dtype (float16 on CUDA, float32 on MPS/CPU)
    dtype = next(vae.parameters()).dtype
    
    img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = img_arr * 2.0 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        latents = latents * 0.18215
    
    all_concepts = [target_concept] + competing_concepts
    combined_prompt = f"a photo of a {', a '.join(all_concepts)}."
    
    target_indices = get_token_indices(tokenizer, combined_prompt, target_concept)
    distractor_indices = [
        get_token_indices(tokenizer, combined_prompt, comp)
        for comp in competing_concepts
    ]
    
    # Filter indices to fit within context window (77 tokens).
    # The tokenizer truncates to max_length, so tokens beyond 76 don't exist in the key tensor.
    context_size = tokenizer.model_max_length  # 77
    target_indices = [i for i in target_indices if i < context_size]
    distractor_indices = [
        [i for i in group if i < context_size]
        for group in distractor_indices
    ]
    distractor_indices = [g for g in distractor_indices if g]  # drop empty groups
    
    if not target_indices:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    text_input = tokenizer(
        combined_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    scheduler.set_timesteps(50, device=device)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([21], device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    
    heat_maps = RawHeatMapCollection()
    locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
    cross_attn_modules = locator.locate(unet)
    
    latent_hw = 4096
    
    class GenIdxTracker:
        def __init__(self):
            self._gen_idx = 0
    tracker = GenIdxTracker()
    
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
    
    try:
        with torch.no_grad():
            _ = unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=text_embeddings
            ).sample
    finally:
        for idx, module in enumerate(cross_attn_modules):
            module.set_processor(original_processors[idx])
    
    x = int(np.sqrt(latent_hw))
    factors = {0, 1, 2, 4, 8, 16, 32, 64}
    
    all_merges = []
    with auto_autocast(dtype=torch.float32):
        for (factor, layer, head), heat_map in heat_maps:
            if factor in factors and factor != 8:
                heat_map = heat_map.unsqueeze(1)
                all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))
    
    if not all_merges:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    maps = torch.stack(all_merges, dim=0)
    maps = maps.mean(0)[:, 0]
    
    target_maps = []
    for tidx in target_indices:
        if tidx < maps.shape[0]:
            target_maps.append(maps[tidx])
    
    if not target_maps:
        return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    heatmap = torch.stack(target_maps).mean(0)
    
    # Resize to original image dimensions
    heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze()
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.cpu()


# Define semantic superclasses for distant negative sampling
IMAGENET_SUPERCLASSES = {
    # Animals
    'mammal': list(range(151, 294)),  # dogs, cats, bears, etc.
    'bird': list(range(80, 101)) + list(range(127, 147)),
    'reptile': list(range(33, 69)),
    'amphibian': list(range(25, 33)),
    'fish': list(range(0, 7)) + list(range(389, 398)),
    'invertebrate': list(range(69, 80)) + list(range(107, 127)) + list(range(300, 328)),
    
    # Vehicles & transport
    'vehicle': [403, 404, 405, 407, 408, 436, 468, 511, 573, 575, 609, 627, 654, 656, 670, 671, 717, 751, 757, 779, 802, 803, 817, 829, 864, 866, 867],
    
    # Food
    'food': list(range(924, 970)),
    
    # Household objects
    'furniture': [423, 453, 492, 493, 495, 516, 520, 526, 532, 559, 564, 765, 831, 857],
    
    # Nature/scenes
    'nature': list(range(970, 981)),
    
    # Clothing
    'clothing': [399, 400, 411, 445, 451, 452, 474, 501, 568, 601, 608, 610, 614, 655, 697, 735, 775, 841, 869],
}


def get_superclass(class_idx: int) -> str:
    """Get the superclass for a given ImageNet class index."""
    for superclass, indices in IMAGENET_SUPERCLASSES.items():
        if class_idx in indices:
            return superclass
    return 'other'


# Defined baselines for relative optimization
BASELINES = {
    'CLIP': {
        'LeGrad': {
            'correct': {'miou': 59.98, 'acc': 78.67, 'map': 83.86},
            'wrong': {'miou': 41.28, 'acc': 64.45, 'map': 68.03}
        },
        'GradCAM': {
            'correct': {'miou': 44.68, 'acc': 69.48, 'map': 74.94},
            'wrong': {'miou': 33.78, 'acc': 58.80, 'map': 67.36}
        },
        'CheferCAM': {
            'correct': {'miou': 48.71, 'acc': 69.32, 'map': 80.36},
            'wrong': {'miou': 44.95, 'acc': 66.51, 'map': 78.77}
        },
        'AttentionCAM': {
            'correct': {'miou': 40.14, 'acc': 68.67, 'map': 70.34},
            'wrong': {'miou': 33.40, 'acc': 62.55, 'map': 65.78}
        }
    },
    'SigLIP': {
        'LeGrad': {
            'correct': {'miou': 49.51, 'acc': 73.28, 'map': 78.32},
            'wrong': {'miou': 37.58, 'acc': 63.70, 'map': 63.95}
        },
        'GradCAM': {
            'correct': {'miou': 38.69, 'acc': 57.78, 'map': 71.43},
            'wrong': {'miou': 34.66, 'acc': 54.55, 'map': 68.50}
        },
        'CheferCAM': {
            'correct': {'miou': 37.66, 'acc': 60.53, 'map': 73.49},
            'wrong': {'miou': 36.77, 'acc': 59.38, 'map': 72.49}
        },
        'AttentionCAM': {
            'correct': {'miou': 38.95, 'acc': 65.63, 'map': 69.91},
            'wrong': {'miou':  32.06, 'acc':  59.62, 'map': 63.36}
        }
    },
    # DAAM uses Stable Diffusion, independent of CLIP/SigLIP
    'DAAM': {
        'DAAM': {
            'correct': {'miou': 65.73, 'acc': 81.34, 'map': 88.55},
            'wrong': {'miou': 59.75, 'acc': 76.80, 'map': 86.44}
        }
    }
}


def get_distant_class_indices(class_idx: int, all_wnids: list, wnid_to_idx: dict) -> list:
    """Get indices of classes from different superclasses (semantically distant)."""
    current_superclass = get_superclass(class_idx)
    distant_indices = []
    
    for superclass, indices in IMAGENET_SUPERCLASSES.items():
        if superclass != current_superclass:
            # Filter to only include indices that are in our dataset
            for idx in indices:
                # Map ImageNet index to our local index if available
                if idx < len(all_wnids):
                    distant_indices.append(idx)
    
    return distant_indices


def compute_gradcam_heatmap(model, image, text_emb_1x, layer_index: int = 8):
    """Compute a GradCAM heatmap (normalized to [0, 1]) for a single text embedding.
    
    Matches benchmark_segmentation_v2.py's compute_gradcam_for_embedding.
    """
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def compute_lrp_heatmap(model, image, text_emb_1x):
    """
    Compute AttentionCAM heatmap using gradient-weighted attention from last layer.
    
    AttentionCAM (also called CAM-Attn) is CLASS-DEPENDENT:
    - Forward pass to get attention maps from last transformer layer
    - Backward pass from text-image similarity to get attention gradients
    - Multiply attention by gradients and average across heads
    
    For SigLIP (attentional pooler models), per the LeGrad paper (Section A.4):
    AttentionCAM uses the attention maps of the attentional pooler instead of
    the last self-attention layer.
    
    Reference: 
    - Chefer et al. "Transformer Interpretability Beyond Attention Visualization"
    - LeGrad paper mentions AttentionCAM as a baseline
    
    Args:
        model: LeWrapper model
        image: Input image tensor [1, 3, H, W]
        text_emb_1x: Text embedding [1, embed_dim] - USED for class-dependent gradient
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    """
    import torch.nn.functional as F
    import math
    from open_clip.timm_model import TimmModel
    
    H_img, W_img = image.shape[-2:]
    
    try:
        is_siglip = isinstance(model.visual, TimmModel)
        
        with torch.enable_grad():
            if is_siglip:
                # SigLIP: Manually forward through trunk then pooler,
                # capturing pooler attention with requires_grad_(True)
                pooler = model.visual.trunk.attn_pool
                blocks = list(model.visual.trunk.blocks)
                
                # Trunk forward (all blocks)
                x = model.visual.trunk.patch_embed(image)
                if x.dim() == 4:
                    B, H, W, C = x.shape
                    x = x.reshape(B, H*W, C)
                else:
                    B, _, C = x.shape
                    
                if model.visual.trunk.pos_embed is not None:
                    x = x + model.visual.trunk.pos_embed
                
                for block in blocks:
                    x = block(x)
                
                B, N, C = x.shape
                
                # Attentional pooler manual forward
                if pooler.pos_embed is not None:
                    x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)
                
                q_latent = pooler.latent.expand(B, -1, -1)
                q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
                kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                q, k = pooler.q_norm(q), pooler.k_norm(k)
                
                attn_probs = (q * pooler.scale) @ k.transpose(-2, -1)
                attn_probs = attn_probs.softmax(dim=-1)  # [B, heads, num_latent, N]
                attn_probs.requires_grad_(True)
                
                x_pool = (attn_probs @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
                x_pool = pooler.proj(x_pool)
                x_pool = pooler.proj_drop(x_pool)
                x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))
                
                if pooler.pool == 'token':
                    pooled_feat = x_pool[:, 0]
                elif pooler.pool == 'avg':
                    pooled_feat = x_pool.mean(1)
                else:
                    pooled_feat = x_pool[:, 0]
                
                image_features = F.normalize(pooled_feat, dim=-1)
                
                # Compute similarity (class-dependent)
                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()
                
                # Gradient w.r.t. pooler attention
                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity,
                    inputs=[attn_probs],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )[0]
                
                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                # Gradient-weighted attention
                cam = attn_probs * grad  # [B, heads, num_latent, N]
                
                # Average over heads and select first latent query
                cam = cam.mean(dim=1)[:, 0]  # [B, N]
                cam = cam.clamp(min=0)
                
                num_patches = cam.shape[-1]
                grid_size = int(math.sqrt(num_patches))
                
                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                heatmap = cam[0].reshape(grid_size, grid_size)
                
            else:
                # CLIP/CoCa: Forward pass populates attention maps via hooks
                image_features = model.encode_image(image, normalize=True)
                
                # Compute similarity (class-dependent)
                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()
                
                # Get last block's attention maps
                if hasattr(model.visual, 'transformer'):
                    blocks_list = list(model.visual.transformer.resblocks)
                else:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                last_block = blocks_list[-1]
                
                if not hasattr(last_block, 'attn') or not hasattr(last_block.attn, 'attention_maps'):
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                attn_map = last_block.attn.attention_maps  # [batch*heads, N, N]
                
                # Compute gradient of similarity w.r.t. attention map
                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity,
                    inputs=[attn_map],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )[0]
                
                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                # Average gradients across all positions to get per-head weights
                grad_weights = grad.mean(dim=[1, 2], keepdim=True)  # [batch*heads, 1, 1]
                
                # Gradient-weighted attention (AttentionCAM formula)
                cam = attn_map * grad_weights  # [batch*heads, N, N]
                
                # Get CLS token attention to patches (row 0, columns 1:)
                cls_attn = cam[:, 0, 1:]  # [batch*heads, num_patches]
                
                # Average across heads and clamp negative values
                cls_attn = cls_attn.mean(dim=0).clamp(min=0)  # [num_patches]
                
                # Reshape to 2D grid
                num_patches = cls_attn.shape[0]
                grid_size = int(math.sqrt(num_patches))
                
                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5
                
                heatmap = cls_attn.reshape(grid_size, grid_size)
        
        # Upsample to image size
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, grid, grid]
        heatmap = F.interpolate(heatmap, size=(H_img, W_img), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()  # [H, W]
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.detach().cpu()
        
    except Exception as e:
        # Fallback on any error
        return torch.ones(H_img, W_img, device='cpu') * 0.5


# batch_intersection_union removed (imported from benchmark_segmentation_v2)


def compute_chefercam(model, image, text_emb_1x):
    """
    Computes GradCAM on the last Attention layer (attn_gradcam baseline).
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: attn_gradcam - GradCAM applied to attention maps from the last layer
    
    Matches benchmark_segmentation_v2.py's compute_chefercam.
    """
    import math
    from open_clip.timm_model import TimmModel
    
    model.zero_grad()
    num_prompts = text_emb_1x.shape[0]
    
    # Determine model type and get blocks
    if isinstance(model.visual, TimmModel):
        # SigLIP / Timm model
        blocks = list(model.visual.trunk.blocks)
        is_timm = True
        # Timm blocks have 'attn' attribute directly
    else:
        # CLIP / Standard VisionTransformer
        blocks = list(model.visual.transformer.resblocks)
        is_timm = False
    
    # Forward pass - need to do this in a way that keeps attention maps in the graph
    # We'll manually forward through the model to capture attention with gradients
    with torch.enable_grad():
        if is_timm:
            # --- SigLIP (Attentional Pooler) ---
            # Per LeGrad paper (Section A.4): CheferCAM treats the attentional
            # pooler as a "decoder transformer" and applies GradCAM on the
            # POOLER's attention maps, not the trunk's last self-attention layer.
            pooler = model.visual.trunk.attn_pool
            blocks = list(model.visual.trunk.blocks)
            
            # Trunk Forward Pass (all blocks, no manual attention needed)
            x = model.visual.trunk.patch_embed(image)
            
            # Flatten if NHWC (LeWrapper sets flatten=False)
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H*W, C)
            else:
                B, _, C = x.shape
                
            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed
            
            # Forward ALL trunk blocks normally
            for block in blocks:
                x = block(x)
            
            B, N, C = x.shape
            
            # --- Attentional Pooler Manual Forward (capture pooler attention) ---
            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)
            
            # Q from learnable latent, K/V from patch tokens
            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            
            q, k = pooler.q_norm(q), pooler.k_norm(k)
            
            # Pooler attention — this is what we need gradients for
            attn_weights = (q * pooler.scale) @ k.transpose(-2, -1)
            attn_weights = attn_weights.softmax(dim=-1)  # [B, heads, num_latent, N]
            attn_weights.requires_grad_(True)
            
            x_pool = (attn_weights @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))
            
            # Pooling
            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0]
                
            image_features = F.normalize(pooled_feat, dim=-1)
            bsz = B
            
        else:
            # CLIP
            # Forward through visual encoder
            x = model.visual.conv1(image)  # [B, C, H', W']
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, N]
            x = x.permute(0, 2, 1)  # [B, N, C]
            
            # Add class token
            batch_size = x.shape[0]
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)  # [B, N+1, C]
            
            # Add positional embedding
            num_patches = x.shape[1] - 1
            if hasattr(model.visual, 'original_pos_embed'):
                pos_embed = model.visual.original_pos_embed
            else:
                pos_embed = model.visual.positional_embedding
            
            if pos_embed.shape[0] != x.shape[1]:
                cls_pos = pos_embed[:1]
                patch_pos = pos_embed[1:]
                orig_size = int(math.sqrt(patch_pos.shape[0]))
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                new_size = int(math.sqrt(num_patches))
                patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
            
            x = x + pos_embed.unsqueeze(0).to(x.dtype)
            
            if hasattr(model.visual, 'ln_pre'):
                x = model.visual.ln_pre(x)
            
            x = x.permute(1, 0, 2)  # [N+1, B, C] for transformer
            
            # Forward through all blocks except the last
            for i in range(len(blocks) - 1):
                x = blocks[i](x)
            
            # For the last block, we need to capture attention with gradients
            # Manually compute attention for the last layer
            last_block = blocks[-1]
            last_attn = last_block.attn
            x_normed = last_block.ln_1(x)
            
            # Compute Q, K, V
            qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, embed_dim = q.shape
            num_heads = last_attn.num_heads
            head_dim = embed_dim // num_heads
            
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            # Compute attention weights (this is what we need gradients for)
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, N, N]
            attn_weights.requires_grad_(True)
            
            # Compute attention output
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = last_attn.out_proj(attn_output)
            
            # Continue forward
            x = x + attn_output
            x = x + last_block.mlp(last_block.ln_2(x))
            
            # Get final image features
            x = x.permute(1, 0, 2)  # [B, N+1, C]
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
        
        # --- Cam Calculation ---
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)  # [1, 1]
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradient w.r.t. attention weights
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False, allow_unused=True)[0]
        
        if grad is None:
            grad = torch.zeros_like(attn_weights)
        
        if is_timm:
            # SigLIP pooler attention: already [B, heads, num_latent, N]
            # Apply ReLU to gradients (GradCAM standard)
            grad = torch.clamp(grad, min=0)
            
            # Weight attention map by gradients
            cam = grad * attn_weights  # [B, heads, num_latent, N]
            
            # Average over heads, select first latent query (like LeGrad does)
            cam = cam.mean(dim=1)[:, 0]  # [B, N]
            
            num_patches = cam.shape[-1]
        else:
            # CLIP: attn_weights is [bsz*heads, N, N], reshape to 4D
            seq_len = attn_weights.shape[1]
            num_heads = blocks[-1].attn.num_heads
            grad = grad.view(bsz, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            
            # Apply ReLU to gradients (GradCAM standard)
            grad = torch.clamp(grad, min=0)
            
            # Weight attention map by gradients
            cam = grad * attn_weights  # [batch, heads, N, N]
            
            # Average over heads
            cam = cam.mean(dim=1)  # [batch, N, N]
            
            # Extract CLS token attention to patches (row 0, columns 1:)
            cam = cam[:, 0, 1:]  # [batch, num_patches]
            
            num_patches = cam.shape[-1]
        
        # Reshape to spatial grid
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size
        
        heatmap = cam.reshape(bsz, 1, h, w)
        
        # Upsample to image size
        heatmap = F.interpolate(
            heatmap, 
            size=image.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()


def compute_transformer_attribution(model, image, text_emb_1x, start_layer=1):
    """
    Computes Transformer Attribution (full LRP-based method from Chefer et al.).
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: transformer_attribution - Full LRP-based method with start_layer
    
    This is the main method from Chefer et al. (2021), not the baseline.
    The reference uses full LRP rules to propagate relevance through transformer layers.
    
    IMPORTANT LIMITATIONS:
    - The reference implementation uses a Supervised ViT with full LRP implementation
    - This implementation uses CLIP and approximates LRP with gradient-weighted attention
    - For exact reproduction of reported results, you need:
      1. A Supervised ViT (e.g., from timm) instead of CLIP
      2. The full LRP implementation from the reference repository
    - This simplified version aggregates gradient-weighted attention across layers,
      which approximates the LRP propagation but is not identical
    
    Args:
        model: LeWrapper model
        image: Input image tensor [1, 3, H, W]
        text_emb_1x: Text embedding [1, embed_dim]
        start_layer: Layer to start attribution from (default 1, as in reference)
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    
    Matches benchmark_segmentation_v2.py's compute_transformer_attribution.
    """
    import math
    from open_clip.timm_model import TimmModel
    
    model.zero_grad()
    
    # Determine model type and get blocks
    if isinstance(model.visual, TimmModel):
        blocks = list(model.visual.trunk.blocks)
        is_timm = True
    else:
        blocks = list(model.visual.transformer.resblocks)
        is_timm = False
    
    num_layers = len(blocks)
    
    # Ensure start_layer is valid
    if start_layer < 0:
        start_layer = num_layers + start_layer
    start_layer = max(0, min(start_layer, num_layers - 1))
    
    num_prompts = text_emb_1x.shape[0]
    
    # Forward pass with gradients enabled to capture attention maps
    with torch.enable_grad():
        all_attn_weights = []
        
        if is_timm:
            # --- SigLIP Forward ---
            x = model.visual.trunk.patch_embed(image)
            
            # Flatten if NHWC (LeWrapper sets flatten=False)
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H*W, C)
                
            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed
                
            pooler = model.visual.trunk.attn_pool
            
            B, N, C = x.shape
            
            for i, block in enumerate(blocks):
                if i < start_layer:
                    x = block(x)
                else:
                    # Manual Attention Forward
                    x_normed = block.norm1(x)
                    attn = block.attn
                    
                    qkv = attn.qkv(x_normed).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = attn.q_norm(q), attn.k_norm(k)
                    
                    attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
                    attn_weights = attn_weights.softmax(dim=-1)
                    attn_weights.requires_grad_(True)
                    all_attn_weights.append(attn_weights) # [B, heads, N, N]
                    
                    attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
                    attn_out = attn.proj(attn_out)
                    attn_out = attn.proj_drop(attn_out)
                    
                    if hasattr(block, 'ls1'):
                        attn_out = block.ls1(attn_out)
                        
                    x = x + attn_out
                    
                    x_mlp = block.mlp(block.norm2(x))
                    if hasattr(block, 'ls2'):
                        x_mlp = block.ls2(x_mlp)
                    x = x + x_mlp
            
            # Attentional Pooler Forward
            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)
            
            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = pooler.q_norm(q), pooler.k_norm(k)
            
            pool_attn = (q * pooler.scale) @ k.transpose(-2, -1)
            pool_attn = pool_attn.softmax(dim=-1)
            
            x_pool = (pool_attn @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))
            
            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0]
                
            image_features = F.normalize(pooled_feat, dim=-1)
            bsz = B
            
        else:
            # --- CLIP Forward ---
            # Forward through visual encoder
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            batch_size = x.shape[0]
            bsz = batch_size
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            # Add positional embedding
            num_patches = x.shape[1] - 1
            if hasattr(model.visual, 'original_pos_embed'):
                pos_embed = model.visual.original_pos_embed
            else:
                pos_embed = model.visual.positional_embedding
            
            if pos_embed.shape[0] != x.shape[1]:
                cls_pos = pos_embed[:1]
                patch_pos = pos_embed[1:]
                orig_size = int(math.sqrt(patch_pos.shape[0]))
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                new_size = int(math.sqrt(num_patches))
                patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
            
            x = x + pos_embed.unsqueeze(0).to(x.dtype)
            
            if hasattr(model.visual, 'ln_pre'):
                x = model.visual.ln_pre(x)
            
            x = x.permute(1, 0, 2)
            
            # Forward through blocks and collect attention maps from start_layer onwards
            for i, block in enumerate(blocks):
                if i < start_layer:
                    # Forward normally for layers before start_layer
                    x = block(x)
                else:
                    # For layers from start_layer, capture attention weights
                    attn_module = block.attn
                    x_normed = block.ln_1(x)
                    
                    qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                    
                    seq_len, bsz, embed_dim = q.shape
                    num_heads = attn_module.num_heads
                    head_dim = embed_dim // num_heads
                    
                    q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    
                    scale = float(head_dim) ** -0.5
                    attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
                    attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, N, N]
                    attn_weights.requires_grad_(True)
                    all_attn_weights.append(attn_weights)
                    
                    attn_output = torch.bmm(attn_weights, v)
                    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
                    attn_output = attn_module.out_proj(attn_output)
                    
                    x = x + attn_output
                    x = x + block.mlp(block.ln_2(x))
            
            # Get final image features
            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradients for all attention layers
        grads = torch.autograd.grad(s, all_attn_weights, retain_graph=False, create_graph=False, allow_unused=True)
        
        # Process each layer's gradient-weighted attention
        layer_contributions = []
        for i, (grad, attn_weights) in enumerate(zip(grads, all_attn_weights)):
            if grad is None:
                grad = torch.zeros_like(attn_weights)

            num_heads = blocks[start_layer + i].attn.num_heads
            
            # Reshape: [bsz*heads, N, N] -> [bsz, heads, N, N] if needed
            if grad.dim() == 3:
                grad = grad.view(bsz, num_heads, grad.shape[1], grad.shape[2])
                attn_weights = attn_weights.view(bsz, num_heads, attn_weights.shape[1], attn_weights.shape[2])
            # For SigLIP, it is already 4D [bsz, heads, N, N]
            
            # Apply ReLU to gradients
            grad = torch.clamp(grad, min=0)
            
            # Weight attention by gradients
            weighted_attn = grad * attn_weights  # [batch, heads, N, N]
            
            # Average over heads
            weighted_attn = weighted_attn.mean(dim=1)  # [batch, N, N]
            
            if is_timm:
                # SigLIP: No CLS token. Use max over query dimension for discriminative heatmaps.
                cls_to_patches = weighted_attn.max(dim=1).values  # [batch, N]
            else:
                # CLIP: Attention is [N+1, N+1]
                # Extract CLS to patches
                cls_to_patches = weighted_attn[:, 0, 1:]  # [batch, num_patches]
            
            layer_contributions.append(cls_to_patches)
        
        # Aggregate across layers (simple sum, as in reference)
        # Note: Full LRP would use more sophisticated propagation rules
        aggregated = sum(layer_contributions)  # [batch, num_patches]
        
        # Reshape to spatial grid
        num_patches = aggregated.shape[-1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size
        
        heatmap = aggregated.reshape(bsz, 1, h, w)
        
        # Upsample to image size
        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()


class AntiHallucinationObjective:
    """
    Optuna objective for anti-hallucination hyperparameter optimization.
    
    Evaluates both:
    - Correct prompts (maximize): prompt matches actual class
    - Wrong prompts (minimize): prompt is for a different class
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        preprocess,
        dataset_file,
        wnid_to_label,
        device,
        image_size=448,
        limit=0,
        num_negatives=1,
        negative_strategy='random',
        composite_lambda=0.5,
        gap_only=False,
        multi_objective=False,
        seed=42,
        use_gradcam=False,
        gradcam_layer=8,
        use_chefercam=False,
        chefercam_method='transformer_attribution',
        transformer_attribution_start_layer=1,
        threshold_mode='fixed',
        fixed_threshold=0.5,
        baseline_metrics=None,
        use_attentioncam=False,
        lrp_start_layer=1,
        use_daam=False,
        use_daam_keyspace_omp=False,
        daam_model_id='Manojb/stable-diffusion-2-base',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.num_negatives = num_negatives
        self.negative_strategy = negative_strategy
        self.composite_lambda = composite_lambda
        self.gap_only = gap_only
        self.multi_objective = multi_objective
        self.rng = random.Random(seed)
        self.use_gradcam = use_gradcam
        self.gradcam_layer = gradcam_layer
        self.use_chefercam = use_chefercam
        self.chefercam_method = chefercam_method
        self.transformer_attribution_start_layer = transformer_attribution_start_layer
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.baseline_metrics = baseline_metrics
        self.use_attentioncam = use_attentioncam
        self.lrp_start_layer = lrp_start_layer
        self.use_daam = use_daam
        self.use_daam_keyspace_omp = use_daam_keyspace_omp
        
        # Initialize DAAM segmenter if needed
        self.daam_segmenter = None
        if self.use_daam:
            if DAAMSegmenter is None:
                raise ImportError("Could not import DAAMSegmenter. Ensure scripts/daam_segmentation.py exists.")
            print(f"[optuna] Initializing DAAMSegmenter ({daam_model_id})...")
            self.daam_segmenter = DAAMSegmenter(model_id=daam_model_id, device=device)
        
        # Load dataset
        self.f = h5py.File(dataset_file, 'r')
        self.imgs_refs = self.f['value/img']
        self.gts_refs = self.f['value/gt']
        self.targets_refs = self.f['value/target']
        self.num_images = self.imgs_refs.shape[0]
        
        self.limit = limit if limit > 0 else self.num_images
        self.limit = min(self.limit, self.num_images)
        
        self.wnid_to_label = wnid_to_label
        
        # Precompute wnids and prompts
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        """Precompute all class embeddings once."""
        print("[optuna] Precomputing class embeddings...")
        
        wnids_in_seg = []
        for idx in range(self.num_images):
            target_ref = self.targets_refs[idx, 0]
            target_data = np.array(self.f[target_ref])
            wnid = ''.join([chr(c) for c in target_data.flatten()])
            wnids_in_seg.append(wnid)
        
        self.wnids_in_seg = wnids_in_seg
        self.unique_wnids = sorted(set(wnids_in_seg))
        
        self.wnid_to_prompt = {}
        self.wnid_to_classname = {}
        for wnid in self.unique_wnids:
            class_label = self.wnid_to_label.get(wnid)
            if class_label is None:
                class_label = get_synset_name(wnid)
            self.wnid_to_classname[wnid] = class_label
            self.wnid_to_prompt[wnid] = f"a photo of a {class_label}."
        
        all_prompts = [self.wnid_to_prompt[w] for w in self.unique_wnids]
        self.wnid_to_idx = {w: i for i, w in enumerate(self.unique_wnids)}
        self.idx_to_wnid = {i: w for w, i in self.wnid_to_idx.items()}
        
        tok_all = self.tokenizer(all_prompts).to(self.device)
        with torch.no_grad():
            self.all_text_embs = self.model.encode_text(tok_all, normalize=True)
        
        print(f"[optuna] Precomputed {len(all_prompts)} class embeddings.")
        print(f"[optuna] Negative strategy: {self.negative_strategy}, num_negatives: {self.num_negatives}")
    
    def _sample_negative_indices(self, correct_idx: int) -> list:
        """Sample negative class indices different from the correct class."""
        all_indices = list(range(len(self.unique_wnids)))
        all_indices.remove(correct_idx)
        
        if self.negative_strategy == 'distant':
            # Try to get semantically distant classes
            distant = get_distant_class_indices(correct_idx, self.unique_wnids, self.wnid_to_idx)
            # Filter to what's available in our dataset
            distant = [d for d in distant if d < len(self.unique_wnids) and d != correct_idx]
            if len(distant) >= self.num_negatives:
                return self.rng.sample(distant, self.num_negatives)
            # Fall back to random if not enough distant classes
            return self.rng.sample(all_indices, min(self.num_negatives, len(all_indices)))
        else:
            # Random sampling
            return self.rng.sample(all_indices, min(self.num_negatives, len(all_indices)))
    
    def evaluate_sparse_config(
        self,
        wn_use_synonyms: bool,
        wn_use_hypernyms: bool,
        wn_use_hyponyms: bool,
        wn_use_siblings: bool,
        dict_include_prompts: bool,
        sparse_threshold: float,
        atoms: int,
        max_dict_cos_sim: float,
        omp_beta: float = 1.0,
        show_progress: bool = False,
    ):
        """
        Evaluate a specific sparse configuration.
        Returns: (correct_miou, wrong_miou, correct_acc, wrong_acc, correct_map, wrong_map,
                  correct_auroc, wrong_auroc, correct_stats, wrong_stats)
        """
        correct_results = {
            'inter': np.zeros(2), 'union': np.zeros(2),
            'pixel_correct': 0, 'pixel_label': 0,
            'ap': [], 'auroc': [],
            'max': [], 'mean': [], 'median': [], 'min': []
        }
        wrong_results = {
            'inter': np.zeros(2), 'union': np.zeros(2),
            'pixel_correct': 0, 'pixel_label': 0,
            'ap': [], 'auroc': [],
            'max': [], 'mean': [], 'median': [], 'min': []
        }
        
        iterator = range(self.limit)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for idx in iterator:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for attempt in range(3):
                try:
                    # Load Image
                    img_ref = self.imgs_refs[idx, 0]
                    img_obj = np.array(self.f[img_ref])
                    img_np = img_obj.transpose(2, 1, 0)
                    base_img = Image.fromarray(img_np)
                
                    img_t = self.preprocess(base_img).unsqueeze(0).to(self.device)
                    H_feat, W_feat = img_t.shape[-2:]
                
                    # Load GT
                    gt_ref = self.gts_refs[idx, 0]
                    gt_wrapper = self.f[gt_ref]
                    if gt_wrapper.dtype == 'object':
                        real_gt_ref = gt_wrapper[0, 0]
                        real_gt = np.array(self.f[real_gt_ref])
                        gt_mask = real_gt.transpose(1, 0)
                    else:
                        gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                
                    # Resize GT
                    gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
                    target_resize = transforms.Resize(
                        (self.image_size, self.image_size),
                        interpolation=InterpolationMode.NEAREST,
                    )
                    gt_pil = target_resize(gt_pil)
                    gt_mask = np.array(gt_pil).astype(np.int32)
                    gt_tensor = torch.from_numpy(gt_mask).long()
                    H_gt, W_gt = gt_mask.shape
                
                    # Get class info
                    wnid = self.wnids_in_seg[idx]
                    class_name = self.wnid_to_classname[wnid]
                    prompt = self.wnid_to_prompt[wnid]
                    cls_idx = self.wnid_to_idx[wnid]
                    original_1x = self.all_text_embs[cls_idx:cls_idx + 1]
                
                    # Build sparse dictionary
                    def build_sparse_embedding(text_emb_1x, target_class_name):
                        """Build sparse residual embedding for a given text embedding."""
                        parts = []
                    
                        # 1) Other class prompts
                        emb_idx = None
                        for i, w in enumerate(self.unique_wnids):
                            if self.wnid_to_classname[w] == target_class_name:
                                emb_idx = i
                                break
                    
                        if dict_include_prompts and len(self.unique_wnids) > 1:
                            if emb_idx is not None:
                                if emb_idx > 0:
                                    parts.append(self.all_text_embs[:emb_idx])
                                if emb_idx + 1 < len(self.unique_wnids):
                                    parts.append(self.all_text_embs[emb_idx + 1:])
                            else:
                                parts.append(self.all_text_embs)
                    
                        # 2) WordNet neighbors
                        use_wn = any([wn_use_synonyms, wn_use_hypernyms, wn_use_hyponyms, wn_use_siblings])
                        if use_wn:
                            target_prompt = f"a photo of a {target_class_name}."
                            raw_neighbors = wordnet_neighbors_configured(
                                target_class_name,
                                use_synonyms=wn_use_synonyms,
                                use_hypernyms=wn_use_hypernyms,
                                use_hyponyms=wn_use_hyponyms,
                                use_siblings=wn_use_siblings,
                                limit_per_relation=8,
                            )
                            if raw_neighbors:
                                neighbor_prompts = [target_prompt.replace(target_class_name, w) for w in raw_neighbors]
                                n_tok = self.tokenizer(neighbor_prompts).to(self.device)
                                with torch.no_grad():
                                    n_emb = self.model.encode_text(n_tok)
                                    n_emb = F.normalize(n_emb, dim=-1)
                                parts.append(n_emb)
                    
                        # Combine dictionary
                        if len(parts) > 0:
                            D = torch.cat(parts, dim=0)
                            D = F.normalize(D, dim=-1)
                        
                            # Filter by cosine similarity
                            if 0.0 < max_dict_cos_sim < 1.0:
                                sim = (D @ text_emb_1x.t()).squeeze(-1).abs()
                                keep = sim < max_dict_cos_sim
                                D = D[keep]
                        else:
                            D = text_emb_1x.new_zeros((0, text_emb_1x.shape[-1]))
                    
                        # OMP sparse residual
                        sparse_1x = omp_sparse_residual(text_emb_1x, D, max_atoms=atoms)
                        return sparse_1x
                
                    def compute_metrics(text_emb_1x, target_class_name, competing_class_names=None, omp_beta=1.0):
                        """Compute heatmap and metrics for a given embedding."""
                        
                        method_name = 'legrad'
                        
                        # DAAM uses its own pipeline - doesn't need CLIP embeddings
                        if self.use_daam:
                            prompt_text = f"a photo of a {target_class_name}."
                            
                            # Build competing concepts from WordNet neighbors and other classes
                            # (same hyperparameters as CLIP-based methods for fair comparison)
                            all_competing = []
                            
                            # 1) Add other class names (if dict_include_prompts)
                            if dict_include_prompts and competing_class_names:
                                all_competing.extend(competing_class_names)
                            
                            # 2) Add WordNet neighbors
                            use_wn = any([wn_use_synonyms, wn_use_hypernyms, wn_use_hyponyms, wn_use_siblings])
                            if use_wn:
                                raw_neighbors = wordnet_neighbors_configured(
                                    target_class_name,
                                    use_synonyms=wn_use_synonyms,
                                    use_hypernyms=wn_use_hypernyms,
                                    use_hyponyms=wn_use_hyponyms,
                                    use_siblings=wn_use_siblings,
                                    limit_per_relation=8,
                                )
                                if raw_neighbors:
                                    # Filter out target class itself
                                    raw_neighbors = [n for n in raw_neighbors if n.lower() != target_class_name.lower()]
                                    all_competing.extend(raw_neighbors)
                            
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_competing = []
                            for c in all_competing:
                                c_lower = c.lower()
                                if c_lower not in seen and c_lower != target_class_name.lower():
                                    seen.add(c_lower)
                                    unique_competing.append(c)
                            
                            # Limit by atoms parameter (like limiting dictionary size in CLIP OMP)
                            if atoms > 0 and len(unique_competing) > atoms:
                                unique_competing = unique_competing[:atoms]
                            
                            # Use Key-Space OMP if we have competing concepts
                            if unique_competing:
                                if self.use_daam_keyspace_omp:
                                    # True key-space OMP: orthogonalize K vectors inside UNet
                                    heatmap = run_daam_with_key_space_omp(
                                        self.daam_segmenter,
                                        base_img,
                                        target_concept=target_class_name,
                                        competing_concepts=unique_competing,
                                        beta=omp_beta,
                                        size=512,
                                    )
                                else:
                                    # Post-hoc heatmap OMP: orthogonalize heatmaps after independent runs
                                    heatmap = self.daam_segmenter.predict_key_space_omp(
                                        base_img,
                                        prompt=prompt_text,
                                        target_concept=target_class_name,
                                        competing_concepts=unique_competing,
                                        size=512,
                                        omp_beta=omp_beta
                                    )
                            else:
                                # Fallback to basic DAAM without OMP (baseline)
                                heatmap = self.daam_segmenter.predict(base_img, prompt_text, size=512)
                            method_name = 'daam'
                        else:
                            # CLIP-based methods use sparse embeddings
                            sparse_1x = build_sparse_embedding(text_emb_1x, target_class_name)
                            
                            # Choose between LRP, CheferCAM, GradCAM and LeGrad (sparse)
                            if self.use_attentioncam:
                                heatmap = compute_lrp_heatmap(self.model, img_t, sparse_1x)
                                method_name = 'lrp'
                            elif self.use_chefercam:
                                if self.chefercam_method == 'transformer_attribution':
                                    heatmap = compute_transformer_attribution(
                                        self.model, img_t, sparse_1x, start_layer=self.transformer_attribution_start_layer
                                    )
                                else:
                                    heatmap = compute_chefercam(self.model, img_t, sparse_1x)
                                method_name = 'chefercam'
                            elif self.use_gradcam:
                                heatmap = compute_gradcam_heatmap(self.model, img_t, sparse_1x, layer_index=self.gradcam_layer)
                                method_name = 'gradcam'
                            else:
                                heatmap = compute_map_for_embedding(self.model, img_t, sparse_1x)
                                method_name = 'legrad'
                    
                        # Get actual heatmap dimensions
                        H_hm, W_hm = heatmap.shape[-2], heatmap.shape[-1]

                        # Resize heatmap to GT size FIRST (matching baseline)
                        heatmap_resized = F.interpolate(
                            heatmap.view(1, 1, H_hm, W_hm),
                            size=(H_gt, W_gt),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    
                        # Normalize based on method (matching baseline)
                        if method_name == 'legrad':
                             # LeGrad is already clamped [0,1], do not re-minmax normalize 
                             # (preserves absolute confidence)
                             heatmap_norm = heatmap_resized
                        else:
                             # GradCAM/CheferCAM need re-normalization after interpolation
                             heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                    
                        # For LeGrad, always use fixed threshold (not mean-based)
                        # This matches compute_positive_negative_baseline.py behavior
                        if method_name == 'legrad':
                            thr = sparse_threshold
                        elif self.threshold_mode == 'mean':
                            thr = heatmap_norm.mean().item()
                        else:
                            thr = sparse_threshold

                        # Create binary predictions
                        Res_1 = (heatmap_norm > thr).float()
                        Res_0 = (heatmap_norm <= thr).float()
                    
                        output = torch.stack([Res_0, Res_1], dim=0)
                        output_AP = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)
                    
                        # Metrics
                        correct, labeled = batch_pix_accuracy(output, gt_tensor)
                        inter, union = batch_intersection_union(output, gt_tensor, nclass=2)
                        ap = get_ap_scores(output_AP, gt_tensor)
                    
                        # Statistics
                        max_val = float(np.max(heatmap_norm.numpy()))
                        mean_val = float(np.mean(heatmap_norm.numpy()))
                        median_val = float(np.median(heatmap_norm.numpy()))
                        min_val = float(np.min(heatmap_norm.numpy()))
                    
                        # AUROC
                        gt_binary = (gt_mask > 0).astype(int).flatten()
                        pred_flat = heatmap_norm.numpy().flatten()
                        if len(np.unique(gt_binary)) > 1:
                             auroc = roc_auc_score(gt_binary, pred_flat)
                        else:
                             auroc = np.nan
                    
                        return inter, union, correct, labeled, ap[0] if ap else 0.0, auroc, max_val, mean_val, median_val, min_val
                
                    # === WRONG PROMPTS (sample first to get competing concepts for DAAM OMP) ===
                    neg_indices = self._sample_negative_indices(cls_idx)
                    neg_class_names = [self.wnid_to_classname[self.idx_to_wnid[ni]] for ni in neg_indices]
                    
                    # === CORRECT PROMPT ===
                    # For DAAM, pass wrong classes as potential competing concepts (filtered by hyperparameters)
                    inter_c, union_c, correct_c, label_c, ap_c, auroc_c, mx_c, mn_c, md_c, mi_c = compute_metrics(
                        original_1x, class_name, competing_class_names=neg_class_names, omp_beta=omp_beta
                    )
                    correct_results['inter'] = correct_results['inter'] + inter_c
                    correct_results['union'] = correct_results['union'] + union_c
                    correct_results['pixel_correct'] += correct_c
                    correct_results['pixel_label'] += label_c
                    correct_results['ap'].append(ap_c)
                    if not np.isnan(auroc_c):
                        correct_results['auroc'].append(auroc_c)
                    correct_results['max'].append(mx_c)
                    correct_results['mean'].append(mn_c)
                    correct_results['median'].append(md_c)
                    correct_results['min'].append(mi_c)
                
                    # === WRONG PROMPTS ===
                    for neg_idx in neg_indices:
                        neg_wnid = self.idx_to_wnid[neg_idx]
                        neg_class_name = self.wnid_to_classname[neg_wnid]
                        neg_emb = self.all_text_embs[neg_idx:neg_idx + 1]
                        
                        # For DAAM, use correct class as a competing concept for wrong prompts
                        wrong_competing = [class_name]
                    
                        inter_w, union_w, correct_w, label_w, ap_w, auroc_w, mx_w, mn_w, md_w, mi_w = compute_metrics(
                            neg_emb, neg_class_name, competing_class_names=wrong_competing, omp_beta=omp_beta
                        )
                        wrong_results['inter'] = wrong_results['inter'] + inter_w
                        wrong_results['union'] = wrong_results['union'] + union_w
                        wrong_results['pixel_correct'] += correct_w
                        wrong_results['pixel_label'] += label_w
                        wrong_results['ap'].append(ap_w)
                        if not np.isnan(auroc_w):
                            wrong_results['auroc'].append(auroc_w)
                        wrong_results['max'].append(mx_w)
                        wrong_results['mean'].append(mn_w)
                        wrong_results['median'].append(md_w)
                        wrong_results['min'].append(mi_w)
                
                    # Success, break attempt loop
                    break
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc; gc.collect()
                    if attempt < 2:
                        print(f"Retrying idx {idx} (attempt {attempt+1})... Error: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(1)
                        continue
                    print(f"Error processing idx {idx} (wnid: {self.wnids_in_seg[idx]}): {e}")
        # Compute averages (Global IoU / Pixel Acc)
        def compute_global_metrics(res_dict):
            # mIoU
            iou = res_dict['inter'].astype(np.float64) / (res_dict['union'].astype(np.float64) + 1e-10)
            miou = 100.0 * iou.mean()
            
            # Pixel Acc
            pix_acc = 100.0 * res_dict['pixel_correct'] / (res_dict['pixel_label'] + 1e-10)
            
            # mAP (mean of APs per image)
            map_score = np.mean(res_dict['ap']) * 100 if res_dict['ap'] else 0.0
            
            # AUROC
            auroc_score = np.mean(res_dict['auroc']) * 100 if res_dict['auroc'] else 0.0
            
            return miou, pix_acc, map_score, auroc_score

        # Correct
        correct_miou, correct_macc, correct_map, correct_auroc = compute_global_metrics(correct_results)
        
        # Wrong
        wrong_miou, wrong_macc, wrong_map, wrong_auroc = compute_global_metrics(wrong_results)
        
        # Statistics
        correct_stats = {
            'max': np.mean(correct_results['max']) if correct_results['max'] else 0.0,
            'mean': np.mean(correct_results['mean']) if correct_results['mean'] else 0.0,
            'median': np.mean(correct_results['median']) if correct_results['median'] else 0.0,
            'min': np.mean(correct_results['min']) if correct_results['min'] else 0.0,
            'n_samples': len(correct_results['ap']),
        }
        wrong_stats = {
            'max': np.mean(wrong_results['max']) if wrong_results['max'] else 0.0,
            'mean': np.mean(wrong_results['mean']) if wrong_results['mean'] else 0.0,
            'median': np.mean(wrong_results['median']) if wrong_results['median'] else 0.0,
            'min': np.mean(wrong_results['min']) if wrong_results['min'] else 0.0,
            'n_samples': len(wrong_results['ap']),
        }
        
        return (correct_miou, wrong_miou, correct_macc, wrong_macc, correct_map, wrong_map,
                correct_auroc, wrong_auroc, correct_stats, wrong_stats)
    
    def __call__(self, trial: optuna.Trial):
        """Optuna objective function."""
        
        # Sample hyperparameters
        wn_use_synonyms = trial.suggest_categorical('wn_use_synonyms', [True, False])
        wn_use_hypernyms = trial.suggest_categorical('wn_use_hypernyms', [True, False])
        wn_use_hyponyms = trial.suggest_categorical('wn_use_hyponyms', [True, False])
        wn_use_siblings = trial.suggest_categorical('wn_use_siblings', [True, False])
        dict_include_prompts = trial.suggest_categorical('dict_include_prompts', [True, False])
        
        # Constraint: At least one dictionary source must be used
        if not (wn_use_synonyms or wn_use_hypernyms or wn_use_hyponyms or wn_use_siblings or dict_include_prompts):
            raise optuna.TrialPruned()
        
        # For LeGrad, always search for best fixed threshold (not mean-based)
        # For GradCAM/CheferCAM/LRP/DAAM, use threshold_mode setting
        if self.threshold_mode == 'fixed' or (not self.use_gradcam and not self.use_chefercam and not self.use_attentioncam and not self.use_daam):
            sparse_threshold = trial.suggest_float('sparse_threshold', 0.1, 0.9, step=0.025)
        else:
            sparse_threshold = 0.5  # Ignored by adaptive thresholding for GradCAM/CheferCAM/LRP
        atoms = trial.suggest_int('atoms', 1, 32)
        max_dict_cos_sim = trial.suggest_float('max_dict_cos_sim', 0.5, 1.0, step=0.05)
        
        # DAAM-specific: omp_beta controls orthogonalization strength in Key-Space OMP
        if self.use_daam:
            omp_beta = trial.suggest_float('omp_beta', 0.0, 2.0, step=0.1)
        else:
            omp_beta = 1.0  # Not used for CLIP-based methods
        
        # Evaluate
        (correct_miou, wrong_miou, correct_acc, wrong_acc, correct_map, wrong_map,
         correct_auroc, wrong_auroc, correct_stats, wrong_stats) = self.evaluate_sparse_config(
            wn_use_synonyms=wn_use_synonyms,
            wn_use_hypernyms=wn_use_hypernyms,
            wn_use_hyponyms=wn_use_hyponyms,
            wn_use_siblings=wn_use_siblings,
            dict_include_prompts=dict_include_prompts,
            sparse_threshold=sparse_threshold,
            atoms=atoms,
            max_dict_cos_sim=max_dict_cos_sim,
            omp_beta=omp_beta,
            show_progress=False,
        )
        
        # Log all metrics
        trial.set_user_attr('correct_miou', correct_miou)
        trial.set_user_attr('wrong_miou', wrong_miou)
        trial.set_user_attr('correct_acc', correct_acc)
        trial.set_user_attr('wrong_acc', wrong_acc)
        trial.set_user_attr('correct_map', correct_map)
        trial.set_user_attr('wrong_map', wrong_map)
        trial.set_user_attr('correct_auroc', correct_auroc)
        trial.set_user_attr('wrong_auroc', wrong_auroc)
        trial.set_user_attr('correct_stats', correct_stats)
        trial.set_user_attr('wrong_stats', wrong_stats)
        
        # Compute multi-metric composite score for single-objective optimization
        # 
        # OBJECTIVE: Maximize the GAP between correct and wrong metrics.
        # 
        # The goal is NOT just to improve correct metrics, but to ensure:
        # 1. Correct metrics improve (or stay high)
        # 2. Wrong metrics degrade (or at least don't improve as much)
        # 3. The GAP (correct - wrong) increases compared to baseline
        #
        # GAP-based formulation:
        #   gap_baseline = baseline_correct - baseline_wrong
        #   gap_current = current_correct - current_wrong
        #   gap_improvement = gap_current - gap_baseline
        #
        # Additionally, we want to reward correct improvements and penalize wrong improvements.
        # 
        # Final composite = gap_improvement + lambda * correct_improvement
        #
        # This ensures:
        # - If both improve equally: gap_improvement ≈ 0, only correct_improvement matters
        # - If correct improves more than wrong: gap_improvement > 0 (good!)
        # - If wrong improves more than correct: gap_improvement < 0 (bad!)
        # - If wrong degrades: gap_improvement > correct_improvement (very good!)
        
        if self.baseline_metrics:
            base_c = self.baseline_metrics['correct']
            base_w = self.baseline_metrics['wrong']
            
            # === GAP-BASED METRICS ===
            # Baseline gaps (how much better was correct than wrong before)
            gap_baseline_miou = base_c['miou'] - base_w['miou']
            gap_baseline_acc = base_c['acc'] - base_w['acc']
            gap_baseline_map = base_c['map'] - base_w['map']
            
            # Current gaps (how much better is correct than wrong now)
            gap_current_miou = correct_miou - wrong_miou
            gap_current_acc = correct_acc - wrong_acc
            gap_current_map = correct_map - wrong_map
            
            # Gap improvements (positive = gap increased = good!)
            gap_improvement_miou = gap_current_miou - gap_baseline_miou
            gap_improvement_acc = gap_current_acc - gap_baseline_acc
            gap_improvement_map = gap_current_map - gap_baseline_map
            
            # Correct improvements (for secondary objective)
            d_c_miou = correct_miou - base_c['miou']
            d_c_acc = correct_acc - base_c['acc']
            d_c_map = correct_map - base_c['map']
            
            # Wrong changes (positive = degraded = good!)
            d_w_miou = base_w['miou'] - wrong_miou
            d_w_acc = base_w['acc'] - wrong_acc
            d_w_map = base_w['map'] - wrong_map
            
            # Store all metrics for analysis
            trial.set_user_attr('delta_correct_miou', d_c_miou)
            trial.set_user_attr('delta_correct_acc', d_c_acc)
            trial.set_user_attr('delta_correct_map', d_c_map)
            
            trial.set_user_attr('delta_wrong_miou', d_w_miou)
            trial.set_user_attr('delta_wrong_acc', d_w_acc)
            trial.set_user_attr('delta_wrong_map', d_w_map)
            
            # AUROC deltas (not optimized, but tracked for analysis)
            d_c_auroc = correct_auroc - base_c.get('auroc', correct_auroc)
            d_w_auroc = base_w.get('auroc', wrong_auroc) - wrong_auroc
            trial.set_user_attr('delta_correct_auroc', d_c_auroc)
            trial.set_user_attr('delta_wrong_auroc', d_w_auroc)
            
            trial.set_user_attr('gap_improvement_miou', gap_improvement_miou)
            trial.set_user_attr('gap_improvement_acc', gap_improvement_acc)
            trial.set_user_attr('gap_improvement_map', gap_improvement_map)
            
            trial.set_user_attr('gap_baseline_miou', gap_baseline_miou)
            trial.set_user_attr('gap_current_miou', gap_current_miou)
            
            # === COMPOSITE SCORE ===
            # === COMPOSITE SCORE ===
            # User Requested Default: (correct_miou + correct_accuracy + correct_map) - (wrong_miou + wrong_accuracy + wrong_map)
            # This is directly: correct - wrong.
            
            # Note: We calculate this directly from current metrics, regardless of baseline.
            # This replaces the previous logic that included baseline comparisons and degradation bonuses.
            
            composite_miou = correct_miou - wrong_miou
            composite_acc = correct_acc - wrong_acc
            composite_map = correct_map - wrong_map
            
            # If user wants the detailed legacy modes (Gap Improvement or Composite with Degradation), 
            # they can be re-enabled here, but we default to the simple formula.
            if self.gap_only:
                # Use improvement over baseline instead of raw gap
                composite_miou = gap_improvement_miou
                composite_acc = gap_improvement_acc
                composite_map = gap_improvement_map
            elif self.composite_lambda != 0.5:
                # Only if user explicitly changed lambda (assuming 0.5 was default), we might consider the old formula
                # For now, we strictly follow the instruction to make the new formula default.
                # But to preserve the option for manual lambda tuning:
                composite_miou = gap_improvement_miou + self.composite_lambda * d_w_miou
                composite_acc = gap_improvement_acc + self.composite_lambda * d_w_acc
                composite_map = gap_improvement_map + self.composite_lambda * d_w_map
            
        else:
            # Fallback: direct gap (correct - wrong)
            # Used when no baseline metrics are provided
            composite_miou = correct_miou - wrong_miou
            composite_acc = correct_acc - wrong_acc
            composite_map = correct_map - wrong_map

        # Sum all composite scores (each weighted equally)
        # AUROC excluded from optimization
        composite = composite_miou + composite_acc + composite_map
        trial.set_user_attr('composite_score', composite)
        trial.set_user_attr('composite_miou', composite_miou)
        trial.set_user_attr('composite_acc', composite_acc)
        trial.set_user_attr('composite_map', composite_map)
        # trial.set_user_attr('composite_auroc', composite_auroc) # AUROC removed from composite
        
        # Report for pruning
        trial.report(composite, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if self.multi_objective:
            # Return tuple for multi-objective: correct/wrong metrics (excluding AUROC)
            # Directions: maximize correct, minimize wrong
            return (correct_miou, correct_acc, correct_map,
                    wrong_miou, wrong_acc, wrong_map)
        else:
            # Return composite score for single-objective
            return composite


def main():
    parser = argparse.ArgumentParser(description='Anti-Hallucination Hyperparameter Optimization')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_siglip', action='store_true',
                        help='Use SigLIP instead of CLIP for optimization')
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Anti-hallucination settings
    parser.add_argument('--num_negatives', type=int, default=1, 
                        help='Number of negative (wrong) prompts per image')
    parser.add_argument('--negative_strategy', type=str, default='random', 
                        choices=['random', 'distant'],
                        help='Strategy for sampling negative prompts')
    parser.add_argument('--composite_lambda', type=float, default=0.5,
                        help='Weight for wrong-degradation bonus (gap_improvement + λ×wrong_degradation)')
    parser.add_argument('--gap_only', action='store_true',
                        help='Pure gap optimization: maximize (correct - wrong) directly, ignore correct improvement')
    parser.add_argument('--multi_objective', action='store_true',
                        help='Use multi-objective Pareto optimization instead of composite score')
    
    # Baseline comparison (from compute_legrad_negative_baseline.py)
    parser.add_argument('--baseline_json', type=str, default=None,
                        help='Path to baseline JSON from compute_legrad_negative_baseline.py')
    parser.add_argument('--baseline_correct_miou', type=float, default=None,
                        help='Manual baseline correct mIoU (overrides --baseline_json)')
    parser.add_argument('--baseline_wrong_miou', type=float, default=None,
                        help='Manual baseline wrong mIoU (overrides --baseline_json)')
    
    # GradCAM settings
    parser.add_argument('--use_gradcam', action='store_true',
                        help='Use GradCAM instead of LeGrad for optimization')
    parser.add_argument('--gradcam_layer', type=int, default=8,
                        help='GradCAM layer index (default: 8)')
    parser.add_argument('--use_chefercam', action='store_true',
                        help='Use CheferCAM (attention GradCAM) instead of LeGrad')
    
    # LRP settings
    parser.add_argument('--use_attentioncam', action='store_true',
                        help='Use AttentionCAM instead of LeGrad')
    parser.add_argument('--lrp_start_layer', type=int, default=1,
                        help='Start layer for LRP attribution (default: 1)')
    
    # DAAM settings
    parser.add_argument('--use_daam', action='store_true',
                        help='Use DAAM (Stable Diffusion Attention) instead of LeGrad')
    parser.add_argument('--daam_model_id', type=str, default='Manojb/stable-diffusion-2-base',
                        help='Stable Diffusion model ID for DAAM')
    parser.add_argument('--use_daam_keyspace_omp', action='store_true',
                        help='Use true key-space OMP (orthogonalize K vectors in UNet) instead of post-hoc heatmap OMP')
    
    # Threshold settings
    parser.add_argument('--threshold_mode', type=str, default='fixed',
                        choices=['mean', 'fixed'],
                        help='Thresholding mode: "mean" (adaptive) or "fixed"')
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='Fixed threshold value when threshold_mode=fixed')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='anti_hallucination_optimization')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--output_json', type=str, default='anti_hallucination_results.json')
    
    args = parser.parse_args()
    
    # Determine model type and method
    if args.use_daam:
        # DAAM uses its own Stable Diffusion pipeline, independent of CLIP/SigLIP
        model_type_key = 'DAAM'
        method_key = 'DAAM'
    else:
        if args.use_siglip:
            model_type_key = 'SigLIP'
        else:
            model_type_key = 'CLIP'
            
        # Determine method
        if args.use_attentioncam:
            method_key = 'AttentionCAM'
        elif args.use_chefercam:
            method_key = 'CheferCAM'
        elif args.use_gradcam:
            method_key = 'GradCAM'
        else:
            method_key = 'LeGrad'
        
    # Get baseline metrics
    baseline_metrics = BASELINES[model_type_key][method_key]
    
    print(f"Using baseline for {model_type_key} - {method_key}:")
    print(f"  Correct: mIoU={baseline_metrics['correct']['miou']}, Acc={baseline_metrics['correct']['acc']}, mAP={baseline_metrics['correct']['map']}")
    print(f"  Wrong:   mIoU={baseline_metrics['wrong']['miou']}, Acc={baseline_metrics['wrong']['acc']}, mAP={baseline_metrics['wrong']['map']}")

    # Load baseline if provided (override hardcoded if JSON given, though user said use hardcoded)
    if args.baseline_json and os.path.exists(args.baseline_json):
        print(f"Loading baseline from {args.baseline_json}...")
        # ... logic to override if needed, but for now we stick to requested plan
        pass
    
    # Calculate baseline composite (conceptually 0 improvement)
    # But we can calculate what the raw score would have been
    # baseline_composite = baseline_correct_miou - args.composite_lambda * (baseline_wrong_miou or 0)
    # In relative mode, 0 is the baseline. Positive is improvement.
    
    # Set model defaults based on --use_siglip
    if args.use_siglip:
        if args.model_name is None:
            args.model_name = 'ViT-B-16-SigLIP'
        if args.pretrained is None:
            args.pretrained = 'webli'
        model_type = 'SigLIP'
    else:
        if args.model_name is None:
            args.model_name = 'ViT-B-16'
        if args.pretrained is None:
            args.pretrained = 'laion2b_s34b_b88k'
        model_type = 'CLIP'
    
    # Load model
    print(f"Loading {model_type} model: {args.model_name} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    
    # Load NLTK data
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass
    
    # Load ImageNet class index
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"[imagenet] Loaded class index with {len(wnid_to_label)} wnids.")
    except Exception as e:
        print(f"[imagenet] Warning: failed to load class index ({e})")
        wnid_to_label = {}
    
    # Create objective
    objective = AntiHallucinationObjective(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        dataset_file=args.mat_file,
        wnid_to_label=wnid_to_label,
        device=args.device,
        image_size=args.image_size,
        limit=args.limit,
        num_negatives=args.num_negatives,
        negative_strategy=args.negative_strategy,
        composite_lambda=args.composite_lambda,
        gap_only=args.gap_only,
        multi_objective=args.multi_objective,
        seed=args.seed,
        use_gradcam=args.use_gradcam,
        gradcam_layer=args.gradcam_layer,
        use_chefercam=args.use_chefercam,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        baseline_metrics=baseline_metrics,
        use_attentioncam=args.use_attentioncam,
        lrp_start_layer=args.lrp_start_layer,
        use_daam=args.use_daam,
        use_daam_keyspace_omp=args.use_daam_keyspace_omp,
        daam_model_id=args.daam_model_id,
    )
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
    if args.multi_objective:
        # Multi-objective: maximize all correct metrics, minimize all wrong metrics (excluding AUROC)
        directions = ['maximize', 'maximize', 'maximize',  # correct: mIoU, Acc, mAP
                      'minimize', 'minimize', 'minimize']  # wrong: mIoU, Acc, mAP
        if args.storage:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                directions=directions,
                sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
            )
        else:
            study = optuna.create_study(
                study_name=args.study_name,
                directions=directions,
                sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
            )
        print(f"\n{'='*60}")
        print("Multi-Objective Optimization Mode (6 objectives)")
        print("MAXIMIZE: Correct mIoU, Accuracy, mAP")
        print("MINIMIZE: Wrong mIoU, Accuracy, mAP")
        print("(AUROC is calculated and printed but not optimized)")
        print(f"{'='*60}\n")
    else:
        # Single-objective with composite score
        if args.storage:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                study_name=args.study_name,
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
            )
        print(f"\n{'='*60}")
        if args.gap_only:
            print("PURE GAP OPTIMIZATION MODE")
            print("Score = Σ(gap_improvement)")
            print("Where: gap_improvement = (correct - wrong) - (baseline_correct - baseline_wrong)")
            print("Goal: Maximize GAP between correct and wrong metrics ONLY")
            print("(Ignores whether correct metrics improve, only cares about gap)")
        elif args.composite_lambda != 0.5:
            print("GAP + WRONG DEGRADATION Optimization")
            print(f"Score = Σ(gap_improvement + {args.composite_lambda} × wrong_degradation)")
            print("Where: gap_improvement = (correct - wrong) - (baseline_correct - baseline_wrong)")
            print("       wrong_degradation = baseline_wrong - current_wrong (positive when wrong ↓)")
            print("Goal: Maximize GAP AND reward configurations that reduce wrong metrics")
        else:
            print("SIMPLE ANTI-HALLUCINATION OPTIMIZATION (Default)")
            print("Score = Σ(correct - wrong)")
            print("Formula: (correct_miou + correct_acc + correct_map) - (wrong_miou + wrong_acc + wrong_map)")
            print("Goal: Maximize the raw gap between correct and wrong metrics.")
        print("(AUROC is calculated and printed but not optimized)")
        print(f"{'='*60}\n")
    
    # Custom callback for detailed logging
    def trial_callback(study, trial):
        """Log detailed metrics after each trial."""
        if trial.state != TrialState.COMPLETE:
            return
        
        # Raw metrics
        c_miou = trial.user_attrs.get('correct_miou', 0)
        w_miou = trial.user_attrs.get('wrong_miou', 0)
        c_acc = trial.user_attrs.get('correct_acc', 0)
        w_acc = trial.user_attrs.get('wrong_acc', 0)
        c_map = trial.user_attrs.get('correct_map', 0)
        w_map = trial.user_attrs.get('wrong_map', 0)
        c_auroc = trial.user_attrs.get('correct_auroc', 0)
        w_auroc = trial.user_attrs.get('wrong_auroc', 0)
        
        # Deltas from baseline
        dc_miou = trial.user_attrs.get('delta_correct_miou', 0)
        dc_acc = trial.user_attrs.get('delta_correct_acc', 0)
        dc_map = trial.user_attrs.get('delta_correct_map', 0)
        dc_auroc = trial.user_attrs.get('delta_correct_auroc', 0)
        dw_miou = trial.user_attrs.get('delta_wrong_miou', 0)
        dw_acc = trial.user_attrs.get('delta_wrong_acc', 0)
        dw_map = trial.user_attrs.get('delta_wrong_map', 0)
        dw_auroc = trial.user_attrs.get('delta_wrong_auroc', 0)
        
        composite = trial.user_attrs.get('composite_score', trial.value)
        
        # Print full metrics table
        print(f"\n{'─'*70}")
        print(f"  Trial {trial.number} │ Score={composite:.2f}")
        print(f"{'─'*70}")
        print(f"  {'':>10} {'mIoU':>8} {'Acc':>8} {'mAP':>8} {'AUROC':>8}")
        print(f"  {'Correct':>10} {c_miou:>7.1f}% {c_acc:>7.1f}% {c_map:>7.1f}% {c_auroc:>7.1f}%")
        print(f"  {'  Δbase':>10} {dc_miou:>+7.1f}  {dc_acc:>+7.1f}  {dc_map:>+7.1f}  {dc_auroc:>+7.1f}")
        print(f"  {'Wrong':>10} {w_miou:>7.1f}% {w_acc:>7.1f}% {w_map:>7.1f}% {w_auroc:>7.1f}%")
        print(f"  {'  Δbase':>10} {dw_miou:>+7.1f}  {dw_acc:>+7.1f}  {dw_map:>+7.1f}  {dw_auroc:>+7.1f}")
        print(f"  {'Gap(C-W)':>10} {c_miou-w_miou:>+7.1f}  {c_acc-w_acc:>+7.1f}  {c_map-w_map:>+7.1f}  {c_auroc-w_auroc:>+7.1f}")
        
        # Best trial summary (for single-objective)
        if not args.multi_objective:
            best = study.best_trial
            b_comp = best.user_attrs.get('composite_score', best.value)
            print(f"  Best: Trial {best.number} (Score={b_comp:.2f})")
        print(f"{'─'*70}")
    
    # Run optimization
    if args.use_daam:
        method_name = "DAAM + Key-Space OMP"
    elif args.use_attentioncam:
        method_name = "AttentionCAM"
    elif args.use_chefercam:
        method_name = "CheferCAM"
    elif args.use_gradcam:
        method_name = "GradCAM"
    else:
        method_name = "LeGrad"

    print(f"Starting Optuna optimization with {args.n_trials} trials")
    if args.use_daam:
        print(f"Model: {args.daam_model_id} (Stable Diffusion)")
    else:
        print(f"Model: {args.model_name} ({args.pretrained}) [{model_type}]")
    print(f"Method: {method_name}")
    if args.use_daam:
        print(f"DAAM model: {args.daam_model_id}")
        print(f"Tuning: WordNet neighbors, omp_beta, atoms, sparse_threshold")
    elif args.use_attentioncam:
        print(f"AttentionCAM start layer: {args.lrp_start_layer}")
    elif args.use_gradcam:
        print(f"GradCAM layer: {args.gradcam_layer}")
    print(f"Negative strategy: {args.negative_strategy}, Num negatives: {args.num_negatives}")
    
    # Enforce n_jobs=1 for DAAM because 'trace' context manager is not thread-safe
    if args.use_daam and args.n_jobs > 1:
        print(f"[WARNING] DAAM is not thread-safe due to model hooks. Forcing n_jobs=1 (was {args.n_jobs}).")
        args.n_jobs = 1
        
    print(f"Composite Lambda: {args.composite_lambda}")
    print(f"Baseline: {model_type_key} - {method_key}")
    print(f"{'='*60}\n")
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        callbacks=[trial_callback],
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    if args.multi_objective:
        # Multi-objective results: show Pareto front
        pareto_trials = study.best_trials
        print(f"\nPareto-optimal trials: {len(pareto_trials)}")
        
        print(f"\n{'='*60}")
        print("Pareto Front (Best Trade-offs):")
        print(f"{'='*60}")
        
        for i, trial in enumerate(pareto_trials[:10], 1):
            vals = trial.values
            c_auroc = trial.user_attrs.get('correct_auroc', 0.0)
            w_auroc = trial.user_attrs.get('wrong_auroc', 0.0)
            print(f"\n#{i} Trial {trial.number}:")
            print(f"  Correct: mIoU={vals[0]:.2f} | Acc={vals[1]:.2f} | mAP={vals[2]:.2f} | AUROC={c_auroc:.2f}")
            print(f"  Wrong:   mIoU={vals[3]:.2f} | Acc={vals[4]:.2f} | mAP={vals[5]:.2f} | AUROC={w_auroc:.2f}")
            print(f"  Params: {trial.params}")
        
        results = {
            'mode': 'multi_objective',
            'n_trials': len(study.trials),
            'pareto_trials': [
                {
                    'trial_number': t.number,
                    'correct': {'miou': t.values[0], 'acc': t.values[1], 'map': t.values[2], 'auroc': t.user_attrs.get('correct_auroc')},
                    'wrong': {'miou': t.values[3], 'acc': t.values[4], 'map': t.values[5], 'auroc': t.user_attrs.get('wrong_auroc')},
                    'params': t.params,
                }
                for t in pareto_trials
            ],
        }
    else:
        # Single-objective results
        best_trial = study.best_trial
        
        # Extract all metrics from best trial
        correct_metrics = {
            'miou': best_trial.user_attrs.get('correct_miou', 0),
            'acc': best_trial.user_attrs.get('correct_acc', 0),
            'map': best_trial.user_attrs.get('correct_map', 0),
            'auroc': best_trial.user_attrs.get('correct_auroc', 0),
        }
        wrong_metrics = {
            'miou': best_trial.user_attrs.get('wrong_miou', 0),
            'acc': best_trial.user_attrs.get('wrong_acc', 0),
            'map': best_trial.user_attrs.get('wrong_map', 0),
            'auroc': best_trial.user_attrs.get('wrong_auroc', 0),
        }
        correct_stats = best_trial.user_attrs.get('correct_stats', {})
        wrong_stats = best_trial.user_attrs.get('wrong_stats', {})
        
        print(f"\n{'='*60}")
        print("BEST TRIAL RESULTS")
        print(f"{'='*60}")
        print(f"\nComposite Score: {best_trial.value:.2f}")
        
        # Get gap metrics
        gap_baseline_miou = best_trial.user_attrs.get('gap_baseline_miou', 0)
        gap_current_miou = best_trial.user_attrs.get('gap_current_miou', 0)
        gap_improvement_miou = best_trial.user_attrs.get('gap_improvement_miou', 0)
        gap_improvement_acc = best_trial.user_attrs.get('gap_improvement_acc', 0)
        gap_improvement_map = best_trial.user_attrs.get('gap_improvement_map', 0)
        
        delta_c_miou = best_trial.user_attrs.get('delta_correct_miou', 0)
        delta_c_acc = best_trial.user_attrs.get('delta_correct_acc', 0)
        delta_c_map = best_trial.user_attrs.get('delta_correct_map', 0)
        
        delta_w_miou = best_trial.user_attrs.get('delta_wrong_miou', 0)
        delta_w_acc = best_trial.user_attrs.get('delta_wrong_acc', 0)
        delta_w_map = best_trial.user_attrs.get('delta_wrong_map', 0)
        
        print(f"\n=== GAP ANALYSIS (Key Anti-Hallucination Metric) ===")
        print(f"  Baseline Gap (C-W):  mIoU={gap_baseline_miou:.2f}")
        print(f"  Current Gap (C-W):   mIoU={gap_current_miou:.2f}")
        print(f"  Gap Improvement:     mIoU={gap_improvement_miou:+.2f}  Acc={gap_improvement_acc:+.2f}  mAP={gap_improvement_map:+.2f}")
        print(f"  (Positive = gap increased = better class discrimination)")
        
        print(f"\n=== CORRECT PROMPTS (image class = text prompt class) ===")
        print(f"  mIoU:     {correct_metrics['miou']:.2f}  (Δ={delta_c_miou:+.2f})")
        print(f"  Accuracy: {correct_metrics['acc']:.2f}  (Δ={delta_c_acc:+.2f})")
        print(f"  mAP:      {correct_metrics['map']:.2f}  (Δ={delta_c_map:+.2f})")
        print(f"  AUROC:    {correct_metrics['auroc']:.2f}")
        if correct_stats:
            print(f"  Max Val:  {correct_stats.get('max', 0):.4f}")
            print(f"  Mean Val: {correct_stats.get('mean', 0):.4f}")
            print(f"  Median:   {correct_stats.get('median', 0):.4f}")
            print(f"  Min Val:  {correct_stats.get('min', 0):.4f}")
            print(f"  Samples:  {correct_stats.get('n_samples', 0)}")
        
        print(f"\n=== WRONG PROMPTS (image class ≠ text prompt class) ===")
        print(f"  mIoU:     {wrong_metrics['miou']:.2f}  (Δ={-delta_w_miou:+.2f})")  # Flip sign for intuition
        print(f"  Accuracy: {wrong_metrics['acc']:.2f}  (Δ={-delta_w_acc:+.2f})")
        print(f"  mAP:      {wrong_metrics['map']:.2f}  (Δ={-delta_w_map:+.2f})")
        print(f"  AUROC:    {wrong_metrics['auroc']:.2f}")
        print(f"  (For 'wrong', positive Δ means metrics increased = BAD)")
        if wrong_stats:
            print(f"  Max Val:  {wrong_stats.get('max', 0):.4f}")
            print(f"  Mean Val: {wrong_stats.get('mean', 0):.4f}")
            print(f"  Median:   {wrong_stats.get('median', 0):.4f}")
            print(f"  Min Val:  {wrong_stats.get('min', 0):.4f}")
            print(f"  Samples:  {wrong_stats.get('n_samples', 0)}")
        
        print(f"\n=== COMPOSITE BREAKDOWN ===")
        if args.gap_only:
            print(f"  mIoU:     {best_trial.user_attrs.get('composite_miou', 0):.2f}  (pure gap_improvement)")
        elif args.composite_lambda != 0.5:
            print(f"  mIoU:     {best_trial.user_attrs.get('composite_miou', 0):.2f}  (gap_impr + λ×wrong_degradation)")
            print(f"           gap_impr={gap_improvement_miou:+.2f}, wrong_degr={delta_w_miou:+.2f}")
        else:
            print(f"  mIoU:     {best_trial.user_attrs.get('composite_miou', 0):.2f}  (correct - wrong)")
        print(f"  Accuracy: {best_trial.user_attrs.get('composite_acc', 0):.2f}")
        print(f"  mAP:      {best_trial.user_attrs.get('composite_map', 0):.2f}")
        
        print(f"\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Top 5 by composite score
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
        
        print(f"\n{'='*60}")
        print("Top 5 trials by composite score:")
        print(f"{'='*60}")
        
        for i, trial in enumerate(sorted_trials, 1):
            print(f"\n#{i} Trial {trial.number}:")
            print(f"  Composite: {trial.value:.2f}")
            print(f"  Correct: mIoU={trial.user_attrs.get('correct_miou', 0):.1f} | "
                  f"Acc={trial.user_attrs.get('correct_acc', 0):.1f} | "
                  f"mAP={trial.user_attrs.get('correct_map', 0):.1f} | "
                  f"AUROC={trial.user_attrs.get('correct_auroc', 0):.1f}")
            print(f"  Wrong:   mIoU={trial.user_attrs.get('wrong_miou', 0):.1f} | "
                  f"Acc={trial.user_attrs.get('wrong_acc', 0):.1f} | "
                  f"mAP={trial.user_attrs.get('wrong_map', 0):.1f} | "
                  f"AUROC={trial.user_attrs.get('wrong_auroc', 0):.1f}")
        
        results = {
            'mode': 'gap_only' if args.gap_only else 'gap_based_composite',
            'gap_only': args.gap_only,
            'composite_lambda': args.composite_lambda,
            'best_composite_score': best_trial.value,
            'best_gap_analysis': {
                'gap_improvement_miou': gap_improvement_miou,
                'gap_improvement_acc': gap_improvement_acc,
                'gap_improvement_map': gap_improvement_map,
                'gap_baseline_miou': gap_baseline_miou,
                'gap_current_miou': gap_current_miou,
            },
            'best_correct': correct_metrics,
            'best_wrong': wrong_metrics,
            'best_deltas': {
                'correct_miou': delta_c_miou,
                'correct_acc': delta_c_acc,
                'correct_map': delta_c_map,
                'wrong_miou': -delta_w_miou,  # Store as actual change (positive = increased)
                'wrong_acc': -delta_w_acc,
                'wrong_map': -delta_w_map,
            },
            'best_correct_stats': correct_stats,
            'best_wrong_stats': wrong_stats,
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'model_settings': {
                'model_name': args.model_name,
                'pretrained': args.pretrained,
                'model_type': model_type,
                'use_siglip': args.use_siglip,
            },
            'baseline_metrics': baseline_metrics,
            'top_5_trials': [
                {
                    'trial_number': t.number,
                    'composite_score': t.value,
                    'gap_improvement_miou': t.user_attrs.get('gap_improvement_miou'),
                    'correct': {
                        'miou': t.user_attrs.get('correct_miou'),
                        'acc': t.user_attrs.get('correct_acc'),
                        'map': t.user_attrs.get('correct_map'),
                        'auroc': t.user_attrs.get('correct_auroc'),
                    },
                    'wrong': {
                        'miou': t.user_attrs.get('wrong_miou'),
                        'acc': t.user_attrs.get('wrong_acc'),
                        'map': t.user_attrs.get('wrong_map'),
                        'auroc': t.user_attrs.get('wrong_auroc'),
                    },
                    'params': t.params,
                }
                for t in sorted_trials
            ],
        }
        
    try:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")
    except (PermissionError, OSError) as e:
        print(f"\n[Warning] Failed to write to {args.output_json}: {e}")
        # Fallback to current working directory
        filename = os.path.basename(args.output_json)
        fallback_path = os.path.join(os.getcwd(), filename)
        
        # If the original path was already just a filename in CWD, we might need a new name
        if os.path.abspath(args.output_json) == os.path.abspath(fallback_path):
             fallback_path = os.path.join(os.getcwd(), f"fallback_{filename}")

        print(f"Attempting fallback to: {fallback_path}")
        try:
            with open(fallback_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {fallback_path}")
        except Exception as e2:
            print(f"[Error] Failed to save results to fallback path: {e2}")


if __name__ == '__main__':
    main()
