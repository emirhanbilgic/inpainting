#!/usr/bin/env python3
"""
Benchmark Segmentation Script (v2)

This script evaluates segmentation methods on the ImageNet-Segmentation dataset
(gtsegs_ijcv.mat). It includes:
  - LeGrad (our method) - original
  - GradCAM - gradcam
  - CheferCAM (GradCAM on attention) - chefercam

Usage:
  python scripts/benchmark_segmentation_v2.py --mat_file scripts/data/gtsegs_ijcv.mat --methods original,gradcam,chefercam
"""

import sys
import os
import argparse
import math
import json
import time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import requests
import warnings
import copy
from typing import List, Dict, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

from legrad import LeWrapper, LePreprocess
import open_clip

# Import sparse encoding helpers
try:
    from sparse_encoding import (
        omp_sparse_residual,
        wordnet_neighbors_configured,
    )
except ImportError:
    # If running from root without proper path setup, try relative import or just warn
    print("Warning: Could not import sparse_encoding. Sparse LeGrad may fail.")
    omp_sparse_residual = None
    wordnet_neighbors_configured = None




# Constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)

try:
    from daam_segmentation import DAAMSegmenter
except ImportError as e:
    print(f"Warning: Failed to import DAAMSegmenter. Error: {e}")
    DAAMSegmenter = None

# DAAM Imports
try:
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention_processor import Attention
    from daam import trace as daam_trace
    from daam.trace import DiffusionHeatMapHooker, UNetCrossAttentionHooker
    from daam.hook import UNetCrossAttentionLocator
    from daam.heatmap import RawHeatMapCollection
    from daam.utils import auto_autocast
except ImportError:
    print("Warning: Failed to import DAAM/Diffusers dependencies.")









# ==============================================================================
# Float32 DAAM Segmenter (for MPS/CPU compatibility)
# ==============================================================================
class Float32DAAMSegmenter(DAAMSegmenter):
    """DAAMSegmenter that uses float32 to avoid NaN on MPS and mixed-precision errors."""
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
            if hasattr(torch.mps, 'empty_cache') and self.device == 'mps':
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                heatmap = torch.zeros((h, w)).to(self.device)

        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.cpu()

# ==============================================================================
# True Key-Space OMP Components (from daam_omp_comparison.py)
# ==============================================================================

class KeySpaceOMPProcessor:
    """
    Custom attention processor that orthogonalizes the target token's key vector
    against distractor token keys in the cross-attention layers of the UNet.
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
        
        # For each target token index
        for target_idx in self.target_token_indices:
            target_key = key[:, target_idx, :]
            
            # Collect all distractor key vectors
            for dist_indices in self.distractor_token_indices:
                for dist_idx in dist_indices:
                    dist_key = key[:, dist_idx, :]
                    
                    # Normalize distractor key
                    dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # Project target onto distractor and subtract
                    projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                    target_key = target_key - self.beta * projection
            
            # Write back orthogonalized key
            key[:, target_idx, :] = target_key
        
        return key
    
    @torch.no_grad()
    def _unravel_attn(self, x):
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
    for i in range(len(tokens)):
        if tokens[i:i+len(concept_tokens)] == concept_tokens:
            # +1 because of the SOS token
            indices.extend(range(i + 1, i + 1 + len(concept_tokens)))
            break
    
    if not indices:
        for i, tok in enumerate(tokens):
            for ct in concept_tokens:
                if ct in tok or tok in ct:
                    indices.append(i + 1)
        
    return indices


def run_daam_with_key_space_omp(
    segmenter: DAAMSegmenter,
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
    
    img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = img_arr * 2.0 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
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


def compute_chefercam(model, image, text_emb_1x):
    """
    Computes GradCAM on the last Attention layer (attn_gradcam baseline).
    
    For SigLIP models, per the LeGrad paper (Section A.4), CheferCAM treats the
    attentional pooler as a "decoder transformer" and applies GradCAM on the
    POOLER's attention maps, not the trunk's last self-attention layer.
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: attn_gradcam - GradCAM applied to attention maps from the last layer
    """
    import math
    from open_clip.timm_model import TimmModel
    
    model.zero_grad()
    num_prompts = text_emb_1x.shape[0]
    
    # Determine model type
    if isinstance(model.visual, TimmModel):
        blocks = list(model.visual.trunk.blocks)
        is_timm = True
    else:
        blocks = list(model.visual.transformer.resblocks)
        is_timm = False
    
    with torch.enable_grad():
        if is_timm:
            # --- SigLIP: GradCAM on attentional pooler ---
            pooler = model.visual.trunk.attn_pool
            blocks = list(model.visual.trunk.blocks)
            
            # Trunk Forward Pass (all blocks normally)
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
            
            # Attentional Pooler Manual Forward (capture pooler attention)
            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)
            
            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = pooler.q_norm(q), pooler.k_norm(k)
            
            attn_weights = (q * pooler.scale) @ k.transpose(-2, -1)
            attn_weights = attn_weights.softmax(dim=-1)  # [B, heads, num_latent, N]
            attn_weights.requires_grad_(True)
            
            x_pool = (attn_weights @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
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
            # --- CLIP ---
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            batch_size = x.shape[0]
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
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
            
            for i in range(len(blocks) - 1):
                x = blocks[i](x)
            
            last_block = blocks[-1]
            last_attn = last_block.attn
            x_normed = last_block.ln_1(x)
            
            qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, embed_dim = q.shape
            num_heads = last_attn.num_heads
            head_dim = embed_dim // num_heads
            
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights.requires_grad_(True)
            
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = last_attn.out_proj(attn_output)
            
            x = x + attn_output
            x = x + last_block.mlp(last_block.ln_2(x))
            
            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Gradient w.r.t. attention weights
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False, allow_unused=True)[0]
        
        if grad is None:
            grad = torch.zeros_like(attn_weights)
        
        if is_timm:
            # SigLIP pooler attention: already [B, heads, num_latent, N]
            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)[:, 0]  # [B, N]
            num_patches = cam.shape[-1]
        else:
            # CLIP: reshape [bsz*heads, N, N] -> [bsz, heads, N, N]
            seq_len = attn_weights.shape[1]
            num_heads = blocks[-1].attn.num_heads
            grad = grad.view(bsz, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)
            cam = cam[:, 0, 1:]  # [batch, num_patches]
            num_patches = cam.shape[-1]
        
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size
        
        heatmap = cam.reshape(bsz, 1, h, w)
        
        heatmap = F.interpolate(
            heatmap, 
            size=image.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
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
    """
    import math
    from open_clip.timm_model import TimmModel
    
    model.zero_grad()
    
    # Determine model type
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
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H*W, C)
            else:
                B, _, C = x.shape
                
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
                    all_attn_weights.append(attn_weights)  # [B, heads, N, N]
                    
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
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            batch_size = x.shape[0]
            bsz = batch_size
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
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
            
            for i, block in enumerate(blocks):
                if i < start_layer:
                    x = block(x)
                else:
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
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    attn_weights.requires_grad_(True)
                    all_attn_weights.append(attn_weights)
                    
                    attn_output = torch.bmm(attn_weights, v)
                    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
                    attn_output = attn_module.out_proj(attn_output)
                    
                    x = x + attn_output
                    x = x + block.mlp(block.ln_2(x))
            
            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradients for all attention layers
        grads = torch.autograd.grad(s, all_attn_weights, retain_graph=False, create_graph=False)
        
        # Process each layer's gradient-weighted attention
        layer_contributions = []
        for i, (grad, attn_weights) in enumerate(zip(grads, all_attn_weights)):
            num_heads = blocks[start_layer + i].attn.num_heads
            
            # Reshape if needed: [bsz*heads, N, N] -> [bsz, heads, N, N]
            if grad.dim() == 3:
                grad = grad.view(bsz, num_heads, grad.shape[1], grad.shape[2])
                attn_weights = attn_weights.view(bsz, num_heads, attn_weights.shape[1], attn_weights.shape[2])
            # For SigLIP, already 4D [bsz, heads, N, N]
            
            grad = torch.clamp(grad, min=0)
            weighted_attn = grad * attn_weights
            weighted_attn = weighted_attn.mean(dim=1)  # [batch, N, N]
            
            if is_timm:
                # SigLIP: No CLS token. Use max over query dimension.
                cls_to_patches = weighted_attn.max(dim=1).values  # [batch, N]
            else:
                cls_to_patches = weighted_attn[:, 0, 1:]  # [batch, num_patches]
            
            layer_contributions.append(cls_to_patches)
        
        aggregated = sum(layer_contributions)  # [batch, num_patches]
        
        num_patches = aggregated.shape[-1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size
        
        heatmap = aggregated.reshape(bsz, 1, h, w)
        
        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()


# =============================================================================
# Reference Metrics Implementation
# Adapted from: https://github.com/hila-chefer/Transformer-Explainability
# File: utils/metrices.py
# =============================================================================

def batch_pix_accuracy(predict, target):
    """
    Batch Pixel Accuracy (reference implementation)
    
    Args:
        predict: input 3D tensor [C, H, W] with class probabilities
        target: label 2D tensor [H, W] with class indices
    
    Returns:
        pixel_correct, pixel_labeled
    """
    _, predict = torch.max(predict, 0)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass=2):
    """
    Batch Intersection of Union (reference implementation)
    
    Args:
        predict: input 3D tensor [C, H, W] with class probabilities
        target: label 2D tensor [H, W] with class indices
        nclass: number of categories (default 2 for binary segmentation)
    
    Returns:
        area_inter, area_union (arrays of length nclass)
    """
    _, predict = torch.max(predict, 0)
    mini = 1
    maxi = nclass
    nbins = nclass
    
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    
    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    
    return area_inter, area_union


def get_ap_scores(predict, target, ignore_index=-1):
    """
    Compute Average Precision (reference implementation)
    
    Args:
        predict: Prediction tensor [C, H, W]
        target: Target tensor [H, W]
    
    Returns:
        List of AP scores
    """
    total = []
    target_expand = target.unsqueeze(0).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    
    # One-hot encoding
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(0).clamp(min=0).long()
    target_1hot = x.scatter_(0, t, 1)
    
    predict_flat = predict.data.cpu().numpy().reshape(-1)
    target_flat = target_1hot.data.cpu().numpy().reshape(-1)
    
    p = predict_flat[target_expand_numpy != ignore_index]
    t_filtered = target_flat[target_expand_numpy != ignore_index]
    
    ap = np.nan_to_num(average_precision_score(t_filtered, p))
    total.append(ap)
    
    return total


# =============================================================================
# Helper Functions
# =============================================================================

def download_imagenet_class_index(dst_path: str):
    """Download the standard ImageNet class index JSON."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(dst_path, "w") as f:
        json.dump(data, f)
    return data


def load_imagenet_class_index(path: str):
    """Load (or download) the ImageNet class index JSON."""
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    print(f"[imagenet] class index not found at {path}, downloading...")
    return download_imagenet_class_index(path)


def build_wnid_to_label_map(class_index):
    """Convert imagenet_class_index mapping into wnid -> primary label (lowercase)."""
    wnid_to_label = {}
    for _, (wnid, label_str) in class_index.items():
        primary = label_str.split(",")[0].replace("_", " ").strip()
        wnid_to_label[wnid] = primary.lower()
    return wnid_to_label


def get_synset_name(wnid):
    """Fallback: derive a name from WordNet if JSON mapping is unavailable."""
    if wn is None:
        return wnid
    try:
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        name = synset.lemmas()[0].name().replace('_', ' ')
        return name
    except Exception:
        return wnid


def compute_gradcam_for_embedding(model, image, text_emb_1x, layer_index: int = 8):
    """Compute a GradCAM heatmap (normalized to [0, 1]) for a single text embedding."""
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def compute_legrad_for_embedding(model, image, text_emb_1x):
    """Compute LeGrad heatmap for a single text embedding."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


def build_sparse_embedding_v2(
    target_wnid, 
    target_text_emb_1x, 
    all_text_embs, 
    unique_wnids, 
    wnid_to_label, 
    tokenizer, 
    model, 
    args
):
    """
    Build sparse residual embedding similar to optimize_anti_hallucination.py.
    """
    if omp_sparse_residual is None:
        raise ImportError("sparse_encoding module not found, cannot build sparse embedding.")

    parts = []
    
    # 1) Other class prompts
    # Find index of target in the global list
    try:
        emb_idx = unique_wnids.index(target_wnid)
    except ValueError:
        emb_idx = None

    if args.dict_include_prompts:
        if len(unique_wnids) > 1 and emb_idx is not None:
             if emb_idx > 0:
                 parts.append(all_text_embs[:emb_idx])
             if emb_idx + 1 < len(unique_wnids):
                 parts.append(all_text_embs[emb_idx + 1:])
        elif emb_idx is None:
             # Just use all if target not in list (shouldn't happen in standard bench)
             parts.append(all_text_embs)

    # 2) WordNet neighbors
    use_wn = any([args.wn_use_synonyms, args.wn_use_hypernyms, args.wn_use_hyponyms, args.wn_use_siblings])
    if use_wn:
        # Get class name
        target_class_name = wnid_to_label.get(target_wnid) or get_synset_name(target_wnid)
        
        target_prompt = f"a photo of a {target_class_name}."
        raw_neighbors = wordnet_neighbors_configured(
            target_class_name,
            use_synonyms=args.wn_use_synonyms,
            use_hypernyms=args.wn_use_hypernyms,
            use_hyponyms=args.wn_use_hyponyms,
            use_siblings=args.wn_use_siblings,
            limit_per_relation=8,
        )
        if raw_neighbors:
            neighbor_prompts = [target_prompt.replace(target_class_name, w) for w in raw_neighbors]
            n_tok = tokenizer(neighbor_prompts).to(args.device)
            with torch.no_grad():
                n_emb = model.encode_text(n_tok)
                n_emb = F.normalize(n_emb, dim=-1)
            parts.append(n_emb)

    # Combine dictionary
    d_words = []
    if len(parts) > 0:
        D = torch.cat(parts, dim=0)
        D = F.normalize(D, dim=-1)

        # Filter by cosine similarity
        # TODO: Need to filter associated words if we filter D
        # For now, we only collect d_words for sparse_residual call if dictionary is used
        # Re-implementing parts of sparse_encoding.py logic here to get words
        
        # Re-collect words to pass back
        if args.dict_include_prompts:
            if len(unique_wnids) > 1 and emb_idx is not None:
                 if emb_idx > 0:
                     for i in range(emb_idx):
                         d_words.append(wnid_to_label.get(unique_wnids[i]) or get_synset_name(unique_wnids[i]))
                 if emb_idx + 1 < len(unique_wnids):
                     for i in range(emb_idx + 1, len(unique_wnids)):
                         d_words.append(wnid_to_label.get(unique_wnids[i]) or get_synset_name(unique_wnids[i]))
        
        if use_wn and raw_neighbors:
             d_words.extend(raw_neighbors)

        if 0.0 < args.max_dict_cos_sim < 1.0:
            sim = (D @ target_text_emb_1x.t()).squeeze(-1).abs()
            keep = sim < args.max_dict_cos_sim
            D = D[keep]
            # Filter d_words
            d_words = [d_words[i] for i in range(len(d_words)) if keep[i]]
            
    else:
        D = target_text_emb_1x.new_zeros((0, target_text_emb_1x.shape[-1]))

    # OMP sparse residual
    if D.shape[0] == 0:
        # Fallback if dictionary is empty
        return target_text_emb_1x, []

    # sparse_1x = omp_sparse_residual(target_text_emb_1x, D, max_atoms=args.atoms)
    # Use return_indices=True to get indices of selected atoms
    sparse_1x, selected_indices = omp_sparse_residual(target_text_emb_1x, D, max_atoms=args.atoms, return_indices=True)
    
    # Get selected words
    selected_words = [d_words[i] for i in selected_indices if i < len(d_words)]
    
    return sparse_1x, selected_words




# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Segmentation (v2)'
    )
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for evaluation (224 for standard, 448 for higher res)')
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Method selection
    parser.add_argument(
        '--methods', type=str, default='original,gradcam,chefercam,daam,daam_omp',
        help="Comma-separated methods: original (LeGrad), gradcam, chefercam, daam, daam_omp"
    )
    
    # CheferCAM/Transformer Attribution settings
    parser.add_argument('--chefercam_method', type=str, default='transformer_attribution',
                        choices=['transformer_attribution', 'attn_gradcam'],
                        help='CheferCAM method: transformer_attribution (full LRP-based, default) or attn_gradcam (baseline)')
    parser.add_argument('--transformer_attribution_start_layer', type=int, default=1,
                        help='Start layer for transformer attribution (default 1, as in reference)')
    
    # Threshold settings
    parser.add_argument('--threshold_mode', type=str, default='mean',
                        choices=['mean', 'fixed'],
                        help='Thresholding mode: "mean" (per-image adaptive as in reference) or "fixed"')
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='Fixed threshold value when threshold_mode=fixed')
    
    # Visualization
    parser.add_argument('--vis_first_k', type=int, default=0)
    parser.add_argument('--vis_output_dir', type=str, default='outputs/segmentation_vis_v2')

    # Sparse / LeGrad optimization parameters
    parser.add_argument('--wn_use_synonyms', type=int, default=0)
    parser.add_argument('--wn_use_hypernyms', type=int, default=0)
    parser.add_argument('--wn_use_hyponyms', type=int, default=0)
    parser.add_argument('--wn_use_siblings', type=int, default=0)
    parser.add_argument('--dict_include_prompts', type=int, default=1)
    parser.add_argument('--atoms', type=int, default=8)
    parser.add_argument('--max_dict_cos_sim', type=float, default=1.0)

    
    args = parser.parse_args()

    # Parse methods
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]

    allowed_methods = {'original', 'gradcam', 'chefercam', 'daam', 'daam_omp'}
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods. Use --methods with: original,gradcam,chefercam,daam")
    
    print(f"Methods to evaluate: {methods}")
    print(f"Threshold mode: {args.threshold_mode}")
    if args.threshold_mode == 'fixed':
        print(f"Fixed threshold: {args.fixed_threshold}")
    
    # Load Model
    print(f"Loading model {args.model_name} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

    # Load DAAM if needed
    daam_segmenter = None
    if 'daam' in methods:
        if DAAMSegmenter is None:
            raise ImportError("DAAMSegmenter could not be imported. Please install daam and diffusers.")
        print("Initializing DAAM Segmenter (Float32)...")
        daam_segmenter = Float32DAAMSegmenter(device=args.device)

    if 'daam_omp' in methods:
        if DAAMSegmenter is None:
             raise ImportError("DAAMSegmenter needed for DAAM OMP.")
        if daam_segmenter is None:
             print("Initializing DAAM Segmenter (Float32) for OMP...")
             daam_segmenter = Float32DAAMSegmenter(device=args.device)

    # Load ImageNet mapping
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"[imagenet] Loaded class index with {len(wnid_to_label)} wnids.")
    except Exception as e:
        print(f"[imagenet] Warning: failed to load class index ({e})")
        wnid_to_label = {}

    # Open dataset
    print(f"Opening dataset {args.mat_file}...")
    mat_path = args.mat_file
    if not os.path.exists(mat_path):
        # Fallback for Kaggle
        kaggle_path = "/kaggle/input/mat-data/gtsegs_ijcv.mat"
        if os.path.exists(kaggle_path):
            print(f"Dataset not found at {mat_path}, using fallback: {kaggle_path}")
            mat_path = kaggle_path
        else:
             print(f"Warning: Dataset not found at {mat_path} and fallback {kaggle_path} also missing.")

    try:
        f = h5py.File(mat_path, 'r')
        imgs_refs = f['value/img']
        gts_refs = f['value/gt']
        targets_refs = f['value/target']
        num_images = imgs_refs.shape[0]
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    limit = args.limit if args.limit > 0 else num_images
    limit = min(limit, num_images)
    print(f"Processing {limit} images...")

    if args.vis_first_k > 0:
        os.makedirs(args.vis_output_dir, exist_ok=True)

    # Precompute prompts
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = ''.join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)
    unique_wnids = sorted(set(wnids_in_seg))

    wnid_to_prompt = {}
    for wnid in unique_wnids:
        class_label = wnid_to_label.get(wnid) or get_synset_name(wnid)
        # Reference uses simpler prompt: "a {label}"
        wnid_to_prompt[wnid] = f"a photo of a {class_label}."

    all_prompts = [wnid_to_prompt[w] for w in unique_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}

    print(f"[prompts] Built {len(all_prompts)} unique class prompts.")
    tok_all = tokenizer(all_prompts).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)

    # Results accumulators (using reference protocol: accumulated inter/union)
    # Initialize as numpy arrays for 2-class (bg, fg) accumulation
    total_inter = {m: np.zeros(2, dtype=np.int64) for m in methods}
    total_union = {m: np.zeros(2, dtype=np.int64) for m in methods}
    total_correct = {m: np.int64(0) for m in methods}
    total_label = {m: np.int64(0) for m in methods}
    total_ap = {m: [] for m in methods}

    for idx in tqdm(range(limit)):
        try:
            # Load Image
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            H_feat, W_feat = img_t.shape[-2:]
            
            # Load GT
            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0, 0]
                real_gt = np.array(f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)

            # Resize GT to match image_size
            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            target_resize = transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            gt_pil = target_resize(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.int32)
            gt_tensor = torch.from_numpy(gt_mask).long()
            H_gt, W_gt = gt_mask.shape
            
            # Get class embedding
            target_ref = targets_refs[idx, 0]
            target_data = np.array(f[target_ref])
            wnid = ''.join([chr(c) for c in target_data.flatten()])
            cls_idx = wnid_to_idx[wnid]
            text_emb_1x = all_text_embs[cls_idx:cls_idx + 1]
            
            heatmaps = {}
            
            # --- ORIGINAL (LeGrad) ---
            if 'original' in methods:
                 # Check if we should use sparse embedding
                use_sparse = (
                    args.wn_use_synonyms or args.wn_use_hypernyms or
                    args.wn_use_hyponyms or args.wn_use_siblings or
                    not args.dict_include_prompts or
                    args.atoms != 8 or args.max_dict_cos_sim < 0.99
                )
                
                final_emb = text_emb_1x
                if use_sparse:
                     final_emb, _ = build_sparse_embedding_v2(
                        target_wnid=wnid,
                        target_text_emb_1x=text_emb_1x,
                        all_text_embs=all_text_embs,
                        unique_wnids=unique_wnids,
                        wnid_to_label=wnid_to_label,
                        tokenizer=tokenizer,
                        model=model,
                        args=args
                     )
                
                heatmap = compute_legrad_for_embedding(model, img_t, final_emb)
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['original'] = heatmap_resized
            
            # --- GRADCAM ---
            if 'gradcam' in methods:
                heatmap = compute_gradcam_for_embedding(model, img_t, text_emb_1x)
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['gradcam'] = heatmap_resized
            
            # --- CHEFERCAM / TRANSFORMER ATTRIBUTION ---
            # Reference: hila-chefer/Transformer-Explainability
            # transformer_attribution: Full LRP-based method (main method, gets reported results)
            # attn_gradcam: Baseline GradCAM on attention maps
            if 'chefercam' in methods:
                if args.chefercam_method == 'transformer_attribution':
                    # Use full Transformer Attribution method (the one that gets reported results)
                    heatmap = compute_transformer_attribution(
                        model, img_t, text_emb_1x, 
                        start_layer=args.transformer_attribution_start_layer
                    )
                else:
                    # Use baseline attn_gradcam method
                    heatmap = compute_chefercam(model, img_t, text_emb_1x)
                
                # Already at image resolution, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['chefercam'] = heatmap_resized
            
            # --- DAAM ---
            if 'daam' in methods and daam_segmenter is not None:
                # Use the class prompt for DAAM
                # DAAMSegmenter handles the forward pass and tracing
                heatmap = daam_segmenter.predict(base_img, wnid_to_prompt[wnid])
                
                # Resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['daam'] = heatmap_resized

            # --- DAAM OMP ---
            if 'daam_omp' in methods and daam_segmenter is not None:
                # 1. Get distractor concepts using sparse encoding logic
                _, selected_words = build_sparse_embedding_v2(
                    target_wnid=wnid,
                    target_text_emb_1x=text_emb_1x,
                    all_text_embs=all_text_embs,
                    unique_wnids=unique_wnids,
                    wnid_to_label=wnid_to_label,
                    tokenizer=tokenizer,
                    model=model,
                    args=args
                )
                
                target_concept = wnid_to_label.get(wnid) or get_synset_name(wnid)
                
                # 2. Run Key-Space OMP
                heatmap = run_daam_with_key_space_omp(
                    daam_segmenter,
                    base_img,
                    target_concept=target_concept,
                    competing_concepts=selected_words,
                    beta=1.0,
                    size=512
                )
                
                # 3. Resize to GT
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['daam_omp'] = heatmap_resized

            
            # --- Compute Metrics (Reference Protocol) ---
            for method in methods:
                heatmap = heatmaps[method]

                # For LeGrad (original), use the already normalized [0,1] values directly
                # For other methods, normalize as per reference protocol
                if method == 'original':
                    heatmap_norm = heatmap  # Already in [0,1] from compute_legrad_for_embedding
                else:
                    # Normalize heatmap
                    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                # Determine threshold
                # LeGrad uses fixed 0.5 threshold as in the original implementation
                if method == 'original':
                    thr = 0.5
                elif args.threshold_mode == 'mean':
                    thr = heatmap_norm.mean().item()
                else:
                    thr = args.fixed_threshold
                
                # Create binary predictions
                Res_1 = (heatmap_norm > thr).float()
                Res_0 = (heatmap_norm <= thr).float()
                
                # Stack for 2-class output [2, H, W]
                output = torch.stack([Res_0, Res_1], dim=0)
                output_AP = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)
                
                # Compute metrics
                correct, labeled = batch_pix_accuracy(output, gt_tensor)
                inter, union = batch_intersection_union(output, gt_tensor, nclass=2)
                ap = get_ap_scores(output_AP, gt_tensor)
                
                total_correct[method] += correct
                total_label[method] += labeled
                total_inter[method] = total_inter[method] + inter
                total_union[method] = total_union[method] + union
                total_ap[method].extend(ap)
            
            # --- Visualization ---
            if idx < args.vis_first_k:
                vis_img = base_img.resize((W_gt, H_gt))
                num_cols = 2 + len(methods)
                fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
                
                axes[0].imshow(vis_img)
                axes[0].set_title('Image')
                axes[0].axis('off')
                
                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title('GT')
                axes[1].axis('off')
                
                for i, method in enumerate(methods):
                    heatmap = heatmaps[method].numpy()
                    # For LeGrad (original), use the already normalized [0,1] values directly
                    if method == 'original':
                        heatmap_norm = heatmap  # Already in [0,1]
                        thr = 0.5  # Fixed threshold for LeGrad
                    else:
                        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        if args.threshold_mode == 'mean':
                            thr = heatmap_norm.mean()
                        else:
                            thr = args.fixed_threshold
                    binary = (heatmap_norm > thr).astype(np.uint8)
                    axes[2 + i].imshow(binary, cmap='gray')
                    axes[2 + i].set_title(f'{method.capitalize()}')
                    axes[2 + i].axis('off')
                
                plt.tight_layout()
                out_path = os.path.join(args.vis_output_dir, f"seg_vis_{idx:04d}.png")
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[vis] Saved {out_path}")
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final Results
    print("\n" + "=" * 60)
    print("RESULTS (Reference Evaluation Protocol)")
    print("=" * 60)
    
    for method in methods:
        pixAcc = 100.0 * total_correct[method] / (total_label[method] + 1e-10)
        IoU = total_inter[method].astype(np.float64) / (total_union[method].astype(np.float64) + 1e-10)
        mIoU = 100.0 * IoU.mean()
        mAP = 100.0 * np.mean(total_ap[method])
        
        print(f"\n{method.upper()}:")
        print(f"  Pixel Accuracy: {pixAcc:.2f}%")
        print(f"  mIoU: {mIoU:.2f}")
        print(f"  mAP: {mAP:.2f}")
        print(f"  IoU per class: bg={100*IoU[0]:.2f}, fg={100*IoU[1]:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

