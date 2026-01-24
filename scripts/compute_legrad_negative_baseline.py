#!/usr/bin/env python3
"""
Compute LeGrad baseline metrics for both CORRECT and WRONG (negative) prompts.

This script evaluates the standard LeGrad method (without sparse encoding) on:
1. CORRECT prompts: text prompt matches the actual class in the image
2. WRONG prompts: text prompt is for a randomly sampled different class

The goal is to establish baseline values for:
- How well LeGrad segments with the correct class prompt
- How much LeGrad "hallucinates" with wrong class prompts

These baseline values can then be used with optimize_anti_hallucination.py
to find sparse encoding parameters that improve upon the baseline.

Usage:
    # Basic usage
    python scripts/compute_legrad_negative_baseline.py \
        --mat_file /path/to/gtsegs_ijcv.mat \
        --limit 100

    # With specific number of negatives
    python scripts/compute_legrad_negative_baseline.py \
        --mat_file /path/to/gtsegs_ijcv.mat \
        --limit 1000 \
        --num_negatives 1

Output JSON format:
{
    "correct": {"miou": ..., "acc": ..., "map": ...},
    "wrong": {"miou": ..., "acc": ..., "map": ...},
    "composite": ...,  # correct_miou - lambda * wrong_miou
    "settings": {...}
}
"""

import sys
import os
import argparse
import json
import random
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
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

from legrad import LeWrapper, LePreprocess
import open_clip

# Import functions from benchmark_segmentation
from benchmark_segmentation import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    compute_iou_acc,
    compute_map_score,
)


def compute_legrad_heatmap(model, image, text_emb_1x):
    """Compute LeGrad heatmap for a single text embedding."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


def compute_gradcam_heatmap(model, image, text_emb_1x, layer_index: int = -1):
    """
    Compute GradCAM heatmap for a single text embedding.
    
    This follows the official Chefer implementation approach:
    - Extracts attention from CLS to patches at specified layer
    - Applies Global Average Pooled gradients as weights
    - Averages over heads, then clamps
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
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
    
    # Handle negative layer index
    if layer_index < 0:
        layer_index = len(blocks) + layer_index
    layer_index = max(0, min(layer_index, len(blocks) - 1))
    
    target_block = blocks[layer_index]
    target_attn = target_block.attn
    num_heads = target_attn.num_heads
    num_prompts = text_emb_1x.shape[0]
    
    with torch.enable_grad():
        if is_timm:
            # --- SigLIP Forward Pass ---
            x = model.visual.trunk.patch_embed(image)
            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed
            
            B, N, C = x.shape
            attn_weights = None
            
            for i, block in enumerate(blocks):
                if i == layer_index:
                    # Manual attention for target layer
                    x_normed = block.norm1(x)
                    attn = block.attn
                    qkv = attn.qkv(x_normed).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = attn.q_norm(q), attn.k_norm(k)
                    
                    attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
                    attn_weights = attn_weights.softmax(dim=-1)  # [B, heads, N, N]
                    
                    attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
                    attn_out = attn.proj(attn_out)
                    attn_out = attn.proj_drop(attn_out)
                    x = x + attn_out
                    x = x + block.mlp(block.norm2(x))
                else:
                    x = block(x)
            
            # Attentional Pooler
            pooler = model.visual.trunk.attn_pool
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
            
            pooled_feat = x_pool[:, 0] if pooler.pool == 'token' else x_pool.mean(1)
            image_features = F.normalize(pooled_feat, dim=-1)
            
            # For SigLIP, use mean attention across all patches
            cam = attn_weights[0].mean(dim=2)  # [heads, N] - average over query positions
            grid_size = int(math.sqrt(N))
            
        else:
            # --- CLIP Forward Pass ---
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            
            batch_size = x.shape[0]
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            num_patches = x.shape[1] - 1
            pos_embed = getattr(model.visual, 'original_pos_embed', model.visual.positional_embedding)
            
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
            attn_weights = None
            
            for i, block in enumerate(blocks):
                if i == layer_index:
                    x_normed = block.ln_1(x)
                    attn = block.attn
                    qkv = F.linear(x_normed, attn.in_proj_weight, attn.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                    
                    seq_len, bsz, embed_dim = q.shape
                    head_dim = embed_dim // num_heads
                    
                    q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                    
                    scale = float(head_dim) ** -0.5
                    attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
                    attn_weights = F.softmax(attn_weights, dim=-1)
                    attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
                    
                    attn_out = torch.bmm(attn_weights.reshape(bsz*num_heads, seq_len, seq_len), v)
                    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
                    attn_out = attn.out_proj(attn_out)
                    
                    x = x + attn_out
                    x = x + block.mlp(block.ln_2(x))
                else:
                    x = block(x)
            
            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
            
            # Extract CLS -> patches attention
            cam = attn_weights[0, :, 0, 1:]  # [heads, num_patches]
            grid_size = int(math.sqrt(cam.shape[-1]))
        
        # Compute similarity and gradients
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False)[0]
        
        if is_timm:
            grad_cam = grad[0].mean(dim=2)  # [heads, N]
        else:
            grad_cam = grad[0, :, 0, 1:]    # [heads, num_patches]
        
        # Reshape to spatial grid: [heads, H, W]
        cam = cam.reshape(-1, grid_size, grid_size)
        grad_cam = grad_cam.reshape(-1, grid_size, grid_size)
        
        # Official Chefer method: GAP over spatial dims, then weight
        grad_gap = grad_cam.mean(dim=[1, 2], keepdim=True)  # [heads, 1, 1]
        cam = (cam * grad_gap).mean(dim=0).clamp(min=0)     # [H, W]
        
        # FIRST normalization (before interpolation, as in official generate_cam_attn)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        heatmap = cam.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        # SECOND normalization (after interpolation, as in official eval_batch)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()


def batch_intersection_union(predict, target, nclass=2):
    """
    Batch Intersection of Union (reference implementation)
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


def compute_chefercam_heatmap(model, image, text_emb_1x):
    """
    Computes GradCAM on the last Attention layer (CheferCAM/attn_gradcam baseline).
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: attn_gradcam - GradCAM applied to attention maps from the last layer
    """
    import math
    from open_clip.timm_model import TimmModel
    
    model.zero_grad()
    
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
    
    last_block = blocks[-1]
    last_attn = last_block.attn
    num_heads = last_attn.num_heads
    num_prompts = text_emb_1x.shape[0]
    
    with torch.enable_grad():
        if is_timm:
            # --- SigLIP (Attentional Pooler) ---
            pooler = model.visual.trunk.attn_pool
            blocks = list(model.visual.trunk.blocks)
            
            # Trunk Forward Pass
            x = model.visual.trunk.patch_embed(image)
            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed
            
            for block in blocks:
                x = block(x)
            
            # --- Attentional Pooler Manual Forward ---
            B, N, C = x.shape
            
            # 1. Pos embed (if any)
            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)
            
            # 2. Compute Q, K, V
            # q from latent
            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            
            # k, v from x
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            
            # Norm
            q, k = pooler.q_norm(q), pooler.k_norm(k)
            
            # Attention
            q = q * pooler.scale
            attn_weights = q @ k.transpose(-2, -1)
            attn_weights = attn_weights.softmax(dim=-1) # [B, num_heads, latent_len, N]
            
            # Output
            x_pool = attn_weights @ v # [B, heads, latent_len, head_dim]
            x_pool = x_pool.transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))
            
            # Pooling
            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0] # fallback
                
            image_features = F.normalize(pooled_feat, dim=-1)
           
        else:
            # --- CLIP Forward Pass ---
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
            
            # Manual attention for last block
            x_normed = last_block.ln_1(x)
            
            qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, embed_dim = q.shape
            head_dim = embed_dim // num_heads
            
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, N, N]
            
            # Convert to [B, heads, N, N] for consistency
            attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            
            # Compute attention output
            attn_output = torch.bmm(attn_weights.reshape(bsz*num_heads, seq_len, seq_len), v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = last_attn.out_proj(attn_output)
            
            # Continue
            x = x + attn_output
            x = x + last_block.mlp(last_block.ln_2(x))
            
            # Final features
            x = x.permute(1, 0, 2)  # [B, N+1, C]
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)
        
        # --- Official Chefer GradCAM Calculation ---
        # Reference: generate_cam_attn in ViT_explanation_generator.py
        #
        #   grad = self.model.blocks[-1].attn.get_attn_gradients()
        #   cam = self.model.blocks[-1].attn.get_attention_map()
        #   cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)   # [heads, H, W] - CLS to patches
        #   grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        #   grad = grad.mean(dim=[1, 2], keepdim=True)   # GAP over spatial dims -> 1 scalar/head
        #   cam = (cam * grad).mean(0).clamp(min=0)      # weight, average heads, then clamp
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)  # [1, 1]
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradient w.r.t. attention weights
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False)[0]
        # grad shape: [B, heads, N, N] for CLIP or [B, heads, latent_len, N] for SigLIP
        
        if is_timm:
            # SigLIP (Attentional Pooler)
            # attn_weights shape: [B, heads, latent_len, N]
            # Average over latent queries to get spatial importance  
            cam = attn_weights[0].mean(dim=1)  # [heads, N] - average over latent queries
            grad_cam = grad[0].mean(dim=1)     # [heads, N]
            
            num_patches = cam.shape[-1]
            grid_size = int(math.sqrt(num_patches))
            
            # Reshape to spatial: [heads, H, W]
            cam = cam.reshape(-1, grid_size, grid_size)
            grad_cam = grad_cam.reshape(-1, grid_size, grid_size)
        else:
            # CLIP (Self Attention)
            # attn_weights shape: [B, heads, N+1, N+1]
            # Step 1: Extract CLS token (row 0) attention to patches (cols 1:)
            cam = attn_weights[0, :, 0, 1:]    # [heads, num_patches]
            grad_cam = grad[0, :, 0, 1:]       # [heads, num_patches]
            
            num_patches = cam.shape[-1]
            grid_size = int(math.sqrt(num_patches))
            
            # Step 2: Reshape to spatial: [heads, H, W]
            cam = cam.reshape(-1, grid_size, grid_size)
            grad_cam = grad_cam.reshape(-1, grid_size, grid_size)
        
        # Step 3: GAP over spatial dimensions for gradients (1 scalar per head)
        grad_gap = grad_cam.mean(dim=[1, 2], keepdim=True)  # [heads, 1, 1]
        
        # Step 4 & 5 & 6: Weight cam by gradient, average over heads, then clamp
        cam = (cam * grad_gap).mean(dim=0).clamp(min=0)  # [H, W]
        
        # FIRST normalization (inside generate_cam_attn in official code)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Reshape to [1, 1, H, W] for interpolation
        heatmap = cam.unsqueeze(0).unsqueeze(0)
        
        # Upsample to image size (interpolate with scale_factor=16 in official, here we go to full image size)
        heatmap = F.interpolate(
            heatmap, 
            size=image.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # SECOND normalization (in eval_batch after interpolation in official code)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()


class LeGradBaselineEvaluator:
    """
    Evaluate standard LeGrad baseline on correct and wrong prompts.
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
        use_gradcam=False,
        gradcam_layer=8,
        use_chefercam=False,
        threshold_mode='mean',
        fixed_threshold=0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.num_negatives = num_negatives
        self.negative_strategy = negative_strategy
        self.use_gradcam = use_gradcam
        self.gradcam_layer = gradcam_layer
        self.use_chefercam = use_chefercam
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        
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
        print("[baseline] Precomputing class embeddings...")
        
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
        
        print(f"[baseline] Precomputed {len(all_prompts)} class embeddings.")
        print(f"[baseline] Negative strategy: {self.negative_strategy}, num_negatives: {self.num_negatives}")
    
    def _sample_negative_indices(self, correct_idx: int) -> list:
        """Sample negative class indices different from the correct class."""
        all_indices = list(range(len(self.unique_wnids)))
        all_indices.remove(correct_idx)
        
        # Random sampling (other strategies can be added here)
        return random.sample(all_indices, min(self.num_negatives, len(all_indices)))
    
    def evaluate(self, show_progress=True):
        """
        Evaluate LeGrad baseline on correct and wrong prompts.
        
        Returns dict with:
        - correct: {miou, acc, map, max, mean, median, min, auroc}
        - wrong: {miou, acc, map, max, mean, median, min, auroc}
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
            iterator = tqdm(iterator, desc="Evaluating LeGrad baseline")
        
        for idx in iterator:
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
                gt_mask = np.array(gt_pil).astype(np.uint8)
                H_gt, W_gt = gt_mask.shape
                
                # Get class info
                wnid = self.wnids_in_seg[idx]
                cls_idx = self.wnid_to_idx[wnid]
                text_emb = self.all_text_embs[cls_idx:cls_idx + 1]
                
                # DEBUG: Print class info for first 3 images
                if idx < 3:
                    correct_class = self.wnid_to_classname.get(wnid, wnid)
                    correct_prompt = self.wnid_to_prompt.get(wnid, f"unknown prompt for {wnid}")
                    print(f"\n[DEBUG] Image {idx}:")
                    print(f"  wnid: {wnid}")
                    print(f"  cls_idx: {cls_idx}")
                    print(f"  CORRECT class: {correct_class}")
                    print(f"  CORRECT prompt: {correct_prompt}")
                
                def compute_metrics(text_emb_1x):
                    """Compute heatmap and metrics for standard LeGrad, GradCAM, or CheferCAM."""
                    if self.use_chefercam:
                        heatmap = compute_chefercam_heatmap(self.model, img_t, text_emb_1x)
                    elif self.use_gradcam:
                        heatmap = compute_gradcam_heatmap(self.model, img_t, text_emb_1x, layer_index=self.gradcam_layer)
                    else:
                        heatmap = compute_legrad_heatmap(self.model, img_t, text_emb_1x)
                    
                    # Resize heatmap to GT size FIRST (matching official Chefer implementation)
                    heatmap_resized = F.interpolate(
                        heatmap.view(1, 1, H_feat, W_feat),
                        size=(H_gt, W_gt),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Normalize AFTER resizing (as in official implementation)
                    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                    
                    # Compute threshold on the RESIZED normalized heatmap (official Chefer method)
                    if self.threshold_mode == 'mean':
                        thr = heatmap_resized.mean().item()
                    else:
                        thr = self.fixed_threshold
                    
                    # Create binary predictions for cumulative IOU
                    Res_1 = (heatmap_resized > thr).float()
                    Res_0 = (heatmap_resized <= thr).float()
                    output_tensor = torch.stack([Res_0, Res_1], dim=1).squeeze(0) # [2, H, W]
                    gt_tensor = torch.from_numpy(gt_mask).long().to(output_tensor.device)
                    
                    # Compute intersection/union
                    inter, union = batch_intersection_union(output_tensor, gt_tensor, nclass=2)
                    
                    # Also compute AP (standard way)
                    ap = compute_map_score(heatmap_resized.squeeze().numpy(), gt_mask)
                    
                    # Pixel accuracy
                    pred_mask = (heatmap_resized.squeeze().numpy() > thr).astype(np.uint8)
                    correct_pixels = (pred_mask == gt_mask).sum()
                    total_pixels = gt_mask.size
                    
                    # Heatmap statistics
                    heatmap_np = heatmap_resized.squeeze().numpy()
                    max_val = np.max(heatmap_np)
                    mean_val = np.mean(heatmap_np)
                    median_val = np.median(heatmap_np)
                    min_val = np.min(heatmap_np)
                    
                    # AUROC
                    gt_binary = (gt_mask > 0).astype(int).flatten()
                    pred_flat = heatmap_np.flatten()
                    
                    if len(np.unique(gt_binary)) > 1:
                        auroc = roc_auc_score(gt_binary, pred_flat)
                    else:
                        auroc = np.nan
                    
                    return inter, union, correct_pixels, total_pixels, ap, max_val, mean_val, median_val, min_val, auroc

                
                # === CORRECT PROMPT ===
                inter_c, union_c, correct_c, label_c, ap_c, mx_c, mn_c, md_c, mi_c, auroc_c = compute_metrics(text_emb)
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
                
                # DEBUG: Print per-image correct metrics (first 3 only)
                if idx < 3:
                    # Compute local metrics for visualization
                    iou_local = inter_c[1] / (union_c[1] + 1e-10)
                    acc_local = 100.0 * correct_c / (label_c + 1e-10)
                    auc_str = f"{auroc_c:.4f}" if not np.isnan(auroc_c) else "nan"
                    print(f"  CORRECT IoU: {iou_local:.4f}, Acc: {acc_local:.2f}%, mAP: {ap_c:.4f}, AUROC: {auc_str}")
                
                # Store for comparison
                correct_iou_this_image = inter_c[1] / (union_c[1] + 1e-10)
                
                # === WRONG PROMPTS ===
                neg_indices = self._sample_negative_indices(cls_idx)
                
                # DEBUG: Print wrong class info for first 3 images
                if idx < 3:
                    print(f"  WRONG class indices sampled: {neg_indices}")
                    for neg_idx in neg_indices:
                        neg_wnid = self.idx_to_wnid[neg_idx]
                        wrong_class = self.wnid_to_classname.get(neg_wnid, neg_wnid)
                        wrong_prompt = self.wnid_to_prompt.get(neg_wnid, f"unknown prompt for {neg_wnid}")
                        print(f"    WRONG class: {wrong_class}, prompt: {wrong_prompt}")
                
                for neg_idx in neg_indices:
                    neg_emb = self.all_text_embs[neg_idx:neg_idx + 1]
                    neg_wnid = self.idx_to_wnid[neg_idx]
                    neg_class = self.wnid_to_classname[neg_wnid]
                    
                    inter_w, union_w, correct_w, label_w, ap_w, mx_w, mn_w, md_w, mi_w, auroc_w = compute_metrics(neg_emb)
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
                    
                    # DEBUG: Print per-image wrong metrics and comparison (first 3 only)
                    if idx < 3:
                        iou_local_w = inter_w[1] / (union_w[1] + 1e-10)
                        comparison = "GOOD (correct > wrong)" if correct_iou_this_image > iou_local_w else "BAD (wrong > correct)"
                        print(f"  WRONG IoU ({neg_class}): {iou_local_w:.4f} -- {comparison}")
                
                
            except Exception as e:
                print(f"[Warning] Error processing image {idx}: {e}")
                continue
        
        # Compute averages (Global IoU / Pixel Acc)
        def compute_global_metrics(res_dict):
            # mIoU
            iou = res_dict['inter'].astype(np.float64) / (res_dict['union'].astype(np.float64) + 1e-10)
            miou = 100.0 * iou.mean()
            
            # Pixel Acc
            pix_acc = 100.0 * res_dict['pixel_correct'] / (res_dict['pixel_label'] + 1e-10)
            
            # mAP (still mean of APs per image)
            map_score = np.mean(res_dict['ap']) * 100 if res_dict['ap'] else 0.0
            
            # AUROC
            auroc_score = np.mean(res_dict['auroc']) * 100 if res_dict['auroc'] else 0.0
            
            return miou, pix_acc, map_score, auroc_score

        # Correct
        correct_miou, correct_acc, correct_map, correct_auroc = compute_global_metrics(correct_results)
        
        # Wrong
        wrong_miou, wrong_acc, wrong_map, wrong_auroc = compute_global_metrics(wrong_results)
        
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
        return {
            'correct': {
                'miou': correct_miou, 'acc': correct_acc, 'map': correct_map, 'auroc': correct_auroc,
                **correct_stats
            },
            'wrong': {
                'miou': wrong_miou, 'acc': wrong_acc, 'map': wrong_map, 'auroc': wrong_auroc,
                **wrong_stats
            }
        }



def main():
    parser = argparse.ArgumentParser(description='Compute LeGrad Negative Baseline')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_siglip', action='store_true',
                        help='Use SigLIP instead of CLIP for comparison')
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Negative sampling settings
    parser.add_argument('--num_negatives', type=int, default=1, 
                        help='Number of negative (wrong) prompts per image')
    parser.add_argument('--negative_strategy', type=str, default='random', 
                        choices=['random'],
                        help='Strategy for sampling negative prompts (only random for baseline)')
    
    # LeGrad threshold
    parser.add_argument('--threshold_mode', type=str, default='mean',
                        choices=['mean', 'fixed'],
                        help='Thresholding mode: "mean" (adaptive) or "fixed"')
    parser.add_argument('--fixed_threshold', type=float, default=0.5, 
                        help='Fixed threshold value when threshold_mode=fixed')
    
    # GradCAM settings
    parser.add_argument('--use_gradcam', action='store_true',
                        help='Use GradCAM instead of LeGrad for baseline evaluation')
    parser.add_argument('--gradcam_layer', type=int, default=8,
                        help='GradCAM layer index (default: 8)')
    parser.add_argument('--use_chefercam', action='store_true',
                        help='Use CheferCAM (attention GradCAM) instead of LeGrad')
    
    # Composite score calculation
    parser.add_argument('--composite_lambda', type=float, default=0.5,
                        help='Weight for wrong-prompt penalty in composite score')
    
    # Output
    parser.add_argument('--output_json', type=str, default='legrad_negative_baseline.json')
    
    args = parser.parse_args()
    
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
    
    # Create evaluator
    evaluator = LeGradBaselineEvaluator(
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
        use_gradcam=args.use_gradcam,
        gradcam_layer=args.gradcam_layer,
        use_chefercam=args.use_chefercam,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
    )
    
    # Run evaluation
    if args.use_chefercam:
        method_name = "CheferCAM"
    elif args.use_gradcam:
        method_name = "GradCAM"
    else:
        method_name = "LeGrad"
    
    print(f"\n{'='*60}")
    print(f"Computing {method_name} Baseline ({model_type}, No Sparse Encoding)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name} ({args.pretrained})")
    print(f"Method: {method_name}")
    if args.use_gradcam:
        print(f"GradCAM layer: {args.gradcam_layer}")
    print(f"Strategy: {args.negative_strategy}")
    print(f"Num negatives per image: {args.num_negatives}")
    print(f"Threshold Mode: {args.threshold_mode}")
    if args.threshold_mode == 'fixed':
        print(f"Fixed Threshold: {args.fixed_threshold}")
    print(f"{'='*60}\n")
    
    results = evaluator.evaluate()
    
    # Compute composite score
    composite = results['correct']['miou'] - args.composite_lambda * results['wrong']['miou']
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} BASELINE RESULTS")
    print(f"{'='*60}")
    
    print(f"\n=== CORRECT PROMPTS (image class = text prompt class) ===")
    print(f"  mIoU:     {results['correct']['miou']:.2f}")
    print(f"  Accuracy: {results['correct']['acc']:.2f}")
    print(f"  mAP:      {results['correct']['map']:.2f}")
    print(f"  Max Val:  {results['correct']['max']:.4f}")
    print(f"  Mean Val: {results['correct']['mean']:.4f}")
    print(f"  Median:   {results['correct']['median']:.4f}")
    print(f"  Min Val:  {results['correct']['min']:.4f}")
    print(f"  AUROC:    {results['correct']['auroc']:.2f}")
    print(f"  Samples:  {results['correct']['n_samples']}")
    
    print(f"\n=== WRONG PROMPTS (image class ≠ text prompt class) ===")
    print(f"  mIoU:     {results['wrong']['miou']:.2f}")
    print(f"  Accuracy: {results['wrong']['acc']:.2f}")
    print(f"  mAP:      {results['wrong']['map']:.2f}")
    print(f"  Max Val:  {results['wrong']['max']:.4f}")
    print(f"  Mean Val: {results['wrong']['mean']:.4f}")
    print(f"  Median:   {results['wrong']['median']:.4f}")
    print(f"  Min Val:  {results['wrong']['min']:.4f}")
    print(f"  AUROC:    {results['wrong']['auroc']:.2f}")
    print(f"  Samples:  {results['wrong']['n_samples']}")
    
    print(f"\n=== COMPOSITE SCORE ===")
    print(f"  Score = correct_mIoU - {args.composite_lambda} × wrong_mIoU")
    print(f"  Score = {results['correct']['miou']:.2f} - {args.composite_lambda} × {results['wrong']['miou']:.2f}")
    print(f"  Score = {composite:.2f}")
    
    print(f"\n{'='*60}")
    print("Use these values as baseline for optimize_anti_hallucination.py:")
    print(f"{'='*60}")
    print(f"  --baseline_correct_miou {results['correct']['miou']:.2f}")
    print(f"  --baseline_wrong_miou {results['wrong']['miou']:.2f}")
    print(f"  --baseline_composite {composite:.2f}")
    
    # Helper function to convert numpy types to Python native types for JSON
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    # Save results
    output = {
        'correct': convert_to_native(results['correct']),
        'wrong': convert_to_native(results['wrong']),
        'composite': float(composite),
        'settings': {
            'threshold_mode': args.threshold_mode,
            'fixed_threshold': args.fixed_threshold,
            'num_negatives': args.num_negatives,
            'negative_strategy': args.negative_strategy,
            'limit': args.limit if args.limit > 0 else 'all',
            'model_name': args.model_name,
            'pretrained': args.pretrained,
            'model_type': model_type,
            'use_siglip': args.use_siglip,
            'image_size': args.image_size,
            'composite_lambda': args.composite_lambda,
            'use_gradcam': args.use_gradcam,
            'gradcam_layer': args.gradcam_layer if args.use_gradcam else None,
            'use_chefercam': args.use_chefercam,
        }
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[output] Results saved to {args.output_json}")
    
    # Print example command to run optimization with these baselines
    print(f"\n{'='*60}")
    print("Example command to run optimization:")
    print(f"{'='*60}")
    print(f"""python scripts/optimize_anti_hallucination.py \\
    --mat_file {args.mat_file} \\
    --n_trials 100 \\
    --limit {args.limit if args.limit > 0 else 1000} \\
    --negative_strategy {args.negative_strategy} \\
    --num_negatives {args.num_negatives} \\
    --composite_lambda {args.composite_lambda} \\
    --output_json anti_hallucination_results.json""")


if __name__ == '__main__':
    main()
