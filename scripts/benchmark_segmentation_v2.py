#!/usr/bin/env python3
"""
Benchmark Segmentation Script (v2)

This script evaluates segmentation methods on the ImageNet-Segmentation dataset
(gtsegs_ijcv.mat). It includes:
  - LeGrad (our method) - original
  - GradCAM - gradcam
  - Attention Rollout (reference implementation from Transformer-Explainability) - rollout
  - TextSpan (Text-Based Decomposition from Gandelsman et al.) - textspan
  - Sparse encoding variants - sparse

The Attention Rollout implementation follows the reference from:
  hila-chefer/Transformer-Explainability (GitHub)

The TextSpan implementation follows the reference from:
  yossigandelsman/clip_text_span (GitHub)
  Paper: "Interpreting CLIP's Image Representation via Text-Based Decomposition"
  https://openreview.net/forum?id=5Ca9sSzuDp
  
Usage:
  python scripts/benchmark_segmentation_v2.py --mat_file scripts/data/gtsegs_ijcv.mat --methods original,gradcam,rollout,textspan
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

# Constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


# =============================================================================
# Reference Attention Rollout Implementation
# Adapted from: https://github.com/hila-chefer/Transformer-Explainability
# File: baselines/ViT/ViT_explanation_generator.py
# =============================================================================

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Compute attention rollout between layers.
    
    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers"
    Implementation adapted from hila-chefer/Transformer-Explainability
    
    Args:
        all_layer_matrices: List of attention matrices [batch, heads, N, N] averaged over heads
        start_layer: Layer to start rollout from (default 0, but reference uses 1)
    
    Returns:
        Joint attention matrix after rollout [batch, N, N]
    """
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device
    
    # Add residual connection (identity matrix)
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    
    # Renormalize rows to sum to 1
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    
    # Recursive matrix multiplication from start_layer
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    return joint_attention


# =============================================================================
# TextSpan Implementation
# Based on: "Interpreting CLIP's Image Representation via Text-Based Decomposition"
# Paper: https://openreview.net/forum?id=5Ca9sSzuDp
# Reference: https://github.com/yossigandelsman/clip_text_span
# =============================================================================

def compute_textspan_spatial(model, image, text_embedding, num_last_layers=4):
    """
    Compute TextSpan spatial decomposition for segmentation.
    
    TextSpan decomposes CLIP's image representation by:
    1. Extracting patch-level features from transformer layers
    2. Projecting them onto the text embedding direction
    3. Aggregating across layers
    
    This follows the spatial decomposition approach from the paper
    "Interpreting CLIP's Image Representation via Text-Based Decomposition"
    
    Args:
        model: LeWrapper model
        image: Input image tensor [1, 3, H, W]
        text_embedding: Text embedding tensor [1, D] (normalized)
        num_last_layers: Number of last layers to aggregate (default 4 as in paper)
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    """
    device = image.device
    blocks = model.visual.transformer.resblocks
    num_blocks = len(blocks)
    embed_dim = blocks[0].attn.embed_dim
    num_heads = blocks[0].attn.num_heads
    head_dim = embed_dim // num_heads
    
    with torch.no_grad():
        # Get the input to the transformer
        x = model.visual.conv1(image)  # [B, embed_dim, H', W']
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, embed_dim]
        
        # Add class token
        batch_size = x.shape[0]
        class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
        class_token = class_token.expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat([class_token, x], dim=1)  # [B, N+1, embed_dim]
        
        # Add positional embedding (handle dynamic size)
        num_patches = x.shape[1]
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
        
        # Resize positional embedding if needed
        if pos_embed.shape[0] != num_patches:
            cls_pos = pos_embed[:1]  # [1, embed_dim]
            patch_pos = pos_embed[1:]  # [N_orig, embed_dim]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            new_size = int(math.sqrt(num_patches - 1))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, embed_dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        
        # Pre-normalization (if exists)
        if hasattr(model.visual, 'ln_pre'):
            x = model.visual.ln_pre(x)
        
        # Process through transformer blocks
        x = x.permute(1, 0, 2)  # [N+1, B, embed_dim] for transformer
        
        # Collect features from last N layers
        spatial_contributions = []
        start_layer = max(0, num_blocks - num_last_layers)
        
        for layer_idx, block in enumerate(blocks):
            # Get attention weights and values for this block
            attn_module = block.attn
            
            # Layer norm
            x_normed = block.ln_1(x)
            
            # Compute Q, K, V
            qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, _ = q.shape
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            # Compute attention weights
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Compute attention output
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = attn_module.out_proj(attn_output)
            
            # Residual connection
            x = x + attn_output
            
            # MLP block
            x = x + block.mlp(block.ln_2(x))
            
            # For last N layers, compute spatial contribution
            if layer_idx >= start_layer:
                # Get spatial tokens (exclude CLS)
                spatial_tokens = x[1:, :, :]  # [N_patches, B, embed_dim]
                spatial_tokens = spatial_tokens.permute(1, 0, 2)  # [B, N_patches, embed_dim]
                
                # Apply layer normalization and projection
                spatial_features = model.visual.ln_post(spatial_tokens)  # [B, N_patches, embed_dim]
                if model.visual.proj is not None:
                    spatial_features = spatial_features @ model.visual.proj  # [B, N_patches, proj_dim]
                
                # Normalize features
                spatial_features = F.normalize(spatial_features, dim=-1)
                
                # Compute similarity with text embedding at each position
                # text_embedding: [1, D], spatial_features: [B, N_patches, D]
                similarity = torch.einsum('bd,bnd->bn', text_embedding, spatial_features)  # [B, N_patches]
                
                spatial_contributions.append(similarity)
        
        # Aggregate contributions across layers
        if len(spatial_contributions) > 0:
            # Sum contributions from all layers
            heatmap = torch.stack(spatial_contributions, dim=0).sum(dim=0)  # [B, N_patches]
        else:
            # Fallback: use final layer only
            spatial_tokens = x[1:, :, :]
            spatial_tokens = spatial_tokens.permute(1, 0, 2)
            spatial_features = model.visual.ln_post(spatial_tokens)
            if model.visual.proj is not None:
                spatial_features = spatial_features @ model.visual.proj
            spatial_features = F.normalize(spatial_features, dim=-1)
            heatmap = torch.einsum('bd,bnd->bn', text_embedding, spatial_features)
        
        # Reshape to grid
        heatmap = heatmap.reshape(1, 1, grid_h, grid_w)
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Upsample to image size
        heatmap = F.interpolate(
            heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False
        )
        
        return heatmap[0, 0].detach().cpu()


def compute_textspan_attention_decomposition(model, image, text_embedding, num_last_layers=4):
    """
    Alternative TextSpan implementation using attention-weighted decomposition.
    
    This version follows the paper more closely by:
    1. Computing the attention from CLS to spatial tokens at each head
    2. Weighting the value contributions by attention
    3. Projecting onto text direction
    
    Args:
        model: LeWrapper model
        image: Input image tensor [1, 3, H, W]
        text_embedding: Text embedding tensor [1, D] (normalized)
        num_last_layers: Number of last layers to aggregate
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    """
    device = image.device
    blocks = model.visual.transformer.resblocks
    num_blocks = len(blocks)
    embed_dim = blocks[0].attn.embed_dim
    num_heads = blocks[0].attn.num_heads
    head_dim = embed_dim // num_heads
    
    with torch.no_grad():
        # Get the input to the transformer
        x = model.visual.conv1(image)
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Add class token
        batch_size = x.shape[0]
        class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
        class_token = class_token.expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat([class_token, x], dim=1)
        
        # Add positional embedding
        num_patches = x.shape[1]
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
        
        if pos_embed.shape[0] != num_patches:
            cls_pos = pos_embed[:1]
            patch_pos = pos_embed[1:]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            new_size = int(math.sqrt(num_patches - 1))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, embed_dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        
        if hasattr(model.visual, 'ln_pre'):
            x = model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # [N+1, B, embed_dim]
        
        # Collect attention contributions
        attention_contributions = []
        start_layer = max(0, num_blocks - num_last_layers)
        
        for layer_idx, block in enumerate(blocks):
            attn_module = block.attn
            x_normed = block.ln_1(x)
            
            qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, _ = q.shape
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, seq_len, seq_len]
            
            if layer_idx >= start_layer:
                # Get attention from CLS (position 0) to all spatial tokens
                # attn_weights: [bsz*heads, seq_len, seq_len]
                cls_attn = attn_weights[:, 0, 1:]  # [bsz*heads, N_patches]
                
                # Reshape to [bsz, heads, N_patches] and average over heads
                cls_attn = cls_attn.view(bsz, num_heads, -1).mean(dim=1)  # [bsz, N_patches]
                
                attention_contributions.append(cls_attn)
            
            # Continue forward pass
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = attn_module.out_proj(attn_output)
            
            x = x + attn_output
            x = x + block.mlp(block.ln_2(x))
        
        # Now weight the final spatial features by aggregated attention
        if len(attention_contributions) > 0:
            # Sum attention contributions
            attn_map = torch.stack(attention_contributions, dim=0).sum(dim=0)  # [B, N_patches]
        else:
            attn_map = torch.ones(batch_size, grid_h * grid_w, device=device)
        
        # Get final spatial features
        spatial_tokens = x[1:, :, :].permute(1, 0, 2)  # [B, N_patches, embed_dim]
        spatial_features = model.visual.ln_post(spatial_tokens)
        if model.visual.proj is not None:
            spatial_features = spatial_features @ model.visual.proj
        spatial_features = F.normalize(spatial_features, dim=-1)
        
        # Compute text similarity
        text_sim = torch.einsum('bd,bnd->bn', text_embedding, spatial_features)  # [B, N_patches]
        
        # Combine attention and text similarity
        # TextSpan uses attention to weight the contribution, then projects onto text
        heatmap = attn_map * text_sim  # [B, N_patches]
        
        # Reshape to grid
        heatmap = heatmap.reshape(1, 1, grid_h, grid_w)
        
        # Apply ReLU (only positive contributions)
        heatmap = torch.clamp(heatmap, min=0)
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Upsample to image size
        heatmap = F.interpolate(
            heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False
        )
        
        return heatmap[0, 0].detach().cpu()


def compute_attention_rollout_reference(model, image, start_layer=1):
    """
    Compute Attention Rollout following the reference implementation.
    
    This implementation:
    1. Uses start_layer=1 (skip first layer) as in the reference
    2. Averages attention heads before rollout
    3. Adds residual connections and renormalizes
    
    Reference: hila-chefer/Transformer-Explainability
    
    IMPORTANT: LeWrapper only hooks layers from starting_depth onwards.
    For rollout, we need ALL layers' attention. This implementation uses
    temporary hooks to capture attention from all layers.
    
    Args:
        model: LeWrapper model with attention map storage
        image: Input image tensor [1, 3, H, W]
        start_layer: Layer to start rollout (reference uses 1)
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    """
    # We need to hook ALL attention layers, not just the ones LeWrapper hooks
    # Use temporary hooks to capture attention from all layers
    all_attentions = []
    hooks = []
    
    def make_hook(storage_list):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights) from nn.MultiheadAttention
            # attn_weights shape: [batch, num_heads, seq_len, seq_len] after our processing
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [batch, tgt_len, src_len] for default, or custom
                if attn_weights is not None:
                    storage_list.append(attn_weights.detach())
        return hook_fn
    
    # Get attention via custom forward that computes and stores attention
    blocks = model.visual.transformer.resblocks
    num_heads = blocks[0].attn.num_heads
    embed_dim = blocks[0].attn.embed_dim
    head_dim = embed_dim // num_heads
    
    # Manual attention computation for each layer
    with torch.no_grad():
        # Get the input to the transformer
        x = model.visual.conv1(image)  # [B, embed_dim, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, embed_dim]
        
        # Add class token
        batch_size = x.shape[0]
        class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
        class_token = class_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([class_token, x], dim=1)  # [B, N+1, embed_dim]
        
        # Add positional embedding (handle dynamic size)
        num_patches = x.shape[1]
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
        
        # Resize positional embedding if needed
        if pos_embed.shape[0] != num_patches:
            # Interpolate positional embedding
            cls_pos = pos_embed[:1]  # [1, embed_dim]
            patch_pos = pos_embed[1:]  # [N_orig, embed_dim]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            new_size = int(math.sqrt(num_patches - 1))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, embed_dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        
        # Pre-normalization (if exists)
        if hasattr(model.visual, 'ln_pre'):
            x = model.visual.ln_pre(x)
        
        # Process through transformer blocks and collect attention
        x = x.permute(1, 0, 2)  # [N+1, B, embed_dim] for transformer
        
        for block in blocks:
            # Manually compute attention for this block
            attn_module = block.attn
            
            # Layer norm
            x_normed = block.ln_1(x)
            
            # Compute Q, K, V
            qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len, bsz, _ = q.shape
            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            
            # Compute attention weights
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Store attention: reshape from [bsz*heads, N, N] to [bsz, heads, N, N]
            attn_reshaped = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            all_attentions.append(attn_reshaped.mean(dim=1))  # Average over heads -> [bsz, N, N]
            
            # Compute attention output
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = attn_module.out_proj(attn_output)
            
            # Residual connection
            x = x + attn_output
            
            # MLP block
            x = x + block.mlp(block.ln_2(x))
    
    if len(all_attentions) == 0:
        raise RuntimeError("No attention maps captured.")
    
    # Compute rollout with reference parameters
    rollout = compute_rollout_attention(all_attentions, start_layer=start_layer)
    
    # Extract CLS token attention to spatial tokens (row 0, cols 1:)
    cls_attn = rollout[:, 0, 1:]  # [batch, num_patches]
    
    # Reshape to grid
    num_patches = cls_attn.shape[-1]
    grid_size = int(math.sqrt(num_patches))
    heatmap = cls_attn.reshape(1, 1, grid_size, grid_size)
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Upsample to image size
    heatmap = F.interpolate(
        heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False
    )
    
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


# =============================================================================
# Sparse Encoding (kept for completeness)
# =============================================================================

def omp_sparse_residual(
    x_1x: torch.Tensor,
    D: torch.Tensor,
    max_atoms: int = 8,
    tol: float = 1e-6,
    return_num_selected: bool = False,
):
    """Orthogonal Matching Pursuit residual."""
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        r_norm = F.normalize(x_1x, dim=-1)
        return (r_norm, 0) if return_num_selected else r_norm

    x = x_1x.clone()
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)
        x_hat = (s.t() @ D_S).to(x.dtype)
        r = (x - x_hat)
        if float(torch.norm(r)) <= tol:
            break
    
    if torch.norm(r) <= tol:
        r_norm = F.normalize(x, dim=-1)
    else:
        r_norm = F.normalize(r, dim=-1)
    
    return (r_norm, len(selected)) if return_num_selected else r_norm


def wordnet_neighbors_configured(
    keyword: str,
    use_synonyms: bool,
    use_hypernyms: bool,
    use_hyponyms: bool,
    use_siblings: bool,
    use_fallback: bool = True,
    limit_per_relation: int = 8
):
    """Configurable WordNet neighbors with optional fallback."""
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn_local
    except Exception as e:
        print(f"[WordNet] Warning: Failed to load NLTK/WordNet: {e}")
        return []

    def get_neighbors_for_term(term: str):
        out = []
        seen = set()
        key_low = term.lower()
        synsets = wn_local.synsets(term, pos=wn_local.NOUN)
        for s in synsets[:limit_per_relation]:
            if use_synonyms:
                for l in s.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name)
                        seen.add(name)
            if use_hypernyms:
                for h in s.hypernyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name)
                            seen.add(name)
            if use_hyponyms:
                for h in s.hyponyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name)
                            seen.add(name)
            if use_siblings:
                for h in s.hypernyms()[:limit_per_relation]:
                    for sib in h.hyponyms()[:limit_per_relation]:
                        for l in sib.lemmas()[:limit_per_relation]:
                            name = l.name().replace('_', ' ').lower()
                            if name != key_low and name not in seen:
                                out.append(name)
                                seen.add(name)
        return out

    out = get_neighbors_for_term(keyword)
    if out:
        return out[:max(1, limit_per_relation * 3)]

    if not use_fallback:
        return []

    words = keyword.split()
    if len(words) > 1:
        for i in range(1, len(words)):
            sub_term = " ".join(words[i:])
            out = get_neighbors_for_term(sub_term)
            if out:
                return out[:max(1, limit_per_relation * 3)]

    return []


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Segmentation (v2) - With Reference Rollout Implementation'
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
        '--methods', type=str, default='original,gradcam,rollout,textspan',
        help="Comma-separated methods: original (LeGrad), gradcam, rollout, textspan, sparse"
    )
    
    # Rollout settings
    parser.add_argument('--rollout_start_layer', type=int, default=0,
                        help='Start layer for attention rollout (0=all layers, 1=skip first)')
    parser.add_argument('--rollout_head_fusion', type=str, default='mean',
                        choices=['mean', 'max', 'min'],
                        help='How to fuse attention heads for rollout')
    
    # TextSpan settings
    parser.add_argument('--textspan_num_layers', type=int, default=4,
                        help='Number of last layers to aggregate for TextSpan (default 4 as in paper)')
    parser.add_argument('--textspan_mode', type=str, default='attention',
                        choices=['spatial', 'attention'],
                        help='TextSpan mode: spatial (direct projection) or attention (attention-weighted)')
    
    # Threshold settings
    parser.add_argument('--threshold_mode', type=str, default='mean',
                        choices=['mean', 'fixed'],
                        help='Thresholding mode: "mean" (per-image adaptive as in reference) or "fixed"')
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='Fixed threshold value when threshold_mode=fixed')
    
    # Sparse settings (kept for completeness)
    parser.add_argument('--atoms', type=int, default=8)
    parser.add_argument('--dict_include_prompts', type=int, default=1)
    parser.add_argument('--wn_use_synonyms', type=int, default=0)
    parser.add_argument('--wn_use_hypernyms', type=int, default=0)
    parser.add_argument('--wn_use_hyponyms', type=int, default=0)
    parser.add_argument('--wn_use_siblings', type=int, default=1)
    parser.add_argument('--wn_fallback_search', type=int, default=1)
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.9)
    parser.add_argument('--sparse_threshold', type=float, default=0.5)
    parser.add_argument('--sparse_target', type=str, default='legrad', choices=['legrad', 'gradcam'])
    
    # Visualization
    parser.add_argument('--vis_first_k', type=int, default=0)
    parser.add_argument('--vis_output_dir', type=str, default='outputs/segmentation_vis_v2')
    
    args = parser.parse_args()

    # Parse methods
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    allowed_methods = {'original', 'gradcam', 'rollout', 'textspan', 'sparse'}
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods. Use --methods with: original,gradcam,rollout,textspan,sparse")
    
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
    try:
        f = h5py.File(args.mat_file, 'r')
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
                heatmap = compute_legrad_for_embedding(model, img_t, text_emb_1x)
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
            
            # --- ROLLOUT (Reference Implementation) ---
            if 'rollout' in methods:
                heatmap = compute_attention_rollout_reference(
                    model, img_t, start_layer=args.rollout_start_layer
                )
                # Already at image resolution, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['rollout'] = heatmap_resized
            
            # --- TEXTSPAN (Text-Based Decomposition) ---
            if 'textspan' in methods:
                if args.textspan_mode == 'spatial':
                    heatmap = compute_textspan_spatial(
                        model, img_t, text_emb_1x, 
                        num_last_layers=args.textspan_num_layers
                    )
                else:
                    heatmap = compute_textspan_attention_decomposition(
                        model, img_t, text_emb_1x,
                        num_last_layers=args.textspan_num_layers
                    )
                # Already at image resolution, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['textspan'] = heatmap_resized
            
            # --- SPARSE ---
            if 'sparse' in methods:
                # Build dictionary
                parts = []
                if bool(args.dict_include_prompts) and len(unique_wnids) > 1:
                    if cls_idx > 0:
                        parts.append(all_text_embs[:cls_idx])
                    if cls_idx + 1 < len(unique_wnids):
                        parts.append(all_text_embs[cls_idx + 1:])
                
                class_label = wnid_to_label.get(wnid) or get_synset_name(wnid)
                use_wn = any([
                    bool(args.wn_use_synonyms),
                    bool(args.wn_use_hypernyms),
                    bool(args.wn_use_hyponyms),
                    bool(args.wn_use_siblings),
                ])
                if use_wn:
                    raw_neighbors = wordnet_neighbors_configured(
                        class_label,
                        use_synonyms=bool(args.wn_use_synonyms),
                        use_hypernyms=bool(args.wn_use_hypernyms),
                        use_hyponyms=bool(args.wn_use_hyponyms),
                        use_siblings=bool(args.wn_use_siblings),
                        use_fallback=bool(args.wn_fallback_search),
                    )
                    if raw_neighbors:
                        prompt = wnid_to_prompt[wnid]
                        neighbor_prompts = [prompt.replace(class_label, w) for w in raw_neighbors]
                        n_tok = tokenizer(neighbor_prompts).to(args.device)
                        with torch.no_grad():
                            n_emb = model.encode_text(n_tok)
                            n_emb = F.normalize(n_emb, dim=-1)
                        parts.append(n_emb)
                
                if parts:
                    D = torch.cat(parts, dim=0)
                    D = F.normalize(D, dim=-1)
                    if args.max_dict_cos_sim is not None and 0.0 < float(args.max_dict_cos_sim) < 1.0:
                        sim = (D @ text_emb_1x.t()).squeeze(-1).abs()
                        keep = sim < float(args.max_dict_cos_sim)
                        D = D[keep]
                else:
                    D = text_emb_1x.new_zeros((0, text_emb_1x.shape[-1]))
                
                sparse_1x = omp_sparse_residual(text_emb_1x, D, max_atoms=args.atoms)
                
                if args.sparse_target == 'gradcam':
                    heatmap = compute_gradcam_for_embedding(model, img_t, sparse_1x)
                else:
                    heatmap = compute_legrad_for_embedding(model, img_t, sparse_1x)
                
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['sparse'] = heatmap_resized
            
            # --- Compute Metrics (Reference Protocol) ---
            for method in methods:
                heatmap = heatmaps[method]
                
                # Normalize heatmap
                heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Determine threshold
                if args.threshold_mode == 'mean':
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

