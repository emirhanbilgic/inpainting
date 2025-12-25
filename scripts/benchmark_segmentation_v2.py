#!/usr/bin/env python3
"""
Benchmark Segmentation Script (v2)

This script evaluates segmentation methods on the ImageNet-Segmentation dataset
(gtsegs_ijcv.mat). It includes:
  - LeGrad (our method) - original
  - GradCAM - gradcam
  - Attention Rollout (reference implementation from Transformer-Explainability) - rollout
  - TextSpan (Text-Based Decomposition from Gandelsman et al.) - textspan
  - Raw Attention (last layer CLS attention averaged over heads) - rawattn
  - Sparse encoding variants - sparse

The Attention Rollout implementation follows the reference from:
  hila-chefer/Transformer-Explainability (GitHub)

The CheferCAM/Transformer Attribution implementation follows the reference from:
  hila-chefer/Transformer-Explainability (GitHub)
  Methods:
    - transformer_attribution: Full LRP-based method (main method, default)
    - attn_gradcam: Baseline GradCAM applied to attention maps

The TextSpan implementation follows the reference from:
  yossigandelsman/clip_text_span (GitHub)
  Paper: "Interpreting CLIP's Image Representation via Text-Based Decomposition"
  https://openreview.net/forum?id=5Ca9sSzuDp

The Raw Attention baseline considers the attention maps from the last layer,
focusing on the weights associated with the [CLS] token. The attention heads
are averaged and the resulting explainability map is normalized.
  
Usage:
  python scripts/benchmark_segmentation_v2.py --mat_file scripts/data/gtsegs_ijcv.mat --methods original,gradcam,rollout,textspan,rawattn,chefercam
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

# TextSpan implementation - no external dependencies needed

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
#
# Key insight from official code (transformer.py forward_per_head):
# - They apply out_proj.weight PER HEAD before summing, via:
#   einsum("bnmhc,dhc->bnmhd", attn_v, out_proj_weight.reshape(embed_dim, heads, head_dim))
# - This gives each head's contribution in embed_dim space (768), not head_dim (64)
# - The result has shape [b, n, m, h, embed_dim]
# - They take CLS query (n=0): [b, m, h, embed_dim]
# - After normalizing through ln_post and projecting through visual.proj: [b, m, h, proj_dim]
# - Sum over layers and heads, take patches only: [b, num_patches, proj_dim]
# - Multiply by classifier and subtract mean across classes
# =============================================================================


class TextSpanExtractor:
    """
    Extracts per-head, per-spatial-position attention contributions from CLIP.
    
    This replicates the PRSLogger from the official TextSpan implementation:
    https://github.com/yossigandelsman/clip_text_span/blob/main/prs_hook.py
    
    CRITICAL: The official implementation applies out_proj.weight per-head, giving
    each head's contribution in embed_dim space. This is different from standard
    attention where heads are merged after computing attn @ v.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attentions = []  # Will store per-layer attention contributions
        
    def reinit(self):
        self.attentions = []
        
    @torch.no_grad()
    def extract_attention_contributions(self, image):
        """
        Extract per-head attention contributions for each spatial position.
        
        Following the official implementation (transformer.py forward_per_head):
        - Compute attn @ v per head
        - Apply out_proj.weight per head: einsum("bhnc,dhc->bhnd")
        - This gives [batch, heads, seq, embed_dim] per layer
        
        Returns:
            attentions: [batch, num_layers, num_positions, num_heads, embed_dim]
            representation: [batch, proj_dim] - the final CLS representation
            grid_size: (grid_h, grid_w) - spatial dimensions
        """
        self.reinit()
        
        visual = self.model.visual
        blocks = visual.transformer.resblocks
        
        embed_dim = blocks[0].attn.embed_dim
        num_heads = blocks[0].attn.num_heads
        head_dim = embed_dim // num_heads
        
        # Reshape out_proj.weight for per-head application
        # out_proj.weight: [embed_dim, embed_dim] -> [embed_dim, num_heads, head_dim]
        
        # Patch embedding
        x = visual.conv1(image)  # [B, C, H', W']
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, D]
        
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        seq_len = num_patches + 1  # +1 for CLS
        
        # Add class token
        cls_token = visual.class_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        
        # Positional embedding (with interpolation if needed)
        pos_embed = visual.positional_embedding
        if pos_embed.shape[0] != x.shape[1]:
            cls_pos = pos_embed[:1]
            patch_pos = pos_embed[1:]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, embed_dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        
        if hasattr(visual, "ln_pre"):
            x = visual.ln_pre(x)
        
        # Transformer expects [seq, batch, dim]
        x = x.permute(1, 0, 2)
        
        # Process through transformer blocks and collect attention contributions
        for block in blocks:
            x_norm = block.ln_1(x)
            
            # Compute Q, K, V
            qkv = F.linear(x_norm, block.attn.in_proj_weight, block.attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            
            seq_len_curr, bsz, _ = q.shape
            
            # Reshape for multi-head attention: [bsz, num_heads, seq, head_dim]
            q = q.view(seq_len_curr, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.view(seq_len_curr, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
            v = v.view(seq_len_curr, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
            
            # Compute attention weights: [bsz, num_heads, seq, seq]
            attn = torch.matmul(q * (head_dim ** -0.5), k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            
            # Compute attn @ v per head: [bsz, num_heads, seq, head_dim]
            attn_v = torch.matmul(attn, v)
            
            # KEY STEP: Apply out_proj.weight PER HEAD (as in official forward_per_head)
            # out_proj.weight: [embed_dim, embed_dim] -> reshape to [embed_dim, num_heads, head_dim]
            out_weight = block.attn.out_proj.weight  # [embed_dim, embed_dim]
            out_weight_per_head = out_weight.view(embed_dim, num_heads, head_dim)
            
            # Apply per-head: einsum("bhnc,dhc->bhnd") 
            # Input: attn_v [bsz, heads, seq, head_dim]
            # Weight: [embed_dim, heads, head_dim]
            # Output: [bsz, heads, seq, embed_dim]
            contrib = torch.einsum("bhnc,dhc->bhnd", attn_v, out_weight_per_head)
            
            # Extract CLS query's attention output (what CLS receives from each position)
            # contrib shape: [bsz, num_heads, seq, embed_dim]
            # We want CLS query (position 0 in query dimension)... but actually this is attn @ v,
            # which is what each query position receives. For CLS, this is already included.
            # 
            # Actually, re-reading the official code: they capture at "attn.out.post" which is
            # the output after applying out_proj per head. The shape is [b, n, m, h, d] where
            # n is query positions and m is key positions... but that's for extended_attn_v.
            #
            # Let me re-read: in forward_per_head, they do:
            # x = torch.einsum("bhnm,bhmc->bnmhc", attn, v)  # [b, n, m, h, head_dim]
            # Then apply out_proj per head:
            # x = torch.einsum("bnmhc,dhc->bnmhd", x, out_proj_weight)  # [b, n, m, h, embed_dim]
            #
            # So it's not just attn @ v, but preserving the key dimension m!
            # This means for each query position n and key position m, we have the contribution.
            
            # Let me redo this properly:
            # attn: [bsz, heads, seq_n, seq_m] (query x key)
            # v: [bsz, heads, seq_m, head_dim]
            # We want: [bsz, seq_n, seq_m, heads, head_dim] via einsum("bhnm,bhmc->bnmhc", attn, v)
            extended_attn_v = torch.einsum("bhnm,bhmc->bnmhc", attn, v)
            
            # Apply out_proj per head: [bsz, seq_n, seq_m, heads, embed_dim]
            contrib = torch.einsum("bnmhc,dhc->bnmhd", extended_attn_v, out_weight_per_head)
            
            # Extract CLS query (n=0): [bsz, seq_m, heads, embed_dim]
            cls_contrib = contrib[:, 0, :, :, :]
            
            # Add bias term distributed across positions and heads
            bias_term = block.attn.out_proj.bias  # [embed_dim]
            seq_m = cls_contrib.shape[1]
            bias_distributed = bias_term.view(1, 1, 1, embed_dim) / (seq_m * num_heads)
            cls_contrib = cls_contrib + bias_distributed
            
            # Store: [bsz, seq_m, num_heads, embed_dim]
            self.attentions.append(cls_contrib.cpu())
            
            # Continue forward pass (standard attention output)
            attn_out = torch.matmul(attn, v)  # [bsz, heads, seq, head_dim]
            attn_out = attn_out.permute(2, 0, 1, 3).reshape(seq_len_curr, bsz, embed_dim)
            attn_out = block.attn.out_proj(attn_out)
            
            x = x + attn_out
            x = x + block.mlp(block.ln_2(x))
        
        # Final representation (CLS token after ln_post and proj)
        x = x.permute(1, 0, 2)  # [B, N+1, D]
        x_cls = visual.ln_post(x[:, 0, :])  # [B, D]
        
        if visual.proj is not None:
            representation = x_cls @ visual.proj
        else:
            representation = x_cls
            
        # Stack attentions: [bsz, num_layers, seq_m, num_heads, embed_dim]
        attentions = torch.stack(self.attentions, dim=1)
        
        return attentions, representation, (grid_h, grid_w)


def compute_textspan_with_classifier(
    model,
    image,
    classifier,  # [proj_dim, num_classes] - text embeddings transposed
    target_class_idx,
    device,
):
    """
    TextSpan segmentation following the official implementation.
    
    This is the key function that replicates:
    https://github.com/yossigandelsman/clip_text_span/blob/main/compute_segmentations.py
    
    Args:
        model: CLIP model (LeWrapper)
        image: Input image tensor [1, 3, H, W]
        classifier: Matrix of text embeddings [proj_dim, num_classes]
        target_class_idx: Index of the target class
        device: torch device
        
    Returns:
        heatmap: [H, W] tensor normalized to [0, 1]
    """
    extractor = TextSpanExtractor(model, device)
    visual = model.visual
    
    with torch.no_grad():
        # Extract attention contributions
        attentions, representation, (grid_h, grid_w) = extractor.extract_attention_contributions(image)
        
        # attentions: [bsz, num_layers, seq_m, num_heads, embed_dim]
        bsz = attentions.shape[0]
        num_layers = attentions.shape[1]
        seq_m = attentions.shape[2]
        num_heads = attentions.shape[3]
        embed_dim = attentions.shape[4]
        
        num_patches = seq_m - 1  # Exclude CLS position
        
        # Move to device
        attentions = attentions.to(device).float()
        
        # Take patch positions only (exclude CLS at position 0): [bsz, layers, patches, heads, embed_dim]
        patch_attentions = attentions[:, :, 1:, :, :]
        
        # Sum over layers (dim=1) and heads (dim=3): [bsz, patches, embed_dim]
        attentions_collapse = patch_attentions.sum(dim=(1, 3))
        
        # Project through ln_post and visual.proj
        # Note: The official impl does more complex normalization accounting for
        # contribution to the mean. Here we use the simpler approach.
        attentions_projected = visual.ln_post(attentions_collapse)
        if visual.proj is not None:
            attentions_projected = attentions_projected @ visual.proj  # [bsz, patches, proj_dim]
        
        # Normalize by representation norm (as in official finalize())
        rep_norm = representation.norm(dim=-1, keepdim=True)  # [bsz, 1]
        attentions_projected = attentions_projected / rep_norm.unsqueeze(1)
        
        # Multiply by classifier to get class activations per patch
        class_heatmap = attentions_projected @ classifier.float()  # [bsz, patches, num_classes]
        class_heatmap = class_heatmap.cpu().numpy()
        
        # Key step: subtract mean across all classes
        # This is what makes TextSpan work - relative activation
        results = []
        for i in range(bsz):
            # Get activation for target class
            target_activation = class_heatmap[i, :, target_class_idx]
            # Subtract mean activation across all classes
            mean_activation = np.mean(class_heatmap[i], axis=1)
            normalized = target_activation - mean_activation
            results.append(normalized)
        
        results = np.stack(results, axis=0)  # [bsz, num_patches]
        results = torch.from_numpy(results).reshape(bsz, grid_h, grid_w)
        
        # Upsample to image size
        patch_size = model.visual.conv1.kernel_size[0]
        Res = F.interpolate(
            results[:, None].float(),
            scale_factor=patch_size,
            mode="bilinear",
            align_corners=False,
        ).to(device)
        
        # Clip negative values (as in official impl)
        Res = torch.clip(Res, 0, Res.max())
        
        # Normalize to [0, 1]
        Res = (Res - Res.min()) / (Res.max() - Res.min() + 1e-8)
        
        return Res[0, 0].cpu()


def compute_textspan_simple(
    model,
    image,
    text_embedding,
    all_text_embeddings,
    target_class_idx,
):
    """
    Simplified TextSpan that uses the provided text embeddings as classifier.
    
    This is a convenience wrapper that builds the classifier from all_text_embeddings
    and calls compute_textspan_with_classifier.
    
    Args:
        model: CLIP model (LeWrapper)
        image: Input image tensor [1, 3, H, W]
        text_embedding: Target class embedding [1, embed_dim] (not used directly, for API compat)
        all_text_embeddings: All class embeddings [num_classes, embed_dim]
        target_class_idx: Index of the target class
        
    Returns:
        heatmap: [H, W] tensor normalized to [0, 1]
    """
    device = image.device
    
    # Build classifier matrix: [embed_dim, num_classes]
    classifier = all_text_embeddings.T  # [embed_dim, num_classes]
    
    return compute_textspan_with_classifier(
        model=model,
        image=image,
        classifier=classifier,
        target_class_idx=target_class_idx,
        device=device,
    )


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
# Raw Attention Implementation
# Based on: "Raw Attention baseline considers the attention maps from the 
# last layer, focusing on the weights associated with the [CLS] token.
# The attention heads are averaged and the resulting map is normalized."
#
# E_Attn = norm(resize(1/h * sum_k(A^L_{:,0,1:})))
# =============================================================================

def compute_raw_attention(model, image):
    pass


def compute_chefercam(model, image, text_emb_1x):
    """
    Computes GradCAM on the last Attention layer (attn_gradcam baseline).
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: attn_gradcam - GradCAM applied to attention maps from the last layer
    
    This is the baseline method from Chefer et al., not the full Transformer Attribution.
    For the full method, see compute_transformer_attribution().
    """
    model.zero_grad()
    
    # Get the last transformer block
    blocks = list(model.visual.transformer.resblocks)
    last_block = blocks[-1]
    num_heads = last_block.attn.num_heads
    num_prompts = text_emb_1x.shape[0]
    
    # Forward pass - need to do this in a way that keeps attention maps in the graph
    # We'll manually forward through the model to capture attention with gradients
    with torch.enable_grad():
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
        last_attn = last_block.attn
        x_normed = last_block.ln_1(x)
        
        # Compute Q, K, V
        qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        seq_len, bsz, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        
        q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
        
        # Compute attention weights (this is what we need gradients for)
        scale = float(head_dim) ** -0.5
        attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, N, N]
        
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
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)  # [1, 1]
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradient w.r.t. attention weights
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False)[0]
        
        # Reshape: [bsz*heads, N, N] -> [bsz, heads, N, N]
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
        
        # Reshape to spatial grid
        num_patches = cam.shape[-1]
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
    """
    model.zero_grad()
    
    blocks = list(model.visual.transformer.resblocks)
    num_layers = len(blocks)
    
    # Ensure start_layer is valid
    if start_layer < 0:
        start_layer = num_layers + start_layer
    start_layer = max(0, min(start_layer, num_layers - 1))
    
    num_prompts = text_emb_1x.shape[0]
    
    # Forward pass with gradients enabled to capture attention maps
    with torch.enable_grad():
        # Forward through visual encoder
        x = model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        batch_size = x.shape[0]
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
        all_attn_weights = []
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
        grads = torch.autograd.grad(s, all_attn_weights, retain_graph=False, create_graph=False)
        
        # Process each layer's gradient-weighted attention
        layer_contributions = []
        for i, (grad, attn_weights) in enumerate(zip(grads, all_attn_weights)):
            num_heads = blocks[start_layer + i].attn.num_heads
            
            # Reshape: [bsz*heads, N, N] -> [bsz, heads, N, N]
            grad = grad.view(bsz, num_heads, grad.shape[1], grad.shape[2])
            attn_weights = attn_weights.view(bsz, num_heads, attn_weights.shape[1], attn_weights.shape[2])
            
            # Apply ReLU to gradients
            grad = torch.clamp(grad, min=0)
            
            # Weight attention by gradients
            weighted_attn = grad * attn_weights  # [batch, heads, N, N]
            
            # Average over heads
            weighted_attn = weighted_attn.mean(dim=1)  # [batch, N, N]
            
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
        '--methods', type=str, default='original,gradcam,rollout,textspan,rawattn,chefercam',
        help="Comma-separated methods: original (LeGrad), gradcam, rollout, textspan, rawattn, chefercam, sparse"
    )
    
    # Rollout settings
    parser.add_argument('--rollout_start_layer', type=int, default=0,
                        help='Start layer for attention rollout (0=all layers, 1=skip first)')
    parser.add_argument('--rollout_head_fusion', type=str, default='mean',
                        choices=['mean', 'max', 'min'],
                        help='How to fuse attention heads for rollout')
    
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
    allowed_methods = {'original', 'gradcam', 'rollout', 'textspan', 'rawattn', 'chefercam', 'sparse'}
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods. Use --methods with: original,gradcam,rollout,textspan,rawattn,chefercam,sparse")
    
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
            # Following official implementation: uses ALL class embeddings as classifier
            # and subtracts mean activation across classes
            if 'textspan' in methods:
                heatmap = compute_textspan_simple(
                    model=model,
                    image=img_t,
                    text_embedding=text_emb_1x,
                    all_text_embeddings=all_text_embs,
                    target_class_idx=cls_idx,
                )
                # Already at image resolution from interpolation, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['textspan'] = heatmap_resized
            
            # --- RAW ATTENTION ---
            # Attention from last layer, CLS token to patches, averaged over heads
            if 'rawattn' in methods:
                heatmap = compute_raw_attention(model, img_t)
                # Already at image resolution, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['rawattn'] = heatmap_resized
            
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

