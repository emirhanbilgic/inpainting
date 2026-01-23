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


def compute_gradcam_heatmap(model, image, text_emb_1x, layer_index: int = 8):
    """Compute GradCAM heatmap for a single text embedding."""
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


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
        
        # --- Common Cam Calculation ---
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)  # [1, 1]
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradient w.r.t. attention weights
        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False)[0]
        
        # Apply ReLU
        grad = torch.clamp(grad, min=0)
        
        # Weight attention map by gradients
        cam = grad * attn_weights  # [batch, heads, N, N]
        
        # Average over heads
        cam = cam.mean(dim=1)  # [batch, N, N]
        
        # Extract attention to patches
        # For CLIP: row 0 is CLS, cols 1: are patches
        # For SigLIP: no CLS token usually, purely spatial. But check architecture.
        # SigLIP doesn't use CLS token for pooling, it uses MAP (Multi-head Attention Pooling)
        # However, timm implementation might differ.
        
        if is_timm:
            # SigLIP (Attentional Pooler)
            # cam shape: [B, latent_len, N]
            # Average over latents/queries to get importance of spatial tokens
            cam = cam.mean(dim=1) # [B, N]
        else:
            # CLIP (Self Attention)
            # cam shape: [B, N+1, N+1]
            # Extract CLS token attention to patches (row 0, cols 1:)
            cam = cam[:, 0, 1:]  # [batch, num_patches]
            
        # Reshape to spatial grid
        num_patches = cam.shape[-1]
        grid_size = int(math.sqrt(num_patches))
        
        # Handle non-square if needed, but usually square
        w = h = grid_size
        
        heatmap = cam.reshape(text_emb_1x.shape[0], 1, h, w)
        
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
    
    def evaluate(self, threshold=0.5, show_progress=True):
        """
        Evaluate LeGrad baseline on correct and wrong prompts.
        
        Returns dict with:
        - correct: {miou, acc, map, max, mean, median, min, auroc}
        - wrong: {miou, acc, map, max, mean, median, min, auroc}
        """
        correct_results = {
            'iou': [], 'acc': [], 'ap': [], 
            'max': [], 'mean': [], 'median': [], 'min': [], 'auroc': []
        }
        wrong_results = {
            'iou': [], 'acc': [], 'ap': [],
            'max': [], 'mean': [], 'median': [], 'min': [], 'auroc': []
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
                    
                    # Normalize heatmap [0, 1] - essential for fair comparison with benchmark script
                    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    
                    if self.threshold_mode == 'mean':
                        thr = heatmap_norm.mean().item()
                    else:
                        thr = self.fixed_threshold
                    
                    heatmap_resized = F.interpolate(
                        heatmap_norm.view(1, 1, H_feat, W_feat),
                        size=(H_gt, W_gt),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().numpy()
                    
                    iou, acc = compute_iou_acc(heatmap_resized, gt_mask, threshold=thr)
                    ap = compute_map_score(heatmap_resized, gt_mask)
                    
                    # New metrics
                    max_val = np.max(heatmap_resized)
                    mean_val = np.mean(heatmap_resized)
                    median_val = np.median(heatmap_resized)
                    min_val = np.min(heatmap_resized)
                    
                    # AUROC
                    gt_binary = (gt_mask > 0).astype(int).flatten()
                    pred_flat = heatmap_resized.flatten()
                    
                    if len(np.unique(gt_binary)) > 1:
                        auroc = roc_auc_score(gt_binary, pred_flat)
                    else:
                        auroc = np.nan
                    
                    return iou, acc, ap, max_val, mean_val, median_val, min_val, auroc
                
                # === CORRECT PROMPT ===
                iou, acc, ap, mx, mn, md, mi, auc = compute_metrics(text_emb)
                correct_results['iou'].append(iou)
                correct_results['acc'].append(acc)
                correct_results['ap'].append(ap)
                correct_results['max'].append(mx)
                correct_results['mean'].append(mn)
                correct_results['median'].append(md)
                correct_results['min'].append(mi)
                if not np.isnan(auc):
                    correct_results['auroc'].append(auc)
                
                # DEBUG: Print per-image correct metrics (first 3 only)
                if idx < 3:
                    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"
                    print(f"  CORRECT IoU: {iou:.4f}, Acc: {acc:.4f}, mAP: {ap:.4f}, AUROC: {auc_str}")
                correct_iou_this_image = iou
                
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
                    
                    iou, acc, ap, mx, mn, md, mi, auc = compute_metrics(neg_emb)
                    wrong_results['iou'].append(iou)
                    wrong_results['acc'].append(acc)
                    wrong_results['ap'].append(ap)
                    wrong_results['max'].append(mx)
                    wrong_results['mean'].append(mn)
                    wrong_results['median'].append(md)
                    wrong_results['min'].append(mi)
                    if not np.isnan(auc):
                        wrong_results['auroc'].append(auc)
                    
                    # DEBUG: Print per-image wrong metrics and comparison (first 3 only)
                    if idx < 3:
                        comparison = "GOOD (correct > wrong)" if correct_iou_this_image > iou else "BAD (wrong > correct)"
                        print(f"  WRONG IoU ({neg_class}): {iou:.4f} -- {comparison}")
                
                
            except Exception as e:
                print(f"[Warning] Error processing image {idx}: {e}")
                continue
        
        # Compute averages
        results = {
            'correct': {
                'miou': np.mean(correct_results['iou']) * 100 if correct_results['iou'] else 0.0,
                'acc': np.mean(correct_results['acc']) * 100 if correct_results['acc'] else 0.0,
                'map': np.mean(correct_results['ap']) * 100 if correct_results['ap'] else 0.0,
                'max': np.mean(correct_results['max']) if correct_results['max'] else 0.0,
                'mean': np.mean(correct_results['mean']) if correct_results['mean'] else 0.0,
                'median': np.mean(correct_results['median']) if correct_results['median'] else 0.0,
                'min': np.mean(correct_results['min']) if correct_results['min'] else 0.0,
                'auroc': np.mean(correct_results['auroc']) * 100 if correct_results['auroc'] else 0.0,
                'n_samples': len(correct_results['iou']),
            },
            'wrong': {
                'miou': np.mean(wrong_results['iou']) * 100 if wrong_results['iou'] else 0.0,
                'acc': np.mean(wrong_results['acc']) * 100 if wrong_results['acc'] else 0.0,
                'map': np.mean(wrong_results['ap']) * 100 if wrong_results['ap'] else 0.0,
                'max': np.mean(wrong_results['max']) if wrong_results['max'] else 0.0,
                'mean': np.mean(wrong_results['mean']) if wrong_results['mean'] else 0.0,
                'median': np.mean(wrong_results['median']) if wrong_results['median'] else 0.0,
                'min': np.mean(wrong_results['min']) if wrong_results['min'] else 0.0,
                'auroc': np.mean(wrong_results['auroc']) * 100 if wrong_results['auroc'] else 0.0,
                'n_samples': len(wrong_results['iou']),
            },
        }
        
        return results


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
    print(f"Threshold: {args.threshold}")
    print(f"{'='*60}\n")
    
    results = evaluator.evaluate(threshold=args.threshold)
    
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
            'threshold': args.threshold,
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
