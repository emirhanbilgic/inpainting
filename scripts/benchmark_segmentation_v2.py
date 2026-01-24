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







def compute_chefercam(model, image, text_emb_1x):
    """
    Computes GradCAM on the last Attention layer (attn_gradcam baseline).
    
    Reference: https://github.com/hila-chefer/Transformer-Explainability
    Method: attn_gradcam - GradCAM applied to attention maps from the last layer
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
        '--methods', type=str, default='original,gradcam,chefercam',
        help="Comma-separated methods: original (LeGrad), gradcam, chefercam"
    )
    
    # Threshold settings
    parser.add_argument('--threshold_mode', type=str, default='mean',
                        choices=['mean', 'fixed'],
                        help='Thresholding mode: "mean" (per-image adaptive as in reference) or "fixed"')
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='Fixed threshold value when threshold_mode=fixed')
    
    # Visualization
    parser.add_argument('--vis_first_k', type=int, default=0)
    parser.add_argument('--vis_output_dir', type=str, default='outputs/segmentation_vis_v2')
    
    args = parser.parse_args()

    # Parse methods
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    allowed_methods = {'original', 'gradcam', 'chefercam'}
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods. Use --methods with: original,gradcam,chefercam")
    
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
            
            # --- CHEFERCAM ---
            # GradCAM applied to attention maps from the last layer
            if 'chefercam' in methods:
                heatmap = compute_chefercam(model, img_t, text_emb_1x)
                
                # Already at image resolution, resize to GT size
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                heatmaps['chefercam'] = heatmap_resized

            
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

