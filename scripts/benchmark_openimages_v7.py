#!/usr/bin/env python3
"""
OpenImagesV7 Segmentation Benchmark (Detailed Breakdown)

Evaluates:
1. Foreground IoU (Target class)
2. Background IoU (Everything else)
3. Mean IoU (Average of FG and BG)

This will help verify if the ~8.75 benchmark refers to FG only or mIoU.
"""

import sys
import os
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import requests
from io import BytesIO

# Import BigQuery
from google.cloud import bigquery

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model components
try:
    from legrad import LeWrapper, LePreprocess
    import open_clip
except ImportError:
    pass

# =============================================================================
# LeGrad Implementation
# =============================================================================

def compute_legrad_for_embedding(model, image, text_emb_1x):
    """Compute LeGrad heatmap for a single text embedding."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits

# =============================================================================
# Attention Rollout (Reference)
# =============================================================================

def compute_rollout_attention(all_layer_matrices, start_layer=1):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device
    
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(device)
    matrices_aug = []
    for mat in all_layer_matrices:
        mat_aug = mat + eye
        mat_aug = mat_aug / mat_aug.sum(dim=-1, keepdim=True)
        matrices_aug.append(mat_aug)
    
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    return joint_attention

def compute_attention_rollout_reference(model, image, start_layer=1):
    blocks = model.visual.transformer.resblocks
    num_heads = blocks[0].attn.num_heads
    embed_dim = blocks[0].attn.embed_dim
    head_dim = embed_dim // num_heads
    
    all_attentions = []
    
    with torch.no_grad():
        x = model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        batch_size = x.shape[0]
        class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
        class_token = class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
            
        if pos_embed.shape[0] != x.shape[1]:
            cls_pos = pos_embed[:1]
            patch_pos = pos_embed[1:]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            new_size = int(math.sqrt(x.shape[1] - 1))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, embed_dim)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        if hasattr(model.visual, 'ln_pre'):
            x = model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        
        for block in blocks:
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
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            attn_reshaped = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            all_attentions.append(attn_reshaped.mean(dim=1))
            
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = attn_module.out_proj(attn_output)
            x = x + attn_output
            x = x + block.mlp(block.ln_2(x))

    rollout = compute_rollout_attention(all_attentions, start_layer=start_layer)
    cls_attn = rollout[:, 0, 1:]
    
    num_patches = cls_attn.shape[-1]
    h = w = int(math.sqrt(num_patches))
    heatmap = cls_attn.reshape(1, 1, h, w)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
    
    return heatmap[0, 0].detach().cpu()

# =============================================================================
# DETAILED METRIC EVALUATION
# =============================================================================

def compute_detailed_miou(heatmap, positive_points, negative_points, threshold=0.5):
    """
    Returns a dictionary with:
    - 'fg_iou': Foreground IoU (Positive Class)
    - 'bg_iou': Background IoU (Negative Class)
    - 'm_iou':  Mean IoU
    """
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    H, W = heatmap_np.shape
    
    # Prediction Masks
    # 1 = Foreground, 0 = Background
    pred_mask = (heatmap_np > threshold).astype(np.int32)
    
    # Counters
    # TP: Pos point in FG mask
    # FN: Pos point in BG mask
    # FP: Neg point in FG mask
    # TN: Neg point in BG mask
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    for y, x in positive_points:
        y, x = int(np.clip(y, 0, H-1)), int(np.clip(x, 0, W-1))
        if pred_mask[y, x] == 1:
            tp += 1
        else:
            fn += 1
            
    for y, x in negative_points:
        y, x = int(np.clip(y, 0, H-1)), int(np.clip(x, 0, W-1))
        if pred_mask[y, x] == 1:
            fp += 1
        else:
            tn += 1
            
    # --- Foreground IoU ---
    # Intersection = TP
    # Union = TP + FP + FN (All Pos Points + Neg Points predicted as Pos)
    union_fg = tp + fp + fn
    iou_fg = (tp / union_fg) if union_fg > 0 else 0.0
    
    # --- Background IoU ---
    # Intersection = TN
    # Union = TN + FN + FP (All Neg Points + Pos Points predicted as Neg)
    union_bg = tn + fn + fp
    iou_bg = (tn / union_bg) if union_bg > 0 else 0.0
    
    # --- Mean IoU ---
    m_iou = (iou_fg + iou_bg) / 2.0
    
    return {
        'fg_iou': iou_fg * 100.0,
        'bg_iou': iou_bg * 100.0,
        'm_iou': m_iou * 100.0
    }

# =============================================================================
# Data Loading
# =============================================================================

def load_openimages_annotations(annotations_csv_path):
    print(f"Loading annotations from {annotations_csv_path}...")
    try:
        df = pd.read_csv(annotations_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}
    
    image_annotations = defaultdict(lambda: {'positive': defaultdict(list), 'negative': defaultdict(list)})
    
    for _, row in df.iterrows():
        image_id = str(row['ImageId'])
        label_id = str(row['Label'])
        class_name = str(row.get('TextLabel', label_id)).strip() if pd.notna(row.get('TextLabel')) else label_id
        
        x, y = float(row['X']), float(row['Y'])
        estimated = str(row.get('EstimatedYesNo', 'yes')).lower()
        
        if estimated == 'yes':
            image_annotations[image_id]['positive'][class_name].append((y, x))
        elif estimated == 'no':
            image_annotations[image_id]['negative'][class_name].append((y, x))
            
    return dict(image_annotations)

def fetch_image_urls_from_bigquery(subset='validation', limit=None):
    print(f"\nConnecting to BigQuery (subset='{subset}')...")
    try:
        client = bigquery.Client()
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
            SELECT image_id, original_url
            FROM `bigquery-public-data.open_images.images`
            WHERE subset = '{subset}'
            {limit_clause}
        """
        df = client.query(query).to_dataframe()
        return dict(zip(df.image_id, df.original_url))
    except Exception as e:
        print(f"BigQuery Error: {e}")
        return {}

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_csv', type=str, default=None)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--method', type=str, default='rollout', choices=['rollout', 'legrad'],
                        help='Method to use: rollout (attention rollout) or legrad (LeGrad)')
    parser.add_argument('--rollout_start_layer', type=int, default=1,
                        help='Start layer for attention rollout (only used when method=rollout)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Locate Annotations
    if not args.annotations_csv:
        for root, _, files in os.walk("/kaggle/input"):
            for file in files:
                if "point" in file.lower() and file.endswith(".csv"):
                    args.annotations_csv = os.path.join(root, file)
                    break
    
    if not args.annotations_csv:
        print("Error: No annotation CSV found.")
        return

    # 2. Load Data
    image_annotations = load_openimages_annotations(args.annotations_csv)
    fetch_limit = 5000 if args.limit < 1000 else args.limit * 2 
    image_id_to_url = fetch_image_urls_from_bigquery(args.subset, fetch_limit)
    
    valid_ids = [i for i in image_annotations.keys() if i in image_id_to_url]
    print(f"Images to process: {len(valid_ids)} (limit: {args.limit})")
    
    # 3. Model
    print(f"Loading {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=device)
    model.eval()
    
    # For LeGrad, wrap model with LeWrapper
    if args.method == 'legrad':
        if 'LeWrapper' not in globals():
            print("Error: LeWrapper not available. Cannot use legrad method.")
            return
        model = LeWrapper(model, layer_index=-2)
        print("Wrapped model with LeWrapper for LeGrad")
    
    if 'LePreprocess' in globals(): 
        preprocess = LePreprocess(preprocess, image_size=224)
    
    # For LeGrad, we need tokenizer for text encoding
    tokenizer = None
    if args.method == 'legrad':
        if 'open_clip' in globals():
            tokenizer = open_clip.get_tokenizer(args.model_name)

    # 4. Evaluate
    results = {'fg': [], 'bg': [], 'mean': []}
    process_ids = valid_ids[:args.limit]
    
    print(f"Running evaluation with method: {args.method}...")
    for idx, image_id in enumerate(tqdm(process_ids)):
        try:
            url = image_id_to_url[image_id]
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200: continue
            
            image = Image.open(BytesIO(resp.content)).convert('RGB')
            img_t = preprocess(image).unsqueeze(0).to(device)
            H_img, W_img = img_t.shape[-2:]
            
            ann = image_annotations[image_id]
            
            # Evaluate against all classes in this image
            for class_name in ann['positive']:
                # Compute heatmap based on method
                if args.method == 'legrad':
                    # LeGrad: encode text and compute heatmap
                    if tokenizer is None:
                        continue
                    prompt = f"a photo of a {class_name}."
                    text_tokens = tokenizer([prompt]).to(device)
                    with torch.no_grad():
                        text_emb = model.encode_text(text_tokens, normalize=True)
                    
                    heatmap = compute_legrad_for_embedding(model, img_t, text_emb)
                    # LeGrad returns heatmap at feature resolution, resize to image size
                    heatmap = F.interpolate(
                        heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                else:
                    # Rollout: compute attention rollout (class-agnostic)
                    heatmap = compute_attention_rollout_reference(model, img_t, start_layer=args.rollout_start_layer)
                    heatmap = F.interpolate(
                        heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
                        size=(H_img, W_img),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                pos_pts = [(int(y*H_img), int(x*W_img)) for y, x in ann['positive'][class_name]]
                neg_pts = [(int(y*H_img), int(x*W_img)) for y, x in ann['negative'].get(class_name, [])]
                
                metrics = compute_detailed_miou(heatmap, pos_pts, neg_pts)
                
                results['fg'].append(metrics['fg_iou'])
                results['bg'].append(metrics['bg_iou'])
                results['mean'].append(metrics['m_iou'])
        
            # Periodic print
            if (idx + 1) % 10 == 0 and len(results['fg']) > 0:
                print(f"  [Interim {idx+1}] FG: {np.mean(results['fg']):.2f} | BG: {np.mean(results['bg']):.2f} | Mean: {np.mean(results['mean']):.2f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*50)
    print(f"FINAL RESULTS ({args.method.upper()})")
    print("="*50)
    if results['fg']:
        fg = np.mean(results['fg'])
        bg = np.mean(results['bg'])
        m = np.mean(results['mean'])
        
        print(f"Foreground IoU (FG): {fg:.4f}")
        print(f"Background IoU (BG): {bg:.4f}")
        print(f"Mean IoU (mIoU):     {m:.4f}")
        print("-" * 30)
        print("Note: If the paper reports ~8.75, it likely refers to 'Foreground IoU'.")
    else:
        print("No results computed.")

if __name__ == "__main__":
    main()