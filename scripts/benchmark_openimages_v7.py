#!/usr/bin/env python3
"""
OpenImagesV7 Point-wise Segmentation Benchmark (Fixed with BigQuery)

This script evaluates explainability methods on the OpenImagesV7 validation set.
It uses Google BigQuery to fetch image URLs dynamically, solving directory detection issues.
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

# Import model components (Graceful fail if libraries missing)
try:
    from legrad import LeWrapper, LePreprocess
    import open_clip
except ImportError:
    pass

# =============================================================================
# Attention Rollout Implementation
# =============================================================================

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device
    
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    
    return joint_attention

def compute_attention_rollout_reference(model, image, start_layer=0):
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
# Point-wise mIoU Evaluation
# =============================================================================

def compute_pointwise_miou(heatmap, positive_points, negative_points, threshold=0.5):
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    H, W = heatmap_np.shape
    pred_mask = (heatmap_np > threshold).astype(np.int32)
    
    tp_fg = 0
    fp_fg = 0
    fn_fg = 0
    tn_fg = 0
    
    for y, x in positive_points:
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        if pred_mask[y, x] == 1:
            tp_fg += 1
        else:
            fn_fg += 1
    
    for y, x in negative_points:
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        if pred_mask[y, x] == 1:
            fp_fg += 1
        else:
            tn_fg += 1
    
    union_fg = tp_fg + fp_fg + fn_fg
    iou_fg = tp_fg / (union_fg + 1e-8)
    
    union_bg = tn_fg + fp_fg + fn_fg
    iou_bg = tn_fg / (union_bg + 1e-8)
    
    return (iou_fg + iou_bg) / 2.0

# =============================================================================
# Data Loading & Parsing
# =============================================================================

def load_openimages_annotations(annotations_csv_path):
    print(f"Loading annotations from {annotations_csv_path}...")
    try:
        df = pd.read_csv(annotations_csv_path)
    except Exception as e:
        print(f"Error reading annotation CSV: {e}")
        return {}
    
    image_annotations = defaultdict(lambda: {'positive': defaultdict(list), 'negative': defaultdict(list)})
    
    # Check if necessary columns exist
    required_cols = ['ImageId', 'Label', 'X', 'Y']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing columns. Found: {df.columns}")
        return {}

    for _, row in df.iterrows():
        image_id = str(row['ImageId'])
        label_id = str(row['Label'])
        
        # Use TextLabel if available, else Label ID
        if 'TextLabel' in row and pd.notna(row['TextLabel']):
            class_name = str(row['TextLabel']).strip()
        else:
            class_name = label_id
            
        x, y = float(row['X']), float(row['Y'])
        estimated = str(row.get('EstimatedYesNo', 'yes')).lower()
        
        if estimated == 'yes':
            image_annotations[image_id]['positive'][class_name].append((y, x))
        elif estimated == 'no':
            image_annotations[image_id]['negative'][class_name].append((y, x))
            
    return dict(image_annotations)

def fetch_image_urls_from_bigquery(subset='validation', limit=None):
    """
    Fetches image URLs directly from the BigQuery public dataset.
    """
    print(f"\nConnecting to BigQuery to fetch image URLs (subset='{subset}')...")
    
    try:
        client = bigquery.Client()
        
        # Construct Query
        # If limit is None, we fetch all (approx 41k for validation)
        # We only need ImageID and the URL
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT image_id, original_url
            FROM `bigquery-public-data.open_images.images`
            WHERE subset = '{subset}'
            {limit_clause}
        """
        
        df = client.query(query).to_dataframe()
        
        if df.empty:
            print("Warning: BigQuery returned no results.")
            return {}

        print(f"Successfully retrieved {len(df)} URLs from BigQuery.")
        
        # Convert to dictionary {image_id: url}
        return dict(zip(df.image_id, df.original_url))

    except Exception as e:
        print(f"BigQuery Error: {e}")
        print("Ensure 'Internet' is enabled in Kaggle Notebook settings.")
        return {}

# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_csv', type=str, default=None, help="Path to point-labels CSV")
    parser.add_argument('--limit', type=int, default=100, help="Number of images to process")
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--subset', type=str, default='validation', help="OpenImages subset (validation/test/train)")
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Annotations
    # Auto-detect annotation file if not provided
    if not args.annotations_csv:
        search_paths = ["/kaggle/input", "."]
        for path in search_paths:
            for root, _, files in os.walk(path):
                for file in files:
                    if "point_labels" in file.lower() and file.endswith(".csv"):
                        args.annotations_csv = os.path.join(root, file)
                        break
                if args.annotations_csv: break
    
    if not args.annotations_csv:
        print("Error: Annotation CSV not found. Please specify --annotations_csv")
        # For testing without CSV, one might comment this return out, but benchmarks need labels.
        return
        
    image_annotations = load_openimages_annotations(args.annotations_csv)
    print(f"Loaded annotations for {len(image_annotations)} images.")
    
    # 2. Load Metadata (URLs) via BigQuery
    # We fetch enough URLs to cover the limit, or all validation URLs to ensure intersection
    fetch_limit = None if args.limit > 1000 else 5000 # Optimization: fetch more than limit to ensure intersection
    image_id_to_url = fetch_image_urls_from_bigquery(subset=args.subset, limit=fetch_limit)
    
    if not image_id_to_url:
        print("Critical Error: Failed to fetch URLs. Exiting.")
        return

    # 3. Model Setup
    print(f"\nLoading model {args.model_name}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=device
        )
        tokenizer = open_clip.get_tokenizer(args.model_name)
        model.eval()
        # Initialize LeGrad wrapper
        model = LeWrapper(model, layer_index=-2)
        preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    except NameError:
        print("Error: 'open_clip' or 'legrad' not imported. Check dependencies.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 4. Filter images (Intersection of Annotations and Available URLs)
    valid_ids = [i for i in image_annotations.keys() if i in image_id_to_url]
    print(f"\nOverlap Analysis:")
    print(f"  Annotation IDs: {len(image_annotations)}")
    print(f"  URL IDs:        {len(image_id_to_url)}")
    print(f"  Valid Intersection: {len(valid_ids)}")
    
    if len(valid_ids) == 0:
        print("No valid images found for processing (Intersection is 0).")
        print("Try increasing the BigQuery limit or checking if 'subset' matches annotation file.")
        return

    # 5. Process Loop
    results = []
    # If args.limit is smaller than valid_ids, slice it
    process_ids = valid_ids[:args.limit] if args.limit > 0 else valid_ids
    
    print(f"Processing {len(process_ids)} images...")
    
    for image_id in tqdm(process_ids):
        try:
            url = image_id_to_url[image_id]
            
            # Download Image on the fly
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
                image = Image.open(BytesIO(resp.content)).convert('RGB')
            except Exception:
                # Silent fail for network issues
                continue
                
            img_t = preprocess(image).unsqueeze(0).to(device)
            H_img, W_img = img_t.shape[-2:]
            
            ann = image_annotations[image_id]
            classes = list(ann['positive'].keys())
            
            for class_name in classes:
                # Convert normalized coordinates to pixel coordinates
                pos_pts = [(int(y*H_img), int(x*W_img)) for y, x in ann['positive'][class_name]]
                neg_pts = [(int(y*H_img), int(x*W_img)) for y, x in ann['negative'].get(class_name, [])]
                
                # Compute Heatmap
                heatmap = compute_attention_rollout_reference(model, img_t)
                
                heatmap = F.interpolate(
                    heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                    size=(H_img, W_img),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                miou = compute_pointwise_miou(heatmap, pos_pts, neg_pts, threshold=args.threshold)
                results.append(miou)
                
        except Exception as e:
            # print(f"Error processing {image_id}: {e}")
            continue

    if results:
        print(f"\nFinal Point-wise mIoU: {np.mean(results):.4f} (N={len(results)})")
    else:
        print("No successful evaluations. (Check internet connection or URL validity)")

if __name__ == "__main__":
    main()