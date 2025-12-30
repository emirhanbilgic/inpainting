#!/usr/bin/env python3
"""
OpenImagesV7 Point-wise Segmentation Benchmark (Fixed for Kaggle BigQuery)

- Uses BigQuery to get Image URLs (Fixes dataset detection).
- Implements LeGrad correctly using text embeddings.
- Keeps Attention Rollout for comparison.
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

# Kaggle BigQuery
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
    print("Error: 'legrad' or 'open_clip' not found. Please install them or check python path.")
    sys.exit(1)

# =============================================================================
# 1. Attention Rollout Implementation (Kept as requested)
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

def compute_attention_rollout_heatmap(model, image, start_layer=0):
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
        
        # Interpolate pos encoding if needed
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
    
    # Normalize min-max
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize to image size
    heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
    
    return heatmap[0, 0].detach().cpu()

# =============================================================================
# 2. LeGrad Implementation (Fixed)
# =============================================================================

def compute_legrad_heatmap(model, image, text_embedding):
    """
    Computes LeGrad heatmap using the specific text embedding.
    """
    with torch.enable_grad():
        # image: [1, 3, H, W]
        # text_embedding: [1, D]
        logits = model.compute_legrad(image=image, text_embedding=text_embedding)
    
    # logits shape: [1, 1, h_feat, w_feat] (typically 14x14 or 24x24)
    logits = logits[0, 0] # [h_feat, w_feat]
    
    # Clamp and detach
    heatmap = logits.clamp(0, 1).detach().cpu()
    
    # Resize to match input image resolution for evaluation
    # We need to add dims back for interpolate: [1, 1, h, w]
    H_img, W_img = image.shape[-2:]
    heatmap = F.interpolate(
        heatmap.view(1, 1, heatmap.shape[0], heatmap.shape[1]),
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False
    )
    
    return heatmap[0, 0]

# =============================================================================
# 3. Evaluation Metrics
# =============================================================================

def compute_pointwise_miou(heatmap, positive_points, negative_points, threshold=0.5):
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    H, W = heatmap_np.shape
    pred_mask = (heatmap_np > threshold).astype(np.int32)
    
    tp_fg = 0
    fn_fg = 0
    fp_fg = 0
    tn_fg = 0
    
    # Check positive points (Should be 1)
    for y, x in positive_points:
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        if pred_mask[y, x] == 1:
            tp_fg += 1
        else:
            fn_fg += 1
    
    # Check negative points (Should be 0)
    for y, x in negative_points:
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        if pred_mask[y, x] == 1:
            fp_fg += 1
        else:
            tn_fg += 1
    
    # IoU Foreground
    union_fg = tp_fg + fp_fg + fn_fg
    iou_fg = tp_fg / (union_fg + 1e-8)
    
    # IoU Background
    union_bg = tn_fg + fp_fg + fn_fg
    iou_bg = tn_fg / (union_bg + 1e-8)
    
    return (iou_fg + iou_bg) / 2.0

# =============================================================================
# 4. Data Loading (BigQuery & CSV)
# =============================================================================

def load_openimages_annotations(annotations_csv_path):
    print(f"Loading annotations from {annotations_csv_path}...")
    # Use pandas engine='python' or dtype specification to avoid warnings
    df = pd.read_csv(annotations_csv_path, dtype={'ImageId': str, 'Label': str})
    
    # Structure: ImageID -> ClassName -> {'positive': [], 'negative': []}
    image_annotations = defaultdict(lambda: defaultdict(lambda: {'positive': [], 'negative': []}))
    
    for _, row in df.iterrows():
        image_id = str(row['ImageId'])
        
        # Get readable class name
        if 'TextLabel' in row and pd.notna(row['TextLabel']):
            class_name = str(row['TextLabel']).strip()
        else:
            # Skip if we don't have a readable text label for the prompt
            continue
            
        x, y = float(row['X']), float(row['Y'])
        # EstimatedYesNo usually indicates if the label is machine-generated (less reliable), 
        # but PointLabels are usually human verified. We check the 'EstimatedYesNo' if it exists.
        estimated = str(row.get('EstimatedYesNo', 'yes')).lower()
        
        # Point annotations logic
        # Usually point labels file has columns: ImageId, Label, X, Y, Measured, etc.
        # We assume standard OpenImages Point format.
        
        # In Point Labels, usually there is no 'EstimatedYesNo', but specific columns.
        # Assuming the CSV provided is the Point-Level labels.
        # If it's a generic CSV, we trust X, Y.
        
        # Heuristic: 
        # In OpenImages Point Labels: 
        #   Confidence=1 -> Positive
        #   Confidence=0 -> Negative
        # If 'EstimatedYesNo' is present (Image-level labels), 'yes' is positive.
        
        is_positive = True
        if 'Confidence' in row:
            if int(row['Confidence']) == 0:
                is_positive = False
        elif estimated == 'no':
            is_positive = False
            
        if is_positive:
            image_annotations[image_id][class_name]['positive'].append((y, x))
        else:
            image_annotations[image_id][class_name]['negative'].append((y, x))
            
    return image_annotations

def get_bigquery_image_urls(limit=1000):
    """
    Queries Kaggle's BigQuery OpenImages dataset for Validation URLs.
    """
    print("\nQuerying BigQuery for image URLs...")
    client = bigquery.Client()
    
    # We query more than the limit to ensure we find intersections with annotations
    query = f"""
        SELECT image_id, original_url
        FROM `bigquery-public-data.open_images.images`
        WHERE subset = 'validation'
    """
    if limit:
        # Fetching a larger pool because we need intersection with annotations
        query += f" LIMIT {limit * 10}" 
        
    try:
        df = client.query(query).to_dataframe()
        print(f"BigQuery returned {len(df)} rows.")
        # Return dict: ImageId -> URL
        return pd.Series(df.original_url.values, index=df.image_id).to_dict()
    except Exception as e:
        print(f"BigQuery Error: {e}")
        return {}

# =============================================================================
# 5. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_csv', type=str, default=None)
    parser.add_argument('--limit', type=int, default=100, help="Number of images to evaluate")
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448, help="Resolution for LeGrad")
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Locate Annotations
    if not args.annotations_csv:
        # Try to find the file automatically in Kaggle input
        for root, _, files in os.walk("/kaggle/input"):
            for file in files:
                if "point_labels" in file.lower() and "validation" in file.lower() and file.endswith(".csv"):
                    args.annotations_csv = os.path.join(root, file)
                    break
            if args.annotations_csv: break
    
    if not args.annotations_csv:
        print("Error: Could not auto-locate 'point_labels' CSV. Please provide --annotations_csv")
        return

    # 2. Load Annotations
    ann_dict = load_openimages_annotations(args.annotations_csv)
    print(f"Found annotations for {len(ann_dict)} images.")
    
    # 3. Get URLs via BigQuery
    # We fetch enough to likely cover the limit
    id_to_url = get_bigquery_image_urls(limit=args.limit)
    
    # 4. Filter intersection
    valid_ids = [i for i in ann_dict.keys() if i in id_to_url]
    print(f"Images with both Annotations and URLs: {len(valid_ids)}")
    
    if not valid_ids:
        print("CRITICAL: No intersection between annotations and BigQuery URLs. Check 'subset' or CSV file.")
        return

    process_ids = valid_ids[:args.limit]
    
    # 5. Load Model
    print(f"\nLoading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    
    # Wrap model for LeGrad (layer_index -2 is standard)
    model = LeWrapper(model, layer_index=-2)
    # Use LePreprocess to handle arbitrary sizes (like 448) cleanly
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    
    # 6. Evaluation Loop
    results = {
        'rollout_miou': [],
        'legrad_miou': []
    }
    
    print(f"\nProcessing {len(process_ids)} images...")
    
    for image_id in tqdm(process_ids):
        url = id_to_url[image_id]
        
        # A. Download Image
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert('RGB')
        except Exception:
            # print(f"Failed to download {image_id}")
            continue

        # B. Preprocess
        try:
            # [1, 3, H, W]
            img_t = preprocess(image).unsqueeze(0).to(device)
            H_img, W_img = img_t.shape[-2:] # Tensor dims
            H_orig, W_orig = image.size[1], image.size[0] # Original dims for metric calc
            
            # Annotations for this image
            # ann_dict[id] is a dict: ClassName -> {'positive': list, 'negative': list}
            img_anns = ann_dict[image_id]
            
            for class_name, points in img_anns.items():
                pos_norm = points['positive']
                neg_norm = points['negative']
                
                # Convert normalized coords (0-1) to Tensor dims for extraction, 
                # but metrics are scale-invariant so we can use tensor dims or orig dims.
                # Let's map normalized points to the tensor H,W so they match the heatmap.
                pos_pts = [(int(y * H_img), int(x * W_img)) for y, x in pos_norm]
                neg_pts = [(int(y * H_img), int(x * W_img)) for y, x in neg_norm]
                
                # --- Method 1: Attention Rollout (Class Agnostic/Generic) ---
                # Rollout is technically class-agnostic in standard implementation, 
                # heavily biased towards the most salient object.
                heatmap_rollout = compute_attention_rollout_heatmap(model, img_t)
                miou_rollout = compute_pointwise_miou(heatmap_rollout, pos_pts, neg_pts, args.threshold)
                results['rollout_miou'].append(miou_rollout)
                
                # --- Method 2: LeGrad (Class Specific) ---
                # 1. Create text embedding
                text_prompt = f"a photo of a {class_name.lower()}."
                text_input = tokenizer([text_prompt]).to(device)
                
                with torch.no_grad():
                    text_embedding = model.encode_text(text_input, normalize=True)
                
                # 2. Compute LeGrad
                heatmap_legrad = compute_legrad_heatmap(model, img_t, text_embedding)
                miou_legrad = compute_pointwise_miou(heatmap_legrad, pos_pts, neg_pts, args.threshold)
                results['legrad_miou'].append(miou_legrad)
                
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    # 7. Final Results
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    
    if results['rollout_miou']:
        print(f"Attention Rollout mIoU: {np.mean(results['rollout_miou']):.4f}")
        print(f"LeGrad mIoU:            {np.mean(results['legrad_miou']):.4f}")
        print(f"Total Samples:          {len(results['legrad_miou'])}")
    else:
        print("No successful samples processed.")

if __name__ == "__main__":
    main()