#!/usr/bin/env python3
"""
OpenImagesV7 Point-wise Segmentation Benchmark

This script evaluates explainability methods on the OpenImagesV7 validation set
using point-wise annotations. Following the protocol from the paper:
- Generate heatmap for each class object present in the image
- Binarize heatmaps using threshold 0.5
- Evaluate using point-wise annotations (positive and negative points)
- Compute p-mIoU (point-wise mean IoU)

The evaluation focuses exclusively on classes that are actually depicted in each image.

Dataset:
  OpenImagesV7 validation set contains 36,702 images with 5,827 unique class labels.
  
  Option 1: Use Kaggle dataset (recommended for Kaggle):
    - Images are in /kaggle/input/open-images/images/
    - Use --use_kaggle flag (auto-detects if path exists)
  
  Option 2: Use local images:
    - Download images to --data_root directory
    - Images should be named as {ImageId}.jpg
  
  Point annotations CSV (oidv7-val-annotations-point-labels.csv):
    - ImageId: Image identifier
    - X, Y: Normalized coordinates [0, 1]
    - Label: Class ID (e.g., /m/01xc8d)
    - EstimatedYesNo: yes/no/unsure (yes = positive, no = negative)
    - TextLabel: Optional text description
  
  Optional: class-descriptions.csv for better label names

Usage on Kaggle:
  python scripts/benchmark_openimages_v7.py \
    --annotations_csv /kaggle/input/open-images/oidv7-val-annotations-point-labels.csv \
    --use_kaggle \
    --methods rollout \
    --model_name ViT-B-16 \
    --pretrained laion2b_s34b_b88k

Usage with local images:
  python scripts/benchmark_openimages_v7.py \
    --data_root /path/to/openimages/validation \
    --annotations_csv ~/Downloads/oidv7-val-annotations-point-labels.csv \
    --methods rollout \
    --model_name ViT-B-16 \
    --pretrained laion2b_s34b_b88k
"""

import sys
import os
import argparse
import math
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import pandas as pd
from collections import defaultdict

# Kaggle dataset paths
KAGGLE_IMAGES_DIR = "/kaggle/input/open-images/images"
KAGGLE_ANNOTATIONS_DIR = "/kaggle/input/open-images/annotations_bbox"
KAGGLE_LABELS_DIR = "/kaggle/input/open-images/labels"
KAGGLE_DICT_DIR = "/kaggle/input/open-images/dict"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess
import open_clip


# =============================================================================
# Attention Rollout Implementation
# =============================================================================

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Compute attention rollout between layers.
    
    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers"
    Implementation adapted from hila-chefer/Transformer-Explainability
    
    Args:
        all_layer_matrices: List of attention matrices [batch, N, N] (averaged over heads)
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


def compute_attention_rollout_reference(model, image, start_layer=0):
    """
    Compute Attention Rollout following the reference implementation.
    
    This implementation:
    1. Uses start_layer (default 0, but reference often uses 1)
    2. Averages attention heads before rollout
    3. Adds residual connections and renormalizes
    
    Reference: hila-chefer/Transformer-Explainability
    
    Args:
        model: LeWrapper model
        image: Input image tensor [1, 3, H, W]
        start_layer: Layer to start rollout (default 0)
    
    Returns:
        Heatmap tensor [H, W] normalized to [0, 1]
    """
    blocks = model.visual.transformer.resblocks
    num_heads = blocks[0].attn.num_heads
    embed_dim = blocks[0].attn.embed_dim
    head_dim = embed_dim // num_heads
    
    all_attentions = []
    
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
        num_patches = x.shape[1] - 1
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
        
        # Resize positional embedding if needed
        if pos_embed.shape[0] != x.shape[1]:
            # Interpolate positional embedding
            cls_pos = pos_embed[:1]  # [1, embed_dim]
            patch_pos = pos_embed[1:]  # [N_orig, embed_dim]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            new_size = int(math.sqrt(num_patches))
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
    if grid_size * grid_size != num_patches:
        # Handle non-square grids
        h = w = int(math.sqrt(num_patches))
        if h * w != num_patches:
            raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
    else:
        h = w = grid_size
    
    heatmap = cls_attn.reshape(1, 1, h, w)
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Upsample to image size
    heatmap = F.interpolate(
        heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False
    )
    
    return heatmap[0, 0].detach().cpu()


# =============================================================================
# Point-wise mIoU Evaluation
# =============================================================================

def compute_pointwise_miou(heatmap, positive_points, negative_points, threshold=0.5):
    """
    Compute point-wise mIoU for a heatmap given positive and negative point annotations.
    
    The metric evaluates:
    - True positives: positive points that fall in predicted foreground (heatmap > threshold)
    - True negatives: negative points that fall in predicted background (heatmap <= threshold)
    - False positives: negative points that fall in predicted foreground
    - False negatives: positive points that fall in predicted background
    
    Then computes IoU for foreground and background classes, and returns mean IoU.
    
    Args:
        heatmap: [H, W] tensor with values in [0, 1]
        positive_points: List of (y, x) tuples for positive annotations
        negative_points: List of (y, x) tuples for negative annotations
        threshold: Threshold for binarization (default 0.5)
    
    Returns:
        p_miou: Point-wise mean IoU (float)
    """
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    H, W = heatmap_np.shape
    
    # Binarize heatmap
    pred_mask = (heatmap_np > threshold).astype(np.int32)
    
    # Create ground truth mask from points
    gt_mask = np.zeros((H, W), dtype=np.int32)
    
    # Mark positive points as foreground (1)
    for y, x in positive_points:
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        gt_mask[y, x] = 1
    
    # Mark negative points as background (0) - they're already 0, but we track them
    # for evaluation purposes
    
    # Compute confusion matrix at point locations
    # We only evaluate at the annotated points
    all_points = positive_points + negative_points
    
    if len(all_points) == 0:
        return 0.0
    
    # Collect predictions and ground truth at point locations
    tp_fg = 0  # True positive foreground: positive point in predicted foreground
    fp_fg = 0  # False positive foreground: negative point in predicted foreground
    fn_fg = 0  # False negative foreground: positive point in predicted background
    tn_fg = 0  # True negative foreground: negative point in predicted background
    
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
    
    # Compute IoU for foreground class
    intersection_fg = tp_fg
    union_fg = tp_fg + fp_fg + fn_fg
    iou_fg = intersection_fg / (union_fg + 1e-8)
    
    # Compute IoU for background class
    intersection_bg = tn_fg
    union_bg = tn_fg + fp_fg + fn_fg
    iou_bg = intersection_bg / (union_bg + 1e-8)
    
    # Mean IoU
    p_miou = (iou_fg + iou_bg) / 2.0
    
    return p_miou


# =============================================================================
# OpenImagesV7 Dataset Loading
# =============================================================================

def load_openimages_annotations(annotations_csv_path, class_descriptions_csv=None):
    """
    Load OpenImagesV7 point-wise annotations from CSV.
    
    Handles the official OpenImagesV7 format:
    - ImageId: Image identifier
    - X, Y: Normalized coordinates [0, 1]
    - Label: Class ID (e.g., /m/01xc8d)
    - EstimatedYesNo: yes/no/unsure (yes = positive, no = negative)
    - TextLabel: Optional text description
    
    Args:
        annotations_csv_path: Path to oidv7-val-annotations-point-labels.csv
        class_descriptions_csv: Optional path to class-descriptions.csv for label mapping
    
    Returns:
        Dictionary mapping image_id -> {
            'positive': {class_label: [(y, x), ...], ...},
            'negative': {class_label: [(y, x), ...], ...}
        }
        And a mapping from class_id (e.g., /m/01xc8d) to class_name
    """
    print(f"Loading annotations from {annotations_csv_path}...")
    
    try:
        df = pd.read_csv(annotations_csv_path)
        print(f"Loaded {len(df)} annotation rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Load class descriptions if provided
        # Download from: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
        label_to_name = {}
        if class_descriptions_csv and os.path.exists(class_descriptions_csv):
            print(f"Loading class descriptions from {class_descriptions_csv}...")
            try:
                class_df = pd.read_csv(class_descriptions_csv, header=None, names=['Label', 'Name'])
                label_to_name = dict(zip(class_df['Label'], class_df['Name']))
                print(f"Loaded {len(label_to_name)} class descriptions")
            except Exception as e:
                print(f"Warning: Could not load class descriptions: {e}")
                print("Will use TextLabel column or label IDs as class names")
        
        # Group by image
        image_annotations = defaultdict(lambda: {'positive': defaultdict(list), 'negative': defaultdict(list)})
        
        # Process annotations
        for _, row in df.iterrows():
            image_id = str(row['ImageId'])
            label_id = str(row['Label'])  # e.g., /m/01xc8d
            
            # Get class name (use TextLabel if available, otherwise use label_id or lookup)
            if 'TextLabel' in row and pd.notna(row['TextLabel']) and str(row['TextLabel']).strip():
                class_name = str(row['TextLabel']).strip()
            elif label_id in label_to_name:
                class_name = label_to_name[label_id]
            else:
                # Fallback to label_id
                class_name = label_id
            
            # Get coordinates (normalized [0, 1])
            x = float(row['X'])
            y = float(row['Y'])
            
            # Determine if positive or negative based on EstimatedYesNo
            estimated = str(row.get('EstimatedYesNo', 'yes')).lower()
            if estimated == 'yes':
                is_positive = True
            elif estimated == 'no':
                is_positive = False
            else:
                # unsure - skip for now, or treat as negative
                # Following paper protocol: focus on positive/negative points
                continue
            
            # Store point (note: y, x order for row, col indexing)
            # Coordinates are normalized [0, 1]
            if is_positive:
                image_annotations[image_id]['positive'][class_name].append((y, x))
            else:
                image_annotations[image_id]['negative'][class_name].append((y, x))
        
        print(f"Loaded annotations for {len(image_annotations)} images")
        
        # Print statistics
        total_positive = sum(len(pts) for img in image_annotations.values() 
                            for pts in img['positive'].values())
        total_negative = sum(len(pts) for img in image_annotations.values() 
                            for pts in img['negative'].values())
        print(f"Total positive points: {total_positive}")
        print(f"Total negative points: {total_negative}")
        
        # Count unique classes
        all_classes = set()
        for img in image_annotations.values():
            all_classes.update(img['positive'].keys())
            all_classes.update(img['negative'].keys())
        print(f"Unique classes: {len(all_classes)}")
        
        return dict(image_annotations), label_to_name
        
    except Exception as e:
        print(f"Error loading annotations: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OpenImagesV7 Point-wise Segmentation Benchmark'
    )
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory containing OpenImagesV7 validation images (optional if using Kaggle)')
    parser.add_argument('--annotations_csv', type=str, default=None,
                        help='Path to OpenImagesV7 validation point annotations CSV. If not provided, will try to find in Kaggle dataset.')
    parser.add_argument('--class_descriptions_csv', type=str, default=None,
                        help='Optional path to class-descriptions.csv for label name mapping')
    parser.add_argument('--use_kaggle', action='store_true',
                        help='Use Kaggle dataset structure (/kaggle/input/open-images). Auto-detected if path exists.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for evaluation')
    
    # Method selection
    parser.add_argument(
        '--methods', type=str, default='rollout',
        help="Comma-separated methods: rollout (others to be added)"
    )
    
    # Rollout settings
    parser.add_argument('--rollout_start_layer', type=int, default=0,
                        help='Start layer for attention rollout')
    
    # Threshold
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarization (default 0.5 as in paper)')
    
    args = parser.parse_args()
    
    # Parse methods
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    allowed_methods = {'rollout'}  # Add more methods later
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods. Use --methods with: rollout")
    
    print(f"Methods to evaluate: {methods}")
    print(f"Threshold: {args.threshold}")
    
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
    
    # Determine if using Kaggle dataset
    use_kaggle = args.use_kaggle or os.path.exists(KAGGLE_IMAGES_DIR)
    
    # Find annotations CSV
    if args.annotations_csv is None:
        if use_kaggle:
            # Try to find CSV in Kaggle dataset
            possible_csv_paths = [
                "/kaggle/input/open-images/oidv7-val-annotations-point-labels.csv",
                "/kaggle/input/oidv7-val-annotations-point-labels/oidv7-val-annotations-point-labels.csv",
                os.path.join(KAGGLE_ANNOTATIONS_DIR, "oidv7-val-annotations-point-labels.csv"),
            ]
            for path in possible_csv_paths:
                if os.path.exists(path):
                    args.annotations_csv = path
                    print(f"Auto-detected annotations CSV: {path}")
                    break
        
        if args.annotations_csv is None:
            raise ValueError("--annotations_csv is required. Please provide path to oidv7-val-annotations-point-labels.csv")
    
    # Find class descriptions CSV
    if args.class_descriptions_csv is None and use_kaggle:
        possible_desc_paths = [
            "/kaggle/input/open-images/oidv7-class-descriptions.csv",
            "/kaggle/input/oidv7-class-descriptions/oidv7-class-descriptions.csv",
            os.path.join(KAGGLE_DICT_DIR, "oidv7-class-descriptions.csv"),
        ]
        for path in possible_desc_paths:
            if os.path.exists(path):
                args.class_descriptions_csv = path
                print(f"Auto-detected class descriptions: {path}")
                break
    
    # Load annotations
    image_annotations, label_to_name = load_openimages_annotations(
        args.annotations_csv, 
        class_descriptions_csv=args.class_descriptions_csv
    )
    
    # Determine image directory
    if use_kaggle:
        images_dir = KAGGLE_IMAGES_DIR
        print(f"Using Kaggle dataset: {images_dir}")
    else:
        if args.data_root is None:
            raise ValueError("--data_root is required when not using Kaggle dataset")
        images_dir = args.data_root
        print(f"Using local images: {images_dir}")
    
    # Filter to images that exist
    valid_image_ids = []
    for image_id in image_annotations.keys():
        # Try common image paths
        possible_paths = [
            os.path.join(images_dir, f"{image_id}.jpg"),
            os.path.join(images_dir, f"{image_id}.png"),
            os.path.join(images_dir, "validation", f"{image_id}.jpg"),
            os.path.join(images_dir, "validation", f"{image_id}.png"),
            # Kaggle structure might have subdirectories
            os.path.join(images_dir, f"{image_id[:2]}", f"{image_id}.jpg"),
            os.path.join(images_dir, f"{image_id[:2]}", f"{image_id}.png"),
        ]
        if any(os.path.exists(p) for p in possible_paths):
            valid_image_ids.append(image_id)
    
    print(f"Found {len(valid_image_ids)} valid images out of {len(image_annotations)} with annotations")
    
    limit = args.limit if args.limit > 0 else len(valid_image_ids)
    limit = min(limit, len(valid_image_ids))
    print(f"Processing {limit} images...")
    
    # Results accumulator
    # For each method, store list of p-mIoU scores per image-class pair
    results = {m: [] for m in methods}
    
    # Process images
    for idx, image_id in enumerate(tqdm(valid_image_ids[:limit])):
        try:
            # Find image path
            image_path = None
            possible_paths = [
                os.path.join(images_dir, f"{image_id}.jpg"),
                os.path.join(images_dir, f"{image_id}.png"),
                os.path.join(images_dir, "validation", f"{image_id}.jpg"),
                os.path.join(images_dir, "validation", f"{image_id}.png"),
                # Kaggle structure might have subdirectories (first 2 chars)
                os.path.join(images_dir, f"{image_id[:2]}", f"{image_id}.jpg"),
                os.path.join(images_dir, f"{image_id[:2]}", f"{image_id}.png"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path is None:
                continue
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            img_t = preprocess(image).unsqueeze(0).to(args.device)
            H_img, W_img = img_t.shape[-2:]
            
            # Get annotations for this image
            ann = image_annotations[image_id]
            
            # Get all classes present in this image (with positive points)
            classes_in_image = list(ann['positive'].keys())
            
            if len(classes_in_image) == 0:
                continue
            
            # For each class, generate heatmap and evaluate
            for class_name in classes_in_image:
                positive_points = ann['positive'][class_name]
                negative_points = ann['negative'].get(class_name, [])
                
                if len(positive_points) == 0:
                    continue
                
                # Create text prompt for this class
                # OpenImagesV7 class names are typically in format like "Animal", "Vehicle", etc.
                # We'll use "a photo of a {class_name}" format
                prompt = f"a photo of a {class_name.lower()}"
                
                # Encode text
                text_tokens = tokenizer([prompt]).to(args.device)
                with torch.no_grad():
                    text_emb = model.encode_text(text_tokens, normalize=True)
                
                # Generate heatmap for each method
                # Note: For rollout, the heatmap is class-agnostic (same for all classes),
                # but we still evaluate it per class using point annotations
                for method in methods:
                    if method == 'rollout':
                        # Rollout doesn't use text embedding - it's class-agnostic
                        # The heatmap shows where the CLS token attends, regardless of class
                        heatmap = compute_attention_rollout_reference(
                            model, img_t, start_layer=args.rollout_start_layer
                        )
                    else:
                        raise NotImplementedError(f"Method {method} not implemented yet")
                    
                    # Resize heatmap to match image size if needed
                    if heatmap.shape != (H_img, W_img):
                        heatmap = F.interpolate(
                            heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]),
                            size=(H_img, W_img),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    
                    # Scale point coordinates to match image size
                    # OpenImagesV7 CSV uses normalized coordinates [0, 1]
                    # Scale to current image size (after preprocessing)
                    scaled_positive = [
                        (int(y * H_img), int(x * W_img))
                        for y, x in positive_points
                    ]
                    scaled_negative = [
                        (int(y * H_img), int(x * W_img))
                        for y, x in negative_points
                    ]
                    
                    # Compute p-mIoU
                    p_miou = compute_pointwise_miou(
                        heatmap, scaled_positive, scaled_negative, threshold=args.threshold
                    )
                    
                    results[method].append(p_miou)
        
        except Exception as e:
            print(f"Error processing image {image_id} (idx {idx}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final Results
    print("\n" + "=" * 60)
    print("RESULTS (OpenImagesV7 Point-wise Evaluation)")
    print("=" * 60)
    
    for method in methods:
        if len(results[method]) == 0:
            print(f"\n{method.upper()}: No valid results")
            continue
        
        p_miou_mean = np.mean(results[method])
        p_miou_std = np.std(results[method])
        
        print(f"\n{method.upper()}:")
        print(f"  p-mIoU: {p_miou_mean:.2f} ± {p_miou_std:.2f}")
        print(f"  Number of image-class pairs: {len(results[method])}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

