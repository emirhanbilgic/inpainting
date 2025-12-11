#!/usr/bin/env python3
"""
Evaluate Sparse GradCAM segmentation with specific hyperparameters.

This script runs GradCAM with sparse text embeddings (via OMP) on the 
ImageNet-Segmentation benchmark and reports PixelAcc, mIoU, and mAP.

Usage:
    # Default sparse settings
    python scripts/evaluate_sparse_gradcam.py
    
    # Custom settings from optimization
    python scripts/evaluate_sparse_gradcam.py \\
        --wn_use_synonyms 0 \\
        --wn_use_hypernyms 1 \\
        --wn_use_hyponyms 0 \\
        --wn_use_siblings 1 \\
        --dict_include_prompts 1 \\
        --sparse_threshold 0.45 \\
        --atoms 16 \\
        --max_dict_cos_sim 0.85
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

# Import shared functions
from benchmark_segmentation import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    compute_iou_acc,
    compute_map_score,
    omp_sparse_residual,
    wordnet_neighbors_configured,
)


def compute_gradcam_for_embedding(model, image, text_emb_1x, layer_index: int = 8):
    """
    Compute a GradCAM heatmap (normalized to [0, 1]) for a single text embedding.
    Returns: 2D tensor [H, W] on CPU.
    """
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sparse GradCAM Segmentation')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # GradCAM layer
    parser.add_argument('--gradcam_layer', type=int, default=8, help='GradCAM layer index')
    
    # Sparse settings
    parser.add_argument('--atoms', type=int, default=8, help='Number of atoms for OMP sparse residual.')
    parser.add_argument('--dict_include_prompts', type=int, default=1, help='Include other class prompts in dictionary (0/1).')
    parser.add_argument('--wn_use_synonyms', type=int, default=0, help='WordNet: include synonyms (0/1).')
    parser.add_argument('--wn_use_hypernyms', type=int, default=0, help='WordNet: include hypernyms (0/1).')
    parser.add_argument('--wn_use_hyponyms', type=int, default=0, help='WordNet: include hyponyms (0/1).')
    parser.add_argument('--wn_use_siblings', type=int, default=1, help='WordNet: include siblings (0/1).')
    parser.add_argument('--wn_fallback_search', type=int, default=1, help='Enable fallback for multi-word names (0/1).')
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.9, help='Max cosine similarity filter for dictionary.')
    parser.add_argument('--sparse_threshold', type=float, default=0.5, help='Binarization threshold for sparse heatmaps.')
    
    # Comparison with original GradCAM
    parser.add_argument('--compare_original', action='store_true', help='Also evaluate original GradCAM for comparison.')
    
    # Visualization
    parser.add_argument('--vis_first_k', type=int, default=0, help='If >0, save visualization grids for first K images.')
    parser.add_argument('--vis_output_dir', type=str, default='outputs/sparse_gradcam_vis', help='Directory to save visualizations.')
    
    args = parser.parse_args()
    
    # Load Model
    print(f"Loading model {args.model_name}...")
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
    
    # Create visualization directory if requested
    if args.vis_first_k > 0:
        os.makedirs(args.vis_output_dir, exist_ok=True)
    
    # Precompute wnids and embeddings
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = ''.join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)
    unique_wnids = sorted(set(wnids_in_seg))
    
    wnid_to_prompt = {}
    wnid_to_classname = {}
    for wnid in unique_wnids:
        class_label = wnid_to_label.get(wnid)
        if class_label is None:
            class_label = get_synset_name(wnid)
        wnid_to_classname[wnid] = class_label
        wnid_to_prompt[wnid] = f"a photo of a {class_label}."
    
    all_prompts = [wnid_to_prompt[w] for w in unique_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}
    
    print(f"[prompts] Built {len(all_prompts)} unique class prompts.")
    tok_all = tokenizer(all_prompts).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)
    
    # Results
    results = {
        'sparse_gradcam': {'iou': [], 'acc': [], 'ap': []},
    }
    if args.compare_original:
        results['original_gradcam'] = {'iou': [], 'acc': [], 'ap': []}
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Sparse GradCAM Configuration:")
    print(f"{'='*60}")
    print(f"  gradcam_layer: {args.gradcam_layer}")
    print(f"  atoms: {args.atoms}")
    print(f"  dict_include_prompts: {bool(args.dict_include_prompts)}")
    print(f"  wn_use_synonyms: {bool(args.wn_use_synonyms)}")
    print(f"  wn_use_hypernyms: {bool(args.wn_use_hypernyms)}")
    print(f"  wn_use_hyponyms: {bool(args.wn_use_hyponyms)}")
    print(f"  wn_use_siblings: {bool(args.wn_use_siblings)}")
    print(f"  max_dict_cos_sim: {args.max_dict_cos_sim}")
    print(f"  sparse_threshold: {args.sparse_threshold}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
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
            
            # Resize GT
            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            target_resize = transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            gt_pil = target_resize(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.uint8)
            H_gt, W_gt = gt_mask.shape
            
            # Get class info
            wnid = wnids_in_seg[idx]
            class_name = wnid_to_classname[wnid]
            prompt = wnid_to_prompt[wnid]
            cls_idx = wnid_to_idx[wnid]
            original_1x = all_text_embs[cls_idx:cls_idx + 1]
            
            # --- Original GradCAM (if comparing) ---
            if args.compare_original:
                heatmap_orig = compute_gradcam_for_embedding(
                    model, img_t, original_1x, layer_index=args.gradcam_layer
                )
                heatmap_orig_resized = F.interpolate(
                    heatmap_orig.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                iou_o, acc_o = compute_iou_acc(heatmap_orig_resized, gt_mask, threshold=0.5)
                ap_o = compute_map_score(heatmap_orig_resized, gt_mask)
                
                results['original_gradcam']['iou'].append(iou_o)
                results['original_gradcam']['acc'].append(acc_o)
                results['original_gradcam']['ap'].append(ap_o)
            
            # --- Sparse GradCAM ---
            # Build dictionary
            parts = []
            
            # 1) Other class prompts
            if bool(args.dict_include_prompts) and len(unique_wnids) > 1:
                if cls_idx > 0:
                    parts.append(all_text_embs[:cls_idx])
                if cls_idx + 1 < len(unique_wnids):
                    parts.append(all_text_embs[cls_idx + 1:])
            
            # 2) WordNet neighbors
            use_wn = any([
                bool(args.wn_use_synonyms),
                bool(args.wn_use_hypernyms),
                bool(args.wn_use_hyponyms),
                bool(args.wn_use_siblings),
            ])
            if use_wn:
                raw_neighbors = wordnet_neighbors_configured(
                    class_name,
                    use_synonyms=bool(args.wn_use_synonyms),
                    use_hypernyms=bool(args.wn_use_hypernyms),
                    use_hyponyms=bool(args.wn_use_hyponyms),
                    use_siblings=bool(args.wn_use_siblings),
                    use_fallback=bool(args.wn_fallback_search),
                    limit_per_relation=8,
                )
                if raw_neighbors:
                    neighbor_prompts = [prompt.replace(class_name, w) for w in raw_neighbors]
                    n_tok = tokenizer(neighbor_prompts).to(args.device)
                    with torch.no_grad():
                        n_emb = model.encode_text(n_tok)
                        n_emb = F.normalize(n_emb, dim=-1)
                    parts.append(n_emb)
            
            # Combine dictionary
            if len(parts) > 0:
                D = torch.cat(parts, dim=0)
                D = F.normalize(D, dim=-1)
                
                # Filter by cosine similarity
                if 0.0 < args.max_dict_cos_sim < 1.0:
                    sim = (D @ original_1x.t()).squeeze(-1).abs()
                    keep = sim < args.max_dict_cos_sim
                    D = D[keep]
            else:
                D = original_1x.new_zeros((0, original_1x.shape[-1]))
            
            # OMP sparse residual
            sparse_1x = omp_sparse_residual(original_1x, D, max_atoms=args.atoms)
            
            # Compute sparse GradCAM
            heatmap_sparse = compute_gradcam_for_embedding(
                model, img_t, sparse_1x, layer_index=args.gradcam_layer
            )
            heatmap_sparse_resized = F.interpolate(
                heatmap_sparse.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Compute metrics
            iou_s, acc_s = compute_iou_acc(heatmap_sparse_resized, gt_mask, threshold=args.sparse_threshold)
            ap_s = compute_map_score(heatmap_sparse_resized, gt_mask)
            
            results['sparse_gradcam']['iou'].append(iou_s)
            results['sparse_gradcam']['acc'].append(acc_s)
            results['sparse_gradcam']['ap'].append(ap_s)
            
            # --- Visualization ---
            if idx < args.vis_first_k:
                vis_img = base_img.resize((W_gt, H_gt))
                gt_bin = gt_mask.astype(np.uint8)
                sparse_bin = (heatmap_sparse_resized > args.sparse_threshold).astype(np.uint8)
                
                if args.compare_original:
                    orig_bin = (heatmap_orig_resized > 0.5).astype(np.uint8)
                    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                    axes[0].imshow(vis_img)
                    axes[0].set_title('Image')
                    axes[0].axis('off')
                    axes[1].imshow(gt_bin, cmap='gray')
                    axes[1].set_title('GT')
                    axes[1].axis('off')
                    axes[2].imshow(orig_bin, cmap='gray')
                    axes[2].set_title('GradCAM (0.5)')
                    axes[2].axis('off')
                    axes[3].imshow(sparse_bin, cmap='gray')
                    axes[3].set_title(f'Sparse ({args.sparse_threshold:.2f})')
                    axes[3].axis('off')
                    axes[4].imshow(heatmap_sparse_resized, cmap='jet')
                    axes[4].set_title('Sparse Heatmap')
                    axes[4].axis('off')
                else:
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    axes[0].imshow(vis_img)
                    axes[0].set_title('Image')
                    axes[0].axis('off')
                    axes[1].imshow(gt_bin, cmap='gray')
                    axes[1].set_title('GT')
                    axes[1].axis('off')
                    axes[2].imshow(sparse_bin, cmap='gray')
                    axes[2].set_title(f'Sparse ({args.sparse_threshold:.2f})')
                    axes[2].axis('off')
                    axes[3].imshow(heatmap_sparse_resized, cmap='jet')
                    axes[3].set_title('Sparse Heatmap')
                    axes[3].axis('off')
                
                plt.tight_layout()
                out_name = f"sparse_gradcam_{idx:04d}.png"
                out_path = os.path.join(args.vis_output_dir, out_name)
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[vis] Saved {out_path}")
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    miou_s = np.mean(results['sparse_gradcam']['iou']) * 100
    macc_s = np.mean(results['sparse_gradcam']['acc']) * 100
    map_s = np.mean(results['sparse_gradcam']['ap']) * 100
    
    print(f"\nSparse GradCAM: PixelAcc={macc_s:.2f}, mIoU={miou_s:.2f}, mAP={map_s:.2f}")
    
    if args.compare_original:
        miou_o = np.mean(results['original_gradcam']['iou']) * 100
        macc_o = np.mean(results['original_gradcam']['acc']) * 100
        map_o = np.mean(results['original_gradcam']['ap']) * 100
        
        print(f"Original GradCAM: PixelAcc={macc_o:.2f}, mIoU={miou_o:.2f}, mAP={map_o:.2f}")
        
        print(f"\nImprovement:")
        print(f"  PixelAcc: {macc_s - macc_o:+.2f}")
        print(f"  mIoU:     {miou_s - miou_o:+.2f}")
        print(f"  mAP:      {map_s - map_o:+.2f}")
    
    print(f"\nProcessed {limit} images in {elapsed/60.0:.2f} minutes.")


if __name__ == '__main__':
    main()
