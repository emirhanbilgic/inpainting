#!/usr/bin/env python3
"""
Classification-Aware Difference Heatmap Analysis

This script investigates whether OMP-based text embedding refinement produces
more targeted heatmap improvements for correctly classified images vs.
incorrectly classified ones.

Pipeline:
  1. CLIP zero-shot classification of all ImageNet-Seg images (445 classes)
  2. Generate LeGrad heatmap (vanilla) using the PREDICTED class
  3. Generate LeGrad+OMP heatmap using the PREDICTED class + visual concept dictionary
  4. Compute difference heatmap: ReLU(OMP - vanilla)
  5. Compute mIoU of difference heatmap vs GT segmentation mask
  6. Plot histograms: correct vs incorrect classification

Usage:
    python scripts/classification_difference_analysis.py \
        --mat_file scripts/data/gtsegs_ijcv.mat \
        --device mps \
        --limit 0
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- PROJECT SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess
import open_clip

from sparse_encoding import omp_sparse_residual


# ==============================================================================
# Metric Functions (from benchmark_segmentation_v2.py)
# ==============================================================================

def batch_intersection_union(predict, target, nclass=2):
    """
    Batch Intersection of Union (reference implementation).
    
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


def compute_miou(heatmap_norm, gt_tensor, threshold):
    """Compute mIoU from a normalized heatmap and GT mask."""
    Res_1 = (heatmap_norm > threshold).float()
    Res_0 = (heatmap_norm <= threshold).float()
    output = torch.stack([Res_0, Res_1], dim=0)
    
    inter, union = batch_intersection_union(output, gt_tensor, nclass=2)
    iou = inter.astype(np.float64) / (union.astype(np.float64) + 1e-10)
    miou = iou.mean()
    return miou, iou


# ==============================================================================
# LeGrad Heatmap Functions
# ==============================================================================

def compute_legrad_heatmap(model, image, text_emb_1x):
    """Compute LeGrad heatmap for a single text embedding."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0].clamp(0, 1).detach().cpu()
    return logits


def get_text_embedding(model, tokenizer, prompt, device):
    """Encode a text prompt into a normalized embedding."""
    tok = tokenizer([prompt]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb


# ==============================================================================
# OMP Dictionary Building
# ==============================================================================

def build_omp_embedding(
    target_wnid,
    target_text_emb_1x,
    llm_dictionary,
    tokenizer,
    model,
    device,
    atoms=30,
    max_dict_cos_sim=0.65,
):
    """
    Build OMP sparse residual embedding using the visual concept dictionary.
    
    Args:
        target_wnid: WordNet ID of the target class
        target_text_emb_1x: [1, embed_dim] text embedding
        llm_dictionary: dict mapping wnid -> {visual_confusers, co_occurring_context, semantic_hierarchy}
        tokenizer: CLIP tokenizer
        model: LeWrapper model
        device: torch device
        atoms: max OMP atoms
        max_dict_cos_sim: filter threshold for dictionary entries
    
    Returns:
        sparse_1x: [1, embed_dim] OMP-refined embedding
    """
    parts = []
    
    # Get concepts from dictionary
    concepts_data = llm_dictionary.get(target_wnid)
    if concepts_data is not None:
        all_concepts = (
            concepts_data.get('visual_confusers', []) +
            concepts_data.get('co_occurring_context', []) +
            concepts_data.get('semantic_hierarchy', [])
        )
        if all_concepts:
            concept_prompts = [f"a photo of a {c}." for c in all_concepts]
            c_tok = tokenizer(concept_prompts).to(device)
            with torch.no_grad():
                c_emb = model.encode_text(c_tok)
                c_emb = F.normalize(c_emb, dim=-1)
            parts.append(c_emb)
    
    if len(parts) == 0:
        return F.normalize(target_text_emb_1x, dim=-1)
    
    D = torch.cat(parts, dim=0)
    D = F.normalize(D, dim=-1)
    
    # Filter by cosine similarity
    if 0.0 < max_dict_cos_sim < 1.0:
        sim = (D @ target_text_emb_1x.t()).squeeze(-1).abs()
        keep = sim < max_dict_cos_sim
        D = D[keep]
    
    if D.shape[0] == 0:
        return F.normalize(target_text_emb_1x, dim=-1)
    
    # OMP sparse residual
    sparse_1x = omp_sparse_residual(target_text_emb_1x, D, max_atoms=atoms)
    return sparse_1x


# ==============================================================================
# WordNet-based class name fallback
# ==============================================================================

def get_synset_name(wnid):
    """Fallback: derive a name from WordNet if JSON mapping is unavailable."""
    try:
        from nltk.corpus import wordnet as wn
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        name = synset.lemmas()[0].name().replace('_', ' ')
        return name
    except Exception:
        return wnid


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Classification-Aware Difference Heatmap Analysis'
    )
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--meta_file', type=str, default='scripts/all_classes_meta.json',
                        help='Path to all_classes_meta.json with 445 class wnid/name pairs')
    parser.add_argument('--dictionary_file', type=str, default='scripts/visual_concept_dictionary_445.json',
                        help='Path to visual concept dictionary for OMP')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='outputs/classification_difference')
    
    # OMP hyperparameters
    parser.add_argument('--sparse_threshold', type=float, default=0.35)
    parser.add_argument('--atoms', type=int, default=30)
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.65)
    
    # Top-k classification
    parser.add_argument('--top_k', type=int, default=5,
                        help='Consider top-k predictions for "correct" classification')
    
    # Visualization
    parser.add_argument('--vis_examples', type=int, default=10,
                        help='Number of example images to visualize per group (correct/incorrect)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 1. Load Model
    # ------------------------------------------------------------------
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
    
    # ------------------------------------------------------------------
    # 2. Load Class Metadata (445 classes)
    # ------------------------------------------------------------------
    print(f"Loading class metadata from {args.meta_file}...")
    with open(args.meta_file, 'r') as f:
        classes_meta = json.load(f)
    
    # Build wnid -> name mapping
    wnid_to_name = {}
    all_wnids = []
    for entry in classes_meta:
        wnid = entry['wnid']
        name = entry['name']
        wnid_to_name[wnid] = name
        all_wnids.append(wnid)
    
    print(f"  Loaded {len(all_wnids)} classes")
    
    # ------------------------------------------------------------------
    # 3. Precompute all 445 class text embeddings for zero-shot classification
    # ------------------------------------------------------------------
    print("Precomputing class embeddings for zero-shot classification...")
    all_prompts = [f"a photo of a {wnid_to_name[w]}." for w in all_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(all_wnids)}
    
    tok_all = tokenizer(all_prompts).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)  # [445, embed_dim]
    
    print(f"  Precomputed {len(all_prompts)} class embeddings")
    
    # ------------------------------------------------------------------
    # 4. Load Visual Concept Dictionary for OMP
    # ------------------------------------------------------------------
    print(f"Loading visual concept dictionary from {args.dictionary_file}...")
    with open(args.dictionary_file, 'r') as f:
        llm_dictionary = json.load(f)
    print(f"  Loaded dictionary with {len(llm_dictionary)} classes")
    
    # ------------------------------------------------------------------
    # 5. Open Dataset
    # ------------------------------------------------------------------
    print(f"Opening dataset {args.mat_file}...")
    mat_path = args.mat_file
    if not os.path.exists(mat_path):
        # Fallback for Kaggle
        kaggle_path = "/kaggle/input/mat-data/gtsegs_ijcv.mat"
        if os.path.exists(kaggle_path):
            print(f"  Dataset not found at {mat_path}, using fallback: {kaggle_path}")
            mat_path = kaggle_path
        else:
            print(f"  ERROR: Dataset not found at {mat_path}")
            return
    
    f = h5py.File(mat_path, 'r')
    imgs_refs = f['value/img']
    gts_refs = f['value/gt']
    targets_refs = f['value/target']
    num_images = imgs_refs.shape[0]
    
    limit = args.limit if args.limit > 0 else num_images
    limit = min(limit, num_images)
    print(f"  Processing {limit}/{num_images} images")
    
    # ------------------------------------------------------------------
    # 6. Main Loop
    # ------------------------------------------------------------------
    results = {
        'correct': [],  # list of per-image dicts
        'incorrect': [],
    }
    
    num_correct = 0
    num_incorrect = 0
    
    # Visualization storage: collect heatmaps for a few examples
    vis_correct = []   # list of dicts with image/heatmap data
    vis_incorrect = []
    
    # Accumulators for overall metrics
    correct_diff_inters = np.zeros(2)
    correct_diff_unions = np.zeros(2)
    incorrect_diff_inters = np.zeros(2)
    incorrect_diff_unions = np.zeros(2)
    
    # Also track vanilla and OMP mIoU separately
    correct_vanilla_inters = np.zeros(2)
    correct_vanilla_unions = np.zeros(2)
    incorrect_vanilla_inters = np.zeros(2)
    incorrect_vanilla_unions = np.zeros(2)
    correct_omp_inters = np.zeros(2)
    correct_omp_unions = np.zeros(2)
    incorrect_omp_inters = np.zeros(2)
    incorrect_omp_unions = np.zeros(2)
    
    for idx in tqdm(range(limit), desc="Processing"):
        try:
            # --- Load Image ---
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            H_feat, W_feat = img_t.shape[-2:]
            
            # --- Load GT Segmentation Mask ---
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
            
            # --- Get GT Class Info ---
            target_ref = targets_refs[idx, 0]
            target_data = np.array(f[target_ref])
            gt_wnid = ''.join([chr(c) for c in target_data.flatten()])
            gt_class_name = wnid_to_name.get(gt_wnid, get_synset_name(gt_wnid))
            
            # --- CLIP Zero-Shot Classification ---
            with torch.no_grad():
                image_features = model.encode_image(img_t, normalize=True)  # [1, embed_dim]
                similarities = (image_features @ all_text_embs.t()).squeeze(0)  # [445]
            
            top_k_indices = torch.topk(similarities, k=min(args.top_k, len(all_wnids))).indices.tolist()
            predicted_idx = top_k_indices[0]
            predicted_wnid = all_wnids[predicted_idx]
            predicted_name = wnid_to_name[predicted_wnid]
            predicted_conf = similarities[predicted_idx].item()
            
            # Check if GT is in top-k
            gt_idx_in_all = wnid_to_idx.get(gt_wnid)
            is_correct = gt_idx_in_all is not None and gt_idx_in_all in top_k_indices
            
            # --- Use PREDICTED class for LeGrad ---
            pred_text_emb = all_text_embs[predicted_idx:predicted_idx + 1]  # [1, embed_dim]
            
            # --- LeGrad Vanilla ---
            heatmap_vanilla = compute_legrad_heatmap(model, img_t, pred_text_emb)
            
            # Resize to GT size
            heatmap_vanilla_resized = F.interpolate(
                heatmap_vanilla.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # --- LeGrad + OMP ---
            omp_emb = build_omp_embedding(
                target_wnid=predicted_wnid,
                target_text_emb_1x=pred_text_emb,
                llm_dictionary=llm_dictionary,
                tokenizer=tokenizer,
                model=model,
                device=args.device,
                atoms=args.atoms,
                max_dict_cos_sim=args.max_dict_cos_sim,
            )
            
            heatmap_omp = compute_legrad_heatmap(model, img_t, omp_emb)
            
            heatmap_omp_resized = F.interpolate(
                heatmap_omp.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # --- Difference Heatmap: ReLU(OMP - vanilla) ---
            diff_heatmap = (heatmap_omp_resized - heatmap_vanilla_resized).clamp(min=0)
            
            # Normalize to [0, 1] if not already
            if diff_heatmap.max() > 0:
                diff_heatmap_norm = diff_heatmap / (diff_heatmap.max() + 1e-8)
            else:
                diff_heatmap_norm = diff_heatmap
            
            # --- Compute mIoU for difference heatmap ---
            diff_miou, diff_iou = compute_miou(diff_heatmap_norm, gt_tensor, args.sparse_threshold)
            
            # --- Also compute mIoU for vanilla and OMP individually ---
            vanilla_miou, vanilla_iou = compute_miou(heatmap_vanilla_resized, gt_tensor, args.sparse_threshold)
            omp_miou, omp_iou = compute_miou(heatmap_omp_resized, gt_tensor, args.sparse_threshold)
            
            # --- Compute IoU accumulators ---
            vanilla_output = torch.stack([
                (heatmap_vanilla_resized <= args.sparse_threshold).float(),
                (heatmap_vanilla_resized > args.sparse_threshold).float()
            ], dim=0)
            omp_output = torch.stack([
                (heatmap_omp_resized <= args.sparse_threshold).float(),
                (heatmap_omp_resized > args.sparse_threshold).float()
            ], dim=0)
            diff_output = torch.stack([
                (diff_heatmap_norm <= args.sparse_threshold).float(),
                (diff_heatmap_norm > args.sparse_threshold).float()
            ], dim=0)
            
            v_inter, v_union = batch_intersection_union(vanilla_output, gt_tensor, nclass=2)
            o_inter, o_union = batch_intersection_union(omp_output, gt_tensor, nclass=2)
            d_inter, d_union = batch_intersection_union(diff_output, gt_tensor, nclass=2)
            
            # --- Store result ---
            result = {
                'idx': idx,
                'gt_wnid': gt_wnid,
                'gt_class': gt_class_name,
                'predicted_wnid': predicted_wnid,
                'predicted_class': predicted_name,
                'predicted_confidence': float(predicted_conf),
                'is_correct': bool(is_correct),
                'vanilla_miou': float(vanilla_miou),
                'omp_miou': float(omp_miou),
                'diff_miou': float(diff_miou),
                'diff_heatmap_max': float(diff_heatmap.max().item()),
                'diff_heatmap_mean': float(diff_heatmap.mean().item()),
            }
            
            # --- Collect visualization data ---
            vis_data = {
                'idx': idx,
                'base_img': base_img.resize((W_gt, H_gt)),
                'gt_mask': gt_mask,
                'heatmap_vanilla': heatmap_vanilla_resized.numpy(),
                'heatmap_omp': heatmap_omp_resized.numpy(),
                'diff_heatmap': diff_heatmap.numpy(),
                'diff_heatmap_norm': diff_heatmap_norm.numpy(),
                'gt_class': gt_class_name,
                'predicted_class': predicted_name,
                'predicted_conf': predicted_conf,
                'vanilla_miou': float(vanilla_miou),
                'omp_miou': float(omp_miou),
                'diff_miou': float(diff_miou),
            }
            
            if is_correct:
                results['correct'].append(result)
                num_correct += 1
                correct_diff_inters += d_inter
                correct_diff_unions += d_union
                correct_vanilla_inters += v_inter
                correct_vanilla_unions += v_union
                correct_omp_inters += o_inter
                correct_omp_unions += o_union
                if len(vis_correct) < args.vis_examples:
                    vis_correct.append(vis_data)
            else:
                results['incorrect'].append(result)
                num_incorrect += 1
                incorrect_diff_inters += d_inter
                incorrect_diff_unions += d_union
                incorrect_vanilla_inters += v_inter
                incorrect_vanilla_unions += v_union
                incorrect_omp_inters += o_inter
                incorrect_omp_unions += o_union
                if len(vis_incorrect) < args.vis_examples:
                    vis_incorrect.append(vis_data)
            
        except Exception as e:
            print(f"\n  [ERROR] idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    f.close()
    
    # ------------------------------------------------------------------
    # 7. Compute Summary Statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    total = num_correct + num_incorrect
    print(f"\nClassification (top-{args.top_k}):")
    print(f"  Correct:   {num_correct}/{total} ({100*num_correct/max(total,1):.1f}%)")
    print(f"  Incorrect: {num_incorrect}/{total} ({100*num_incorrect/max(total,1):.1f}%)")
    
    # Global mIoU for each group
    def global_miou(inter, union):
        iou = inter.astype(np.float64) / (union.astype(np.float64) + 1e-10)
        return 100.0 * iou.mean()
    
    print(f"\n{'Metric':<25} {'Correct':>12} {'Incorrect':>12} {'Gap':>10}")
    print("-" * 60)
    
    corr_v = global_miou(correct_vanilla_inters, correct_vanilla_unions)
    incorr_v = global_miou(incorrect_vanilla_inters, incorrect_vanilla_unions)
    print(f"{'LeGrad vanilla mIoU':<25} {corr_v:>11.2f}% {incorr_v:>11.2f}% {corr_v-incorr_v:>9.2f}")
    
    corr_o = global_miou(correct_omp_inters, correct_omp_unions)
    incorr_o = global_miou(incorrect_omp_inters, incorrect_omp_unions)
    print(f"{'LeGrad+OMP mIoU':<25} {corr_o:>11.2f}% {incorr_o:>11.2f}% {corr_o-incorr_o:>9.2f}")
    
    corr_d = global_miou(correct_diff_inters, correct_diff_unions)
    incorr_d = global_miou(incorrect_diff_inters, incorrect_diff_unions)
    print(f"{'Difference mIoU':<25} {corr_d:>11.2f}% {incorr_d:>11.2f}% {corr_d-incorr_d:>9.2f}")
    
    # Per-image mean mIoU
    correct_diff_mious = [r['diff_miou'] for r in results['correct']]
    incorrect_diff_mious = [r['diff_miou'] for r in results['incorrect']]
    
    if correct_diff_mious:
        print(f"\n  Correct   diff mIoU (per-image mean): {100*np.mean(correct_diff_mious):.2f}% ± {100*np.std(correct_diff_mious):.2f}%")
    if incorrect_diff_mious:
        print(f"  Incorrect diff mIoU (per-image mean): {100*np.mean(incorrect_diff_mious):.2f}% ± {100*np.std(incorrect_diff_mious):.2f}%")
    
    # ------------------------------------------------------------------
    # 8. Plot Histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram 1: Difference mIoU
    ax = axes[0]
    if correct_diff_mious:
        ax.hist(correct_diff_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_diff_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_diff_mious:
        ax.hist(incorrect_diff_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_diff_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU (Difference Heatmap vs GT)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('ReLU(LeGrad_OMP − LeGrad) vs GT Mask', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Histogram 2: Vanilla LeGrad mIoU
    correct_vanilla_mious = [r['vanilla_miou'] for r in results['correct']]
    incorrect_vanilla_mious = [r['vanilla_miou'] for r in results['incorrect']]
    
    ax = axes[1]
    if correct_vanilla_mious:
        ax.hist(correct_vanilla_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_vanilla_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_vanilla_mious:
        ax.hist(incorrect_vanilla_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_vanilla_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU (Vanilla LeGrad vs GT)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('LeGrad Vanilla vs GT Mask', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Histogram 3: OMP LeGrad mIoU
    correct_omp_mious = [r['omp_miou'] for r in results['correct']]
    incorrect_omp_mious = [r['omp_miou'] for r in results['incorrect']]
    
    ax = axes[2]
    if correct_omp_mious:
        ax.hist(correct_omp_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_omp_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_omp_mious:
        ax.hist(incorrect_omp_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_omp_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU (LeGrad+OMP vs GT)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('LeGrad+OMP vs GT Mask', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.suptitle(
        f'Classification-Aware Segmentation Analysis (top-{args.top_k}, threshold={args.sparse_threshold})',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    hist_path = os.path.join(args.output_dir, 'histograms.pdf')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to {hist_path}")
    
    # ------------------------------------------------------------------
    # 9. Additional Plot: Improvement scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    correct_vanilla_arr = np.array([r['vanilla_miou'] for r in results['correct']])
    correct_omp_arr = np.array([r['omp_miou'] for r in results['correct']])
    incorrect_vanilla_arr = np.array([r['vanilla_miou'] for r in results['incorrect']])
    incorrect_omp_arr = np.array([r['omp_miou'] for r in results['incorrect']])
    
    if len(correct_vanilla_arr) > 0:
        ax.scatter(correct_vanilla_arr, correct_omp_arr, alpha=0.4, c='#2196F3', s=15, label=f'Correct (n={len(correct_vanilla_arr)})')
    if len(incorrect_vanilla_arr) > 0:
        ax.scatter(incorrect_vanilla_arr, incorrect_omp_arr, alpha=0.4, c='#F44336', s=15, label=f'Incorrect (n={len(incorrect_vanilla_arr)})')
    
    # Diagonal line (no improvement)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
    ax.set_xlabel('LeGrad Vanilla mIoU', fontsize=12)
    ax.set_ylabel('LeGrad+OMP mIoU', fontsize=12)
    ax.set_title('OMP Improvement: Vanilla vs OMP mIoU', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    scatter_path = os.path.join(args.output_dir, 'scatter_improvement.pdf')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")
    
    # ------------------------------------------------------------------
    # 9b. Per-Image Visualization Panels
    # ------------------------------------------------------------------
    def save_vis_panel(vis_list, group_name, output_dir, threshold):
        """Save a multi-row visualization panel for a group of images."""
        if not vis_list:
            print(f"  No {group_name} examples to visualize.")
            return
        
        n = len(vis_list)
        ncols = 6  # Original, GT, Vanilla, OMP, Diff Raw, Diff Binary
        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3, n * 3))
        if n == 1:
            axes = axes[np.newaxis, :]  # ensure 2D
        
        # Column headers
        col_titles = ['Original', 'GT Mask', 'LeGrad Vanilla', 'LeGrad+OMP', 
                       'Difference', 'Diff Binary']
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=11, fontweight='bold', pad=8)
        
        for i, vd in enumerate(vis_list):
            # Col 0: Original image
            axes[i, 0].imshow(vd['base_img'])
            label = f"GT: {vd['gt_class']}"
            if vd['predicted_class'] != vd['gt_class']:
                label += f"\nPred: {vd['predicted_class']}"
            axes[i, 0].set_ylabel(f"#{vd['idx']}", fontsize=9, rotation=0, labelpad=30, va='center')
            axes[i, 0].text(0.02, 0.98, label, transform=axes[i, 0].transAxes,
                           fontsize=7, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 1: GT mask
            axes[i, 1].imshow(vd['gt_mask'], cmap='gray', vmin=0, vmax=1)
            
            # Col 2: Vanilla heatmap
            axes[i, 2].imshow(vd['heatmap_vanilla'], cmap='jet', vmin=0, vmax=1)
            axes[i, 2].text(0.02, 0.02, f"mIoU: {100*vd['vanilla_miou']:.1f}%",
                           transform=axes[i, 2].transAxes, fontsize=8, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 3: OMP heatmap
            axes[i, 3].imshow(vd['heatmap_omp'], cmap='jet', vmin=0, vmax=1)
            axes[i, 3].text(0.02, 0.02, f"mIoU: {100*vd['omp_miou']:.1f}%",
                           transform=axes[i, 3].transAxes, fontsize=8, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 4: Difference heatmap (raw, normalized)
            axes[i, 4].imshow(vd['diff_heatmap_norm'], cmap='hot', vmin=0, vmax=1)
            axes[i, 4].text(0.02, 0.02, f"mIoU: {100*vd['diff_miou']:.1f}%",
                           transform=axes[i, 4].transAxes, fontsize=8, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 5: Difference binary (thresholded)
            diff_binary = (vd['diff_heatmap_norm'] > threshold).astype(np.uint8)
            axes[i, 5].imshow(diff_binary, cmap='gray', vmin=0, vmax=1)
            
            # Remove ticks
            for j in range(ncols):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        color = '#2196F3' if group_name == 'Correctly Classified' else '#F44336'
        fig.suptitle(
            f'{group_name} Examples (n={n})',
            fontsize=14, fontweight='bold', color=color, y=1.01
        )
        plt.tight_layout()
        
        fname = f"vis_{group_name.lower().replace(' ', '_')}.pdf"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {group_name} visualization: {out_path}")
    
    print("\nGenerating per-image visualizations...")
    save_vis_panel(vis_correct, 'Correctly Classified', args.output_dir, args.sparse_threshold)
    save_vis_panel(vis_incorrect, 'Incorrectly Classified', args.output_dir, args.sparse_threshold)
    
    # ------------------------------------------------------------------
    # 10. Save Detailed Results
    # ------------------------------------------------------------------
    results_summary = {
        'config': {
            'model': args.model_name,
            'pretrained': args.pretrained,
            'image_size': args.image_size,
            'sparse_threshold': args.sparse_threshold,
            'atoms': args.atoms,
            'max_dict_cos_sim': args.max_dict_cos_sim,
            'top_k': args.top_k,
            'num_images': limit,
        },
        'summary': {
            'num_correct': num_correct,
            'num_incorrect': num_incorrect,
            'classification_accuracy': float(num_correct / max(total, 1)),
            'correct_vanilla_miou': float(corr_v),
            'incorrect_vanilla_miou': float(incorr_v),
            'correct_omp_miou': float(corr_o),
            'incorrect_omp_miou': float(incorr_o),
            'correct_diff_miou': float(corr_d),
            'incorrect_diff_miou': float(incorr_d),
        },
        'per_image': results,
    }
    
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as jf:
        json.dump(results_summary, jf, indent=2)
    print(f"Detailed results saved to {json_path}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
