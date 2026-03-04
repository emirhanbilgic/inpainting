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


from sklearn.metrics import average_precision_score, roc_auc_score

# ==============================================================================
# Metric Functions (from benchmark_segmentation_v2.py)
# ==============================================================================

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
    pixel_correct, pixel_labeled = batch_pix_accuracy(output, gt_tensor)
    
    output_AP = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)
    ap = get_ap_scores(output_AP, gt_tensor)
    ap_val = ap[0] if ap else 0.0
    
    iou = inter.astype(np.float64) / (union.astype(np.float64) + 1e-10)
    miou = iou.mean()
    return miou, iou, pixel_correct, pixel_labeled, ap_val


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

def build_omp_embedding_llm(
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
    Build OMP sparse residual embedding using the LLM visual concept dictionary.
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


def build_omp_embedding_from_predictions(
    predicted_idx,
    target_text_emb_1x,
    similarities,
    all_text_embs,
    dict_top_k=40,
    atoms=30,
    max_dict_cos_sim=0.65,
):
    """
    Build OMP sparse residual using per-image CLIP top-K predictions as dictionary.
    
    For each image, CLIP predicts similarity scores for all 445 classes.
    We take the top-K runner-up predictions (excluding #1) and use their
    text embeddings as OMP dictionary atoms.
    
    This is maximally targeted: the dictionary is exactly what CLIP thinks
    THIS specific image could be (its confusers for this particular image).
    
    For correctly classified images: removes the specific competing signals
    that CLIP sees in this image, sharpening the correct prediction.
    For incorrectly classified images: the runner-ups may include the GT class,
    and removing them may degrade the heatmap further.
    """
    # Get top-(K+1) predictions, skip the #1 (which is the predicted class)
    top_k_total = min(dict_top_k + 1, len(similarities))
    _, top_indices = torch.topk(similarities, k=top_k_total)
    
    # Exclude the predicted class itself (index 0 in sorted top-k)
    confuser_indices = [i.item() for i in top_indices if i.item() != predicted_idx]
    confuser_indices = confuser_indices[:dict_top_k]  # limit to dict_top_k
    
    if len(confuser_indices) == 0:
        return F.normalize(target_text_emb_1x, dim=-1)
    
    D = all_text_embs[confuser_indices]  # [K, embed_dim]
    D = F.normalize(D, dim=-1)
    
    # Filter by cosine similarity
    if 0.0 < max_dict_cos_sim < 1.0:
        sim = (D @ target_text_emb_1x.t()).squeeze(-1).abs()
        keep = sim < max_dict_cos_sim
        D = D[keep]
    
    if D.shape[0] == 0:
        return F.normalize(target_text_emb_1x, dim=-1)
    
    # OMP sparse residual
    sparse_1x = omp_sparse_residual(target_text_emb_1x, D, max_atoms=min(atoms, D.shape[0]))
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
    
    # Dictionary mode
    parser.add_argument('--dict_mode', type=str, default='clip_confusion',
                        choices=['llm', 'clip_confusion'],
                        help='Dictionary mode: "clip_confusion" (class-conditional negative mining) or "llm" (visual concept dictionary)')
    parser.add_argument('--dict_top_k', type=int, default=40,
                        help='Number of top confusers per class for clip_confusion mode')
    
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
    # 4. Build OMP Dictionary
    # ------------------------------------------------------------------
    llm_dictionary = None
    
    if args.dict_mode == 'llm':
        print(f"Loading LLM visual concept dictionary from {args.dictionary_file}...")
        with open(args.dictionary_file, 'r') as jf:
            llm_dictionary = json.load(jf)
        print(f"  Loaded dictionary with {len(llm_dictionary)} classes")
    elif args.dict_mode == 'clip_confusion':
        print(f"Using per-image CLIP prediction dictionary (top-{args.dict_top_k} runner-ups per image)")
    
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
    
    # Signed difference (no ReLU) accumulators
    correct_signed_inters = np.zeros(2)
    correct_signed_unions = np.zeros(2)
    incorrect_signed_inters = np.zeros(2)
    incorrect_signed_unions = np.zeros(2)
    
    # Pix Acc, AP accumulators
    correct_vanilla_pixel_correct = 0; correct_vanilla_pixel_labeled = 0
    incorrect_vanilla_pixel_correct = 0; incorrect_vanilla_pixel_labeled = 0
    
    correct_omp_pixel_correct = 0; correct_omp_pixel_labeled = 0
    incorrect_omp_pixel_correct = 0; incorrect_omp_pixel_labeled = 0
    
    correct_diff_pixel_correct = 0; correct_diff_pixel_labeled = 0
    incorrect_diff_pixel_correct = 0; incorrect_diff_pixel_labeled = 0
    
    correct_signed_pixel_correct = 0; correct_signed_pixel_labeled = 0
    incorrect_signed_pixel_correct = 0; incorrect_signed_pixel_labeled = 0
    
    correct_vanilla_aps = []
    incorrect_vanilla_aps = []
    
    correct_omp_aps = []
    incorrect_omp_aps = []
    
    correct_diff_aps = []
    incorrect_diff_aps = []
    
    correct_signed_aps = []
    incorrect_signed_aps = []
    
    # AUROC
    correct_vanilla_aurocs = []
    incorrect_vanilla_aurocs = []
    
    correct_omp_aurocs = []
    incorrect_omp_aurocs = []
    
    correct_diff_aurocs = []
    incorrect_diff_aurocs = []
    
    correct_signed_aurocs = []
    incorrect_signed_aurocs = []
    
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
            if args.dict_mode == 'clip_confusion':
                omp_emb = build_omp_embedding_from_predictions(
                    predicted_idx=predicted_idx,
                    target_text_emb_1x=pred_text_emb,
                    similarities=similarities,
                    all_text_embs=all_text_embs,
                    dict_top_k=args.dict_top_k,
                    atoms=args.atoms,
                    max_dict_cos_sim=args.max_dict_cos_sim,
                )
            else:
                omp_emb = build_omp_embedding_llm(
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
            diff_raw = heatmap_omp_resized - heatmap_vanilla_resized  # signed
            diff_relu = diff_raw.clamp(min=0)  # ReLU
            
            # Normalize ReLU difference to [0, 1]
            if diff_relu.max() > 0:
                diff_relu_norm = diff_relu / (diff_relu.max() + 1e-8)
            else:
                diff_relu_norm = diff_relu
            
            # Normalize signed difference to [0, 1] (shift from [-1,1] to [0,1])
            # Map: -1 -> 0, 0 -> 0.5, +1 -> 1
            diff_signed_norm = (diff_raw + 1.0) / 2.0
            diff_signed_norm = diff_signed_norm.clamp(0, 1)
            
            # --- Compute mIoU, AP, Acc for ReLU difference ---
            diff_miou, diff_iou, diff_pix_c, diff_pix_l, diff_ap = compute_miou(diff_relu_norm, gt_tensor, args.sparse_threshold)
            
            # --- Compute mIoU, AP, Acc for signed difference ---
            # For signed: threshold at 0.5 + threshold/2 means "positive change above threshold"
            signed_thr = 0.5 + args.sparse_threshold / 2.0
            signed_miou, signed_iou, signed_pix_c, signed_pix_l, signed_ap = compute_miou(diff_signed_norm, gt_tensor, signed_thr)
            
            # --- Also compute mIoU, AP, Acc for vanilla and OMP individually ---
            vanilla_miou, vanilla_iou, v_pix_c, v_pix_l, v_ap = compute_miou(heatmap_vanilla_resized, gt_tensor, args.sparse_threshold)
            omp_miou, omp_iou, o_pix_c, o_pix_l, o_ap = compute_miou(heatmap_omp_resized, gt_tensor, args.sparse_threshold)
            
            # --- Compute AUROC ---
            gt_binary = (gt_mask > 0).astype(int).flatten()
            if len(np.unique(gt_binary)) > 1:
                v_auroc = roc_auc_score(gt_binary, heatmap_vanilla_resized.numpy().flatten())
                o_auroc = roc_auc_score(gt_binary, heatmap_omp_resized.numpy().flatten())
                diff_auroc = roc_auc_score(gt_binary, diff_relu_norm.numpy().flatten())
                signed_auroc = roc_auc_score(gt_binary, diff_signed_norm.numpy().flatten())
            else:
                v_auroc = np.nan; o_auroc = np.nan; diff_auroc = np.nan; signed_auroc = np.nan
            
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
                (diff_relu_norm <= args.sparse_threshold).float(),
                (diff_relu_norm > args.sparse_threshold).float()
            ], dim=0)
            signed_output = torch.stack([
                (diff_signed_norm <= signed_thr).float(),
                (diff_signed_norm > signed_thr).float()
            ], dim=0)
            
            v_inter, v_union = batch_intersection_union(vanilla_output, gt_tensor, nclass=2)
            o_inter, o_union = batch_intersection_union(omp_output, gt_tensor, nclass=2)
            d_inter, d_union = batch_intersection_union(diff_output, gt_tensor, nclass=2)
            s_inter, s_union = batch_intersection_union(signed_output, gt_tensor, nclass=2)
            
            # --- Focus Change Drop: avg(vanilla, OMP) mIoU - signed diff mIoU ---
            avg_miou = (float(vanilla_miou) + float(omp_miou)) / 2.0
            focus_drop = avg_miou - float(signed_miou)
            omp_drop = float(omp_miou) - float(signed_miou)
            
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
                'signed_diff_miou': float(signed_miou),
                'focus_drop': float(focus_drop),
                'omp_drop': float(omp_drop),
                'avg_miou': float(avg_miou),
                'diff_heatmap_max': float(diff_relu.max().item()),
                'diff_heatmap_mean': float(diff_relu.mean().item()),
                'signed_diff_max': float(diff_raw.max().item()),
                'signed_diff_min': float(diff_raw.min().item()),
                'signed_diff_mean': float(diff_raw.mean().item()),
            }
            
            # --- Average heatmap for visualization ---
            avg_heatmap = (heatmap_vanilla_resized.numpy() + heatmap_omp_resized.numpy()) / 2.0
            
            # --- Collect visualization data ---
            vis_data = {
                'idx': idx,
                'base_img': base_img.resize((W_gt, H_gt)),
                'gt_mask': gt_mask,
                'heatmap_vanilla': heatmap_vanilla_resized.numpy(),
                'heatmap_omp': heatmap_omp_resized.numpy(),
                'heatmap_avg': avg_heatmap,
                'diff_relu': diff_relu.numpy(),
                'diff_relu_norm': diff_relu_norm.numpy(),
                'diff_signed': diff_raw.numpy(),
                'diff_signed_norm': diff_signed_norm.numpy(),
                'gt_class': gt_class_name,
                'predicted_class': predicted_name,
                'predicted_conf': predicted_conf,
                'vanilla_miou': float(vanilla_miou),
                'omp_miou': float(omp_miou),
                'avg_miou': float(avg_miou),
                'diff_miou': float(diff_miou),
                'signed_diff_miou': float(signed_miou),
                'focus_drop': float(focus_drop),
                'omp_drop': float(omp_drop),
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
                correct_signed_inters += s_inter
                correct_signed_unions += s_union
                
                correct_vanilla_pixel_correct += v_pix_c
                correct_vanilla_pixel_labeled += v_pix_l
                correct_omp_pixel_correct += o_pix_c
                correct_omp_pixel_labeled += o_pix_l
                correct_diff_pixel_correct += diff_pix_c
                correct_diff_pixel_labeled += diff_pix_l
                correct_signed_pixel_correct += signed_pix_c
                correct_signed_pixel_labeled += signed_pix_l
                
                correct_vanilla_aps.append(v_ap)
                correct_omp_aps.append(o_ap)
                correct_diff_aps.append(diff_ap)
                correct_signed_aps.append(signed_ap)
                
                if not np.isnan(v_auroc): correct_vanilla_aurocs.append(v_auroc)
                if not np.isnan(o_auroc): correct_omp_aurocs.append(o_auroc)
                if not np.isnan(diff_auroc): correct_diff_aurocs.append(diff_auroc)
                if not np.isnan(signed_auroc): correct_signed_aurocs.append(signed_auroc)
                
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
                incorrect_signed_inters += s_inter
                incorrect_signed_unions += s_union
                
                incorrect_vanilla_pixel_correct += v_pix_c
                incorrect_vanilla_pixel_labeled += v_pix_l
                incorrect_omp_pixel_correct += o_pix_c
                incorrect_omp_pixel_labeled += o_pix_l
                incorrect_diff_pixel_correct += diff_pix_c
                incorrect_diff_pixel_labeled += diff_pix_l
                incorrect_signed_pixel_correct += signed_pix_c
                incorrect_signed_pixel_labeled += signed_pix_l
                
                incorrect_vanilla_aps.append(v_ap)
                incorrect_omp_aps.append(o_ap)
                incorrect_diff_aps.append(diff_ap)
                incorrect_signed_aps.append(signed_ap)
                
                if not np.isnan(v_auroc): incorrect_vanilla_aurocs.append(v_auroc)
                if not np.isnan(o_auroc): incorrect_omp_aurocs.append(o_auroc)
                if not np.isnan(diff_auroc): incorrect_diff_aurocs.append(diff_auroc)
                if not np.isnan(signed_auroc): incorrect_signed_aurocs.append(signed_auroc)
                
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
    
    def global_acc(correct, labeled):
        return 100.0 * (correct / (labeled + 1e-10))
    
    def mean_met(arr):
        return 100.0 * np.mean(arr) if len(arr) > 0 else 0.0
    
    print(f"\n{'Metric':<25} {'Correct':>12} {'Incorrect':>12} {'Gap':>10}")
    print("-" * 60)
    
    # Vanilla
    corr_v = global_miou(correct_vanilla_inters, correct_vanilla_unions)
    incorr_v = global_miou(incorrect_vanilla_inters, incorrect_vanilla_unions)
    print(f"{'LeGrad vanilla mIoU':<25} {corr_v:>11.2f}% {incorr_v:>11.2f}% {corr_v-incorr_v:>9.2f}")
    
    corr_v_acc = global_acc(correct_vanilla_pixel_correct, correct_vanilla_pixel_labeled)
    incorr_v_acc = global_acc(incorrect_vanilla_pixel_correct, incorrect_vanilla_pixel_labeled)
    print(f"{'LeGrad vanilla Acc':<25} {corr_v_acc:>11.2f}% {incorr_v_acc:>11.2f}% {corr_v_acc-incorr_v_acc:>9.2f}")
    
    corr_v_ap = mean_met(correct_vanilla_aps)
    incorr_v_ap = mean_met(incorrect_vanilla_aps)
    print(f"{'LeGrad vanilla mAP':<25} {corr_v_ap:>11.2f}% {incorr_v_ap:>11.2f}% {corr_v_ap-incorr_v_ap:>9.2f}")
    
    corr_v_auroc = mean_met(correct_vanilla_aurocs)
    incorr_v_auroc = mean_met(incorrect_vanilla_aurocs)
    print(f"{'LeGrad vanilla AUROC':<25} {corr_v_auroc:>11.2f}% {incorr_v_auroc:>11.2f}% {corr_v_auroc-incorr_v_auroc:>9.2f}")
    print()
    
    # OMP
    corr_o = global_miou(correct_omp_inters, correct_omp_unions)
    incorr_o = global_miou(incorrect_omp_inters, incorrect_omp_unions)
    print(f"{'LeGrad+OMP mIoU':<25} {corr_o:>11.2f}% {incorr_o:>11.2f}% {corr_o-incorr_o:>9.2f}")
    
    corr_o_acc = global_acc(correct_omp_pixel_correct, correct_omp_pixel_labeled)
    incorr_o_acc = global_acc(incorrect_omp_pixel_correct, incorrect_omp_pixel_labeled)
    print(f"{'LeGrad+OMP Acc':<25} {corr_o_acc:>11.2f}% {incorr_o_acc:>11.2f}% {corr_o_acc-incorr_o_acc:>9.2f}")
    
    corr_o_ap = mean_met(correct_omp_aps)
    incorr_o_ap = mean_met(incorrect_omp_aps)
    print(f"{'LeGrad+OMP mAP':<25} {corr_o_ap:>11.2f}% {incorr_o_ap:>11.2f}% {corr_o_ap-incorr_o_ap:>9.2f}")
    
    corr_o_auroc = mean_met(correct_omp_aurocs)
    incorr_o_auroc = mean_met(incorrect_omp_aurocs)
    print(f"{'LeGrad+OMP AUROC':<25} {corr_o_auroc:>11.2f}% {incorr_o_auroc:>11.2f}% {corr_o_auroc-incorr_o_auroc:>9.2f}")
    print()
    
    # ReLU Diff
    corr_d = global_miou(correct_diff_inters, correct_diff_unions)
    incorr_d = global_miou(incorrect_diff_inters, incorrect_diff_unions)
    print(f"{'ReLU Diff mIoU':<25} {corr_d:>11.2f}% {incorr_d:>11.2f}% {corr_d-incorr_d:>9.2f}")
    
    # Signed Diff
    corr_s = global_miou(correct_signed_inters, correct_signed_unions)
    incorr_s = global_miou(incorrect_signed_inters, incorrect_signed_unions)
    print(f"{'Signed Diff mIoU':<25} {corr_s:>11.2f}% {incorr_s:>11.2f}% {corr_s-incorr_s:>9.2f}")
    print("-" * 60)
    
    # Per-image mean mIoU
    correct_diff_mious = [r['diff_miou'] for r in results['correct']]
    incorrect_diff_mious = [r['diff_miou'] for r in results['incorrect']]
    correct_signed_mious = [r['signed_diff_miou'] for r in results['correct']]
    incorrect_signed_mious = [r['signed_diff_miou'] for r in results['incorrect']]
    correct_focus_drops = [r['focus_drop'] for r in results['correct']]
    incorrect_focus_drops = [r['focus_drop'] for r in results['incorrect']]
    correct_omp_drops = [r['omp_drop'] for r in results['correct']]
    incorrect_omp_drops = [r['omp_drop'] for r in results['incorrect']]
    
    # Focus Change Drop = avg(vanilla, OMP) mIoU - signed diff mIoU
    corr_avg = (corr_v + corr_o) / 2.0
    incorr_avg = (incorr_v + incorr_o) / 2.0
    corr_drop = corr_avg - corr_s
    incorr_drop = incorr_avg - incorr_s
    print(f"{'Focus Change Drop':<25} {corr_drop:>11.2f}  {incorr_drop:>11.2f}  {corr_drop-incorr_drop:>9.2f}")
    
    # OMP Drop = OMP mIoU - signed diff mIoU
    corr_omp_drop = corr_o - corr_s
    incorr_omp_drop = incorr_o - incorr_s
    print(f"{'OMP Drop':<25} {corr_omp_drop:>11.2f}  {incorr_omp_drop:>11.2f}  {corr_omp_drop-incorr_omp_drop:>9.2f}")
    
    if correct_diff_mious:
        print(f"\n  Correct   ReLU diff mIoU  (per-image): {100*np.mean(correct_diff_mious):.2f}% ± {100*np.std(correct_diff_mious):.2f}%")
    if incorrect_diff_mious:
        print(f"  Incorrect ReLU diff mIoU  (per-image): {100*np.mean(incorrect_diff_mious):.2f}% ± {100*np.std(incorrect_diff_mious):.2f}%")
    if correct_signed_mious:
        print(f"  Correct   signed diff mIoU(per-image): {100*np.mean(correct_signed_mious):.2f}% ± {100*np.std(correct_signed_mious):.2f}%")
    if incorrect_signed_mious:
        print(f"  Incorrect signed diff mIoU(per-image): {100*np.mean(incorrect_signed_mious):.2f}% ± {100*np.std(incorrect_signed_mious):.2f}%")
    if correct_focus_drops:
        print(f"  Correct   focus drop      (per-image): {100*np.mean(correct_focus_drops):.2f}% ± {100*np.std(correct_focus_drops):.2f}%")
    if incorrect_focus_drops:
        print(f"  Incorrect focus drop      (per-image): {100*np.mean(incorrect_focus_drops):.2f}% ± {100*np.std(incorrect_focus_drops):.2f}%")
    if correct_omp_drops:
        print(f"  Correct   OMP drop        (per-image): {100*np.mean(correct_omp_drops):.2f}% ± {100*np.std(correct_omp_drops):.2f}%")
    if incorrect_omp_drops:
        print(f"  Incorrect OMP drop        (per-image): {100*np.mean(incorrect_omp_drops):.2f}% ± {100*np.std(incorrect_omp_drops):.2f}%")
    
    # ------------------------------------------------------------------
    # 8. Plot Histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Histogram 1: Vanilla LeGrad mIoU
    correct_vanilla_mious = [r['vanilla_miou'] for r in results['correct']]
    incorrect_vanilla_mious = [r['vanilla_miou'] for r in results['incorrect']]
    
    ax = axes[0, 0]
    if correct_vanilla_mious:
        ax.hist(correct_vanilla_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_vanilla_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_vanilla_mious:
        ax.hist(incorrect_vanilla_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_vanilla_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('LeGrad Vanilla vs GT', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Histogram 2: OMP LeGrad mIoU
    correct_omp_mious = [r['omp_miou'] for r in results['correct']]
    incorrect_omp_mious = [r['omp_miou'] for r in results['incorrect']]
    
    ax = axes[0, 1]
    if correct_omp_mious:
        ax.hist(correct_omp_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_omp_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_omp_mious:
        ax.hist(incorrect_omp_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_omp_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('LeGrad+OMP vs GT', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Histogram 3: ReLU Difference mIoU
    ax = axes[0, 2]
    if correct_diff_mious:
        ax.hist(correct_diff_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_diff_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_diff_mious:
        ax.hist(incorrect_diff_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_diff_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('ReLU(OMP − Vanilla) vs GT', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Histogram 4: Signed Difference mIoU
    ax = axes[1, 0]
    if correct_signed_mious:
        ax.hist(correct_signed_mious, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (n={len(correct_signed_mious)})', edgecolor='white', linewidth=0.5)
    if incorrect_signed_mious:
        ax.hist(incorrect_signed_mious, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (n={len(incorrect_signed_mious)})', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('mIoU', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Signed (OMP − Vanilla) vs GT', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Histogram 5: Focus Change Drop
    ax = axes[1, 1]
    if correct_omp_drops:
        ax.hist(correct_omp_drops, bins=30, alpha=0.6, color='#2196F3', label=f'Correct (μ={100*np.mean(correct_omp_drops):.1f}%)', edgecolor='white', linewidth=0.5)
    if incorrect_omp_drops:
        ax.hist(incorrect_omp_drops, bins=30, alpha=0.6, color='#F44336', label=f'Incorrect (μ={100*np.mean(incorrect_omp_drops):.1f}%)', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('OMP Drop (OMP mIoU − signed diff mIoU)', fontsize=10)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('OMP Drop', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 6: summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"Classification (top-{args.top_k}):\n"
        f"  Correct:   {num_correct}/{total} ({100*num_correct/max(total,1):.1f}%)\n"
        f"  Incorrect: {num_incorrect}/{total} ({100*num_incorrect/max(total,1):.1f}%)\n\n"
        f"OMP Drop (OMP mIoU - signed diff):\n"
        f"  Correct:   {100*np.mean(correct_omp_drops):.1f}% ± {100*np.std(correct_omp_drops):.1f}%\n"
        f"  Incorrect: {100*np.mean(incorrect_omp_drops):.1f}% ± {100*np.std(incorrect_omp_drops):.1f}%\n"
        f"  Gap: {100*(np.mean(correct_omp_drops)-np.mean(incorrect_omp_drops)):+.1f}%"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, va='center', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
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
        ncols = 8  # Original, GT, Vanilla, OMP, Avg, ReLU Diff, Signed Diff, Diff Binary
        fig, axes = plt.subplots(n, ncols, figsize=(ncols * 2.5, n * 2.5))
        if n == 1:
            axes = axes[np.newaxis, :]  # ensure 2D
        
        # Column headers
        col_titles = ['Original', 'GT Mask', 'LeGrad', 'LeGrad+OMP',
                       'Avg(V,O)', 'Signed(Δ)', 'ReLU(Δ)', 'Δ Binary']
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=9, fontweight='bold', pad=8)
        
        for i, vd in enumerate(vis_list):
            # Col 0: Original image
            axes[i, 0].imshow(vd['base_img'])
            label = f"GT: {vd['gt_class']}"
            if vd['predicted_class'] != vd['gt_class']:
                label += f"\nPred: {vd['predicted_class']}"
            axes[i, 0].set_ylabel(f"#{vd['idx']}", fontsize=8, rotation=0, labelpad=25, va='center')
            axes[i, 0].text(0.02, 0.98, label, transform=axes[i, 0].transAxes,
                           fontsize=5, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 1: GT mask
            axes[i, 1].imshow(vd['gt_mask'], cmap='gray', vmin=0, vmax=1)
            
            # Col 2: Vanilla heatmap
            axes[i, 2].imshow(vd['heatmap_vanilla'], cmap='jet', vmin=0, vmax=1)
            axes[i, 2].text(0.02, 0.02, f"{100*vd['vanilla_miou']:.1f}%",
                           transform=axes[i, 2].transAxes, fontsize=6, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 3: OMP heatmap
            axes[i, 3].imshow(vd['heatmap_omp'], cmap='jet', vmin=0, vmax=1)
            axes[i, 3].text(0.02, 0.02, f"{100*vd['omp_miou']:.1f}%",
                           transform=axes[i, 3].transAxes, fontsize=6, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 4: Average heatmap (vanilla + OMP) / 2
            axes[i, 4].imshow(vd['heatmap_avg'], cmap='jet', vmin=0, vmax=1)
            axes[i, 4].text(0.02, 0.02, f"{100*vd['avg_miou']:.1f}%",
                           transform=axes[i, 4].transAxes, fontsize=6, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 5: Signed difference (diverging colormap: blue=negative, red=positive)
            axes[i, 5].imshow(vd['diff_signed'], cmap='RdBu_r', vmin=-1, vmax=1)
            drop_val = vd['omp_drop']
            axes[i, 5].text(0.02, 0.02, f"drop:{100*drop_val:.0f}%",
                           transform=axes[i, 5].transAxes, fontsize=6, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 6: ReLU difference heatmap
            axes[i, 6].imshow(vd['diff_relu_norm'], cmap='hot', vmin=0, vmax=1)
            axes[i, 6].text(0.02, 0.02, f"{100*vd['diff_miou']:.1f}%",
                           transform=axes[i, 6].transAxes, fontsize=6, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Col 7: Difference binary (thresholded ReLU)
            diff_binary = (vd['diff_relu_norm'] > threshold).astype(np.uint8)
            axes[i, 7].imshow(diff_binary, cmap='gray', vmin=0, vmax=1)
            
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
            'correct_relu_diff_miou': float(corr_d),
            'incorrect_relu_diff_miou': float(incorr_d),
            'correct_signed_diff_miou': float(corr_s),
            'incorrect_signed_diff_miou': float(incorr_s),
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
