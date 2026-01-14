#!/usr/bin/env python3
"""
Evaluate both LeGrad and Sparse LeGrad on CORRECT and WRONG (negative) prompts.

Shows side-by-side comparison of:
- LeGrad (baseline) with correct/wrong prompts
- Sparse LeGrad (with given hyperparameters) with correct/wrong prompts

Usage:
    python scripts/evaluate_positive_negative.py \
        --wn_use_hypernyms 1 \
        --sparse_threshold 0.4 \
        --atoms 20 \
        --max_dict_cos_sim 0.5 \
        --limit 50
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
from sklearn.metrics import average_precision_score
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

from benchmark_segmentation import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    compute_iou_acc,
    compute_map_score,
    omp_sparse_residual,
    wordnet_neighbors_configured,
)


def compute_legrad_heatmap(model, image, text_emb_1x):
    """Compute LeGrad heatmap for a single text embedding."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


def main():
    parser = argparse.ArgumentParser(description='Evaluate Positive/Negative Prompts')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Negative sampling
    parser.add_argument('--num_negatives', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Sparse encoding parameters
    parser.add_argument('--atoms', type=int, default=8)
    parser.add_argument('--dict_include_prompts', type=int, default=1)
    parser.add_argument('--wn_use_synonyms', type=int, default=0)
    parser.add_argument('--wn_use_hypernyms', type=int, default=0)
    parser.add_argument('--wn_use_hyponyms', type=int, default=0)
    parser.add_argument('--wn_use_siblings', type=int, default=1)
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.9)
    
    # Thresholds
    parser.add_argument('--legrad_threshold', type=float, default=0.5)
    parser.add_argument('--sparse_threshold', type=float, default=0.5)
    
    # Composite lambda
    parser.add_argument('--composite_lambda', type=float, default=0.5)
    
    # Output
    parser.add_argument('--output_json', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load model
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
    
    # Load NLTK
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
        print(f"[imagenet] Warning: {e}")
        wnid_to_label = {}
    
    # Load dataset
    print(f"Opening dataset {args.mat_file}...")
    f = h5py.File(args.mat_file, 'r')
    imgs_refs = f['value/img']
    gts_refs = f['value/gt']
    targets_refs = f['value/target']
    num_images = imgs_refs.shape[0]
    
    limit = min(args.limit if args.limit > 0 else num_images, num_images)
    
    # Precompute embeddings
    print("[eval] Precomputing class embeddings...")
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = ''.join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)
    
    unique_wnids = sorted(set(wnids_in_seg))
    wnid_to_classname = {}
    wnid_to_prompt = {}
    for wnid in unique_wnids:
        label = wnid_to_label.get(wnid) or get_synset_name(wnid)
        wnid_to_classname[wnid] = label
        wnid_to_prompt[wnid] = f"a photo of a {label}."
    
    all_prompts = [wnid_to_prompt[w] for w in unique_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}
    idx_to_wnid = {i: w for w, i in wnid_to_idx.items()}
    
    tok_all = tokenizer(all_prompts).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)
    
    print(f"[eval] Precomputed {len(all_prompts)} class embeddings.")
    
    # Random generator for negative sampling
    rng = random.Random(args.seed)
    
    # Results storage
    results = {
        'legrad': {'correct': {'iou': [], 'acc': [], 'ap': []}, 'wrong': {'iou': [], 'acc': [], 'ap': []}},
        'sparse': {'correct': {'iou': [], 'acc': [], 'ap': []}, 'wrong': {'iou': [], 'acc': [], 'ap': []}},
    }
    
    print(f"\n{'='*70}")
    print("Sparse Encoding Parameters:")
    print(f"  atoms={args.atoms}, dict_include_prompts={args.dict_include_prompts}")
    print(f"  wn_synonyms={args.wn_use_synonyms}, wn_hypernyms={args.wn_use_hypernyms}")
    print(f"  wn_hyponyms={args.wn_use_hyponyms}, wn_siblings={args.wn_use_siblings}")
    print(f"  max_dict_cos_sim={args.max_dict_cos_sim}")
    print(f"  legrad_threshold={args.legrad_threshold}, sparse_threshold={args.sparse_threshold}")
    print(f"{'='*70}\n")
    
    for idx in tqdm(range(limit), desc="Evaluating"):
        try:
            # Load image
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
            cls_idx = wnid_to_idx[wnid]
            class_name = wnid_to_classname[wnid]
            text_emb = all_text_embs[cls_idx:cls_idx + 1]
            
            def build_sparse_emb(original_emb, target_class_name):
                """Build sparse residual embedding."""
                parts = []
                
                # Other class prompts
                if bool(args.dict_include_prompts) and len(unique_wnids) > 1:
                    emb_idx = None
                    for i, w in enumerate(unique_wnids):
                        if wnid_to_classname[w] == target_class_name:
                            emb_idx = i
                            break
                    if emb_idx is not None:
                        if emb_idx > 0:
                            parts.append(all_text_embs[:emb_idx])
                        if emb_idx + 1 < len(unique_wnids):
                            parts.append(all_text_embs[emb_idx + 1:])
                    else:
                        parts.append(all_text_embs)
                
                # WordNet neighbors
                use_wn = any([args.wn_use_synonyms, args.wn_use_hypernyms, 
                              args.wn_use_hyponyms, args.wn_use_siblings])
                if use_wn:
                    prompt = f"a photo of a {target_class_name}."
                    neighbors = wordnet_neighbors_configured(
                        target_class_name,
                        use_synonyms=bool(args.wn_use_synonyms),
                        use_hypernyms=bool(args.wn_use_hypernyms),
                        use_hyponyms=bool(args.wn_use_hyponyms),
                        use_siblings=bool(args.wn_use_siblings),
                        use_fallback=True,
                        limit_per_relation=8,
                    )
                    if neighbors:
                        neighbor_prompts = [prompt.replace(target_class_name, w) for w in neighbors]
                        n_tok = tokenizer(neighbor_prompts).to(args.device)
                        with torch.no_grad():
                            n_emb = model.encode_text(n_tok)
                            n_emb = F.normalize(n_emb, dim=-1)
                        parts.append(n_emb)
                
                if len(parts) > 0:
                    D = torch.cat(parts, dim=0)
                    D = F.normalize(D, dim=-1)
                    
                    # Filter by cosine similarity
                    if 0.0 < args.max_dict_cos_sim < 1.0:
                        sim = (D @ original_emb.t()).squeeze(-1).abs()
                        keep = sim < args.max_dict_cos_sim
                        D = D[keep]
                else:
                    D = original_emb.new_zeros((0, original_emb.shape[-1]))
                
                sparse_emb = omp_sparse_residual(original_emb, D, max_atoms=args.atoms)
                return sparse_emb
            
            def compute_metrics(emb, threshold):
                """Compute heatmap and metrics."""
                heatmap = compute_legrad_heatmap(model, img_t, emb)
                heatmap_resized = F.interpolate(
                    heatmap.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                iou, acc = compute_iou_acc(heatmap_resized, gt_mask, threshold=threshold)
                ap = compute_map_score(heatmap_resized, gt_mask)
                return iou, acc, ap
            
            # === CORRECT PROMPT ===
            # LeGrad
            iou, acc, ap = compute_metrics(text_emb, args.legrad_threshold)
            results['legrad']['correct']['iou'].append(iou)
            results['legrad']['correct']['acc'].append(acc)
            results['legrad']['correct']['ap'].append(ap)
            
            # Sparse
            sparse_emb = build_sparse_emb(text_emb, class_name)
            iou, acc, ap = compute_metrics(sparse_emb, args.sparse_threshold)
            results['sparse']['correct']['iou'].append(iou)
            results['sparse']['correct']['acc'].append(acc)
            results['sparse']['correct']['ap'].append(ap)
            
            # === WRONG PROMPTS ===
            all_indices = list(range(len(unique_wnids)))
            all_indices.remove(cls_idx)
            neg_indices = rng.sample(all_indices, min(args.num_negatives, len(all_indices)))
            
            for neg_idx in neg_indices:
                neg_wnid = idx_to_wnid[neg_idx]
                neg_class_name = wnid_to_classname[neg_wnid]
                neg_emb = all_text_embs[neg_idx:neg_idx + 1]
                
                # LeGrad
                iou, acc, ap = compute_metrics(neg_emb, args.legrad_threshold)
                results['legrad']['wrong']['iou'].append(iou)
                results['legrad']['wrong']['acc'].append(acc)
                results['legrad']['wrong']['ap'].append(ap)
                
                # Sparse
                sparse_neg_emb = build_sparse_emb(neg_emb, neg_class_name)
                iou, acc, ap = compute_metrics(sparse_neg_emb, args.sparse_threshold)
                results['sparse']['wrong']['iou'].append(iou)
                results['sparse']['wrong']['acc'].append(acc)
                results['sparse']['wrong']['ap'].append(ap)
                
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            continue
    
    # Compute averages
    def avg(lst):
        return np.mean(lst) * 100 if lst else 0.0
    
    legrad_correct_miou = avg(results['legrad']['correct']['iou'])
    legrad_correct_acc = avg(results['legrad']['correct']['acc'])
    legrad_correct_map = avg(results['legrad']['correct']['ap'])
    legrad_wrong_miou = avg(results['legrad']['wrong']['iou'])
    legrad_wrong_acc = avg(results['legrad']['wrong']['acc'])
    legrad_wrong_map = avg(results['legrad']['wrong']['ap'])
    
    sparse_correct_miou = avg(results['sparse']['correct']['iou'])
    sparse_correct_acc = avg(results['sparse']['correct']['acc'])
    sparse_correct_map = avg(results['sparse']['correct']['ap'])
    sparse_wrong_miou = avg(results['sparse']['wrong']['iou'])
    sparse_wrong_acc = avg(results['sparse']['wrong']['acc'])
    sparse_wrong_map = avg(results['sparse']['wrong']['ap'])
    
    legrad_composite = legrad_correct_miou - args.composite_lambda * legrad_wrong_miou
    sparse_composite = sparse_correct_miou - args.composite_lambda * sparse_wrong_miou
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<12} | {'Prompt':<8} | {'mIoU':>8} | {'Acc':>8} | {'mAP':>8}")
    print("-" * 70)
    print(f"{'LeGrad':<12} | {'Correct':<8} | {legrad_correct_miou:>8.2f} | {legrad_correct_acc:>8.2f} | {legrad_correct_map:>8.2f}")
    print(f"{'LeGrad':<12} | {'Wrong':<8} | {legrad_wrong_miou:>8.2f} | {legrad_wrong_acc:>8.2f} | {legrad_wrong_map:>8.2f}")
    print("-" * 70)
    print(f"{'Sparse':<12} | {'Correct':<8} | {sparse_correct_miou:>8.2f} | {sparse_correct_acc:>8.2f} | {sparse_correct_map:>8.2f}")
    print(f"{'Sparse':<12} | {'Wrong':<8} | {sparse_wrong_miou:>8.2f} | {sparse_wrong_acc:>8.2f} | {sparse_wrong_map:>8.2f}")
    
    print(f"\n{'='*70}")
    print("COMPOSITE SCORES (correct_mIoU - λ × wrong_mIoU)")
    print(f"{'='*70}")
    print(f"LeGrad:  {legrad_correct_miou:.2f} - {args.composite_lambda} × {legrad_wrong_miou:.2f} = {legrad_composite:.2f}")
    print(f"Sparse:  {sparse_correct_miou:.2f} - {args.composite_lambda} × {sparse_wrong_miou:.2f} = {sparse_composite:.2f}")
    print(f"\nImprovement: {sparse_composite - legrad_composite:+.2f}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    correct_diff = sparse_correct_miou - legrad_correct_miou
    wrong_diff = sparse_wrong_miou - legrad_wrong_miou
    print(f"Correct mIoU change: {correct_diff:+.2f} ({'↑ better' if correct_diff > 0 else '↓ worse'})")
    print(f"Wrong mIoU change:   {wrong_diff:+.2f} ({'↓ better (less hallucination)' if wrong_diff < 0 else '↑ worse (more hallucination)'})")
    
    # Save to JSON if requested
    if args.output_json:
        output = {
            'legrad': {
                'correct': {'miou': legrad_correct_miou, 'acc': legrad_correct_acc, 'map': legrad_correct_map},
                'wrong': {'miou': legrad_wrong_miou, 'acc': legrad_wrong_acc, 'map': legrad_wrong_map},
                'composite': legrad_composite,
            },
            'sparse': {
                'correct': {'miou': sparse_correct_miou, 'acc': sparse_correct_acc, 'map': sparse_correct_map},
                'wrong': {'miou': sparse_wrong_miou, 'acc': sparse_wrong_acc, 'map': sparse_wrong_map},
                'composite': sparse_composite,
            },
            'improvement': sparse_composite - legrad_composite,
            'settings': vars(args),
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
