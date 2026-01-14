#!/usr/bin/env python3
"""
Evaluate Sparse LeGrad on CORRECT and WRONG (negative) prompts.
Compares against known LeGrad baseline values.
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
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    import nltk
except ImportError:
    pass

from legrad import LeWrapper, LePreprocess
import open_clip

from benchmark_segmentation import (
    load_imagenet_class_index, build_wnid_to_label_map, get_synset_name,
    compute_iou_acc, compute_map_score, omp_sparse_residual, wordnet_neighbors_configured,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sparse Encoding on Pos/Neg Prompts')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    parser.add_argument('--num_negatives', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Sparse encoding params
    parser.add_argument('--atoms', type=int, default=8)
    parser.add_argument('--dict_include_prompts', type=int, default=1)
    parser.add_argument('--wn_use_synonyms', type=int, default=0)
    parser.add_argument('--wn_use_hypernyms', type=int, default=0)
    parser.add_argument('--wn_use_hyponyms', type=int, default=0)
    parser.add_argument('--wn_use_siblings', type=int, default=1)
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.9)
    parser.add_argument('--sparse_threshold', type=float, default=0.5)
    
    # Known baseline values
    parser.add_argument('--baseline_correct_miou', type=float, default=58.66)
    parser.add_argument('--baseline_correct_acc', type=float, default=77.52)
    parser.add_argument('--baseline_correct_map', type=float, default=82.49)
    parser.add_argument('--baseline_wrong_miou', type=float, default=41.63)
    parser.add_argument('--baseline_wrong_acc', type=float, default=67.55)
    parser.add_argument('--baseline_wrong_map', type=float, default=71.47)
    parser.add_argument('--composite_lambda', type=float, default=0.5)
    
    parser.add_argument('--output_json', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass
    
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
    except:
        wnid_to_label = {}
    
    f = h5py.File(args.mat_file, 'r')
    imgs_refs, gts_refs, targets_refs = f['value/img'], f['value/gt'], f['value/target']
    num_images = imgs_refs.shape[0]
    limit = min(args.limit if args.limit > 0 else num_images, num_images)
    
    # Precompute embeddings
    wnids_in_seg = []
    for idx in range(num_images):
        target_data = np.array(f[targets_refs[idx, 0]])
        wnids_in_seg.append(''.join([chr(c) for c in target_data.flatten()]))
    
    unique_wnids = sorted(set(wnids_in_seg))
    wnid_to_classname = {w: wnid_to_label.get(w) or get_synset_name(w) for w in unique_wnids}
    wnid_to_prompt = {w: f"a photo of a {wnid_to_classname[w]}." for w in unique_wnids}
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}
    idx_to_wnid = {i: w for w, i in wnid_to_idx.items()}
    
    tok_all = tokenizer([wnid_to_prompt[w] for w in unique_wnids]).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)
    
    rng = random.Random(args.seed)
    correct_results = {'iou': [], 'acc': [], 'ap': []}
    wrong_results = {'iou': [], 'acc': [], 'ap': []}
    
    print(f"\n{'='*60}")
    print(f"Sparse params: atoms={args.atoms}, prompts={args.dict_include_prompts}, "
          f"syn={args.wn_use_synonyms}, hyper={args.wn_use_hypernyms}, "
          f"hypo={args.wn_use_hyponyms}, sib={args.wn_use_siblings}, "
          f"cos_sim={args.max_dict_cos_sim}, thr={args.sparse_threshold}")
    print(f"{'='*60}\n")
    
    for idx in tqdm(range(limit), desc="Evaluating Sparse"):
        try:
            img_obj = np.array(f[imgs_refs[idx, 0]])
            base_img = Image.fromarray(img_obj.transpose(2, 1, 0))
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            H_feat, W_feat = img_t.shape[-2:]
            
            gt_wrapper = f[gts_refs[idx, 0]]
            if gt_wrapper.dtype == 'object':
                gt_mask = np.array(f[gt_wrapper[0, 0]]).transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
            
            gt_pil = transforms.Resize((args.image_size, args.image_size), 
                                        interpolation=InterpolationMode.NEAREST)(
                Image.fromarray(gt_mask.astype(np.uint8)))
            gt_mask = np.array(gt_pil).astype(np.uint8)
            H_gt, W_gt = gt_mask.shape
            
            wnid = wnids_in_seg[idx]
            cls_idx = wnid_to_idx[wnid]
            class_name = wnid_to_classname[wnid]
            text_emb = all_text_embs[cls_idx:cls_idx + 1]
            
            def build_sparse(emb, target_name):
                parts = []
                if bool(args.dict_include_prompts) and len(unique_wnids) > 1:
                    emb_idx = next((i for i, w in enumerate(unique_wnids) 
                                   if wnid_to_classname[w] == target_name), None)
                    if emb_idx is not None:
                        if emb_idx > 0: parts.append(all_text_embs[:emb_idx])
                        if emb_idx + 1 < len(unique_wnids): parts.append(all_text_embs[emb_idx + 1:])
                    else:
                        parts.append(all_text_embs)
                
                if any([args.wn_use_synonyms, args.wn_use_hypernyms, args.wn_use_hyponyms, args.wn_use_siblings]):
                    neighbors = wordnet_neighbors_configured(
                        target_name, bool(args.wn_use_synonyms), bool(args.wn_use_hypernyms),
                        bool(args.wn_use_hyponyms), bool(args.wn_use_siblings), True, 8)
                    if neighbors:
                        n_tok = tokenizer([f"a photo of a {target_name}.".replace(target_name, w) for w in neighbors]).to(args.device)
                        with torch.no_grad():
                            parts.append(F.normalize(model.encode_text(n_tok), dim=-1))
                
                if parts:
                    D = F.normalize(torch.cat(parts, 0), dim=-1)
                    if 0 < args.max_dict_cos_sim < 1:
                        D = D[(D @ emb.t()).squeeze(-1).abs() < args.max_dict_cos_sim]
                else:
                    D = emb.new_zeros((0, emb.shape[-1]))
                return omp_sparse_residual(emb, D, max_atoms=args.atoms)
            
            def compute_metrics(emb):
                with torch.enable_grad():
                    heatmap = model.compute_legrad(image=img_t, text_embedding=emb)[0, 0].clamp(0, 1).detach().cpu()
                heatmap = F.interpolate(heatmap.view(1, 1, H_feat, W_feat), (H_gt, W_gt), 
                                        mode='bilinear', align_corners=False).squeeze().numpy()
                iou, acc = compute_iou_acc(heatmap, gt_mask, args.sparse_threshold)
                return iou, acc, compute_map_score(heatmap, gt_mask)
            
            # Correct prompt
            sparse_emb = build_sparse(text_emb, class_name)
            iou, acc, ap = compute_metrics(sparse_emb)
            correct_results['iou'].append(iou)
            correct_results['acc'].append(acc)
            correct_results['ap'].append(ap)
            
            # Wrong prompts
            all_indices = list(range(len(unique_wnids)))
            all_indices.remove(cls_idx)
            for neg_idx in rng.sample(all_indices, min(args.num_negatives, len(all_indices))):
                neg_name = wnid_to_classname[idx_to_wnid[neg_idx]]
                neg_emb = all_text_embs[neg_idx:neg_idx + 1]
                sparse_neg = build_sparse(neg_emb, neg_name)
                iou, acc, ap = compute_metrics(sparse_neg)
                wrong_results['iou'].append(iou)
                wrong_results['acc'].append(acc)
                wrong_results['ap'].append(ap)
        except Exception as e:
            continue
    
    # Compute averages
    def avg(lst): return np.mean(lst) * 100 if lst else 0.0
    
    sparse_correct = {'miou': avg(correct_results['iou']), 'acc': avg(correct_results['acc']), 'map': avg(correct_results['ap'])}
    sparse_wrong = {'miou': avg(wrong_results['iou']), 'acc': avg(wrong_results['acc']), 'map': avg(wrong_results['ap'])}
    
    baseline_composite = args.baseline_correct_miou - args.composite_lambda * args.baseline_wrong_miou
    sparse_composite = sparse_correct['miou'] - args.composite_lambda * sparse_wrong['miou']
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n{'':15} | {'mIoU':>10} | {'Acc':>10} | {'mAP':>10}")
    print("-" * 55)
    print(f"{'Baseline Corr':<15} | {args.baseline_correct_miou:>10.2f} | {args.baseline_correct_acc:>10.2f} | {args.baseline_correct_map:>10.2f}")
    print(f"{'Sparse Correct':<15} | {sparse_correct['miou']:>10.2f} | {sparse_correct['acc']:>10.2f} | {sparse_correct['map']:>10.2f}")
    print(f"{'Δ Correct':<15} | {sparse_correct['miou']-args.baseline_correct_miou:>+10.2f} | {sparse_correct['acc']-args.baseline_correct_acc:>+10.2f} | {sparse_correct['map']-args.baseline_correct_map:>+10.2f}")
    print("-" * 55)
    print(f"{'Baseline Wrong':<15} | {args.baseline_wrong_miou:>10.2f} | {args.baseline_wrong_acc:>10.2f} | {args.baseline_wrong_map:>10.2f}")
    print(f"{'Sparse Wrong':<15} | {sparse_wrong['miou']:>10.2f} | {sparse_wrong['acc']:>10.2f} | {sparse_wrong['map']:>10.2f}")
    print(f"{'Δ Wrong':<15} | {sparse_wrong['miou']-args.baseline_wrong_miou:>+10.2f} | {sparse_wrong['acc']-args.baseline_wrong_acc:>+10.2f} | {sparse_wrong['map']-args.baseline_wrong_map:>+10.2f}")
    
    print(f"\n{'='*70}")
    print("COMPOSITE SCORE (correct_mIoU - λ × wrong_mIoU)")
    print(f"{'='*70}")
    print(f"Baseline: {args.baseline_correct_miou:.2f} - {args.composite_lambda}×{args.baseline_wrong_miou:.2f} = {baseline_composite:.2f}")
    print(f"Sparse:   {sparse_correct['miou']:.2f} - {args.composite_lambda}×{sparse_wrong['miou']:.2f} = {sparse_composite:.2f}")
    print(f"\n🎯 Improvement: {sparse_composite - baseline_composite:+.2f}")
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({'sparse_correct': sparse_correct, 'sparse_wrong': sparse_wrong, 
                      'sparse_composite': sparse_composite, 'improvement': sparse_composite - baseline_composite}, f, indent=2)


if __name__ == '__main__':
    main()
