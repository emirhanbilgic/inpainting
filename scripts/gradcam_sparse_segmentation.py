#!/usr/bin/env python3
"""
GradCAM Sparse Segmentation Benchmark

This script evaluates GradCAM with sparse embeddings on ImageNet-Segmentation.
Compares:
  1. Original GradCAM (with original text embedding)
  2. Sparse GradCAM (with OMP sparse residual embedding)

Usage:
    python scripts/gradcam_sparse_segmentation.py
    python scripts/gradcam_sparse_segmentation.py --limit 100  # faster test
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
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
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

# -----------------------------------------------------------------------------
# Helper functions (imported/copied from benchmark_segmentation.py)
# -----------------------------------------------------------------------------

IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def download_imagenet_class_index(dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(dst_path, "w") as f:
        json.dump(data, f)
    return data


def load_imagenet_class_index(path: str):
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    print(f"[imagenet] class index not found at {path}, downloading...")
    return download_imagenet_class_index(path)


def build_wnid_to_label_map(class_index):
    wnid_to_label = {}
    for _, (wnid, label_str) in class_index.items():
        primary = label_str.split(",")[0].replace("_", " ").strip()
        wnid_to_label[wnid] = primary.lower()
    return wnid_to_label


def get_synset_name(wnid: str) -> str:
    if wn is None:
        return wnid
    try:
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        name = synset.lemmas()[0].name().replace('_', ' ')
        return name
    except Exception:
        return wnid


def compute_iou_acc(heatmap: np.ndarray, gt_mask: np.ndarray, threshold: float):
    """Compute 2-class mIoU and pixel accuracy."""
    pred_mask = (heatmap > threshold).astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    # Foreground IoU
    fg_intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fg_union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
    iou_fg = fg_intersection / (fg_union + 1e-6)

    # Background IoU
    bg_intersection = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
    bg_union = np.logical_or(pred_mask == 0, gt_mask == 0).sum()
    iou_bg = bg_intersection / (bg_union + 1e-6)

    # 2-class mean IoU
    iou = (iou_bg + iou_fg) / 2.0

    # Pixel accuracy
    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size
    acc = correct / total

    return iou, acc


def compute_map_score(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute 2-class mean average precision."""
    gt_mask = gt_mask.astype(int)
    
    fg_prob = heatmap.flatten()
    bg_prob = 1.0 - fg_prob
    
    gt_flat = gt_mask.flatten()
    fg_target = (gt_flat == 1).astype(int)
    bg_target = (gt_flat == 0).astype(int)
    
    all_scores = np.concatenate([bg_prob, fg_prob])
    all_targets = np.concatenate([bg_target, fg_target])
    
    if all_targets.sum() == 0:
        return 0.0
    
    return average_precision_score(all_targets, all_scores)


def compute_gradcam_for_embedding(model, image: torch.Tensor, text_emb_1x: torch.Tensor, layer_index: int = 8):
    """Compute GradCAM heatmap (normalized to [0, 1])."""
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def omp_sparse_residual(
    x_1x: torch.Tensor,
    D: torch.Tensor,
    max_atoms: int = 8,
    tol: float = 1e-6,
    return_num_selected: bool = False,
):
    """Orthogonal Matching Pursuit sparse residual."""
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        r_norm = F.normalize(x_1x, dim=-1)
        return (r_norm, 0) if return_num_selected else r_norm

    x = x_1x.clone()
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)
        x_hat = (s.t() @ D_S).to(x.dtype)
        r = (x - x_hat)
        if float(torch.norm(r)) <= tol:
            break
            
    if torch.norm(r) <= tol:
        r_norm = F.normalize(x, dim=-1)
    else:
        r_norm = F.normalize(r, dim=-1)
    return (r_norm, len(selected)) if return_num_selected else r_norm


def wordnet_neighbors_configured(
    keyword: str,
    use_synonyms: bool,
    use_hypernyms: bool,
    use_hyponyms: bool,
    use_siblings: bool,
    use_fallback: bool = True,
    limit_per_relation: int = 8
):
    """Get WordNet neighbors for a keyword."""
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    def get_neighbors_for_term(term: str):
        out = []
        seen = set()
        key_low = term.lower()
        synsets = wn.synsets(term, pos=wn.NOUN)
        for s in synsets[:limit_per_relation]:
            if use_synonyms:
                for l in s.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name)
                        seen.add(name)
            if use_hypernyms:
                for h in s.hypernyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name)
                            seen.add(name)
            if use_hyponyms:
                for h in s.hyponyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name)
                            seen.add(name)
            if use_siblings:
                for h in s.hypernyms()[:limit_per_relation]:
                    for sib in h.hyponyms()[:limit_per_relation]:
                        for l in sib.lemmas()[:limit_per_relation]:
                            name = l.name().replace('_', ' ').lower()
                            if name != key_low and name not in seen:
                                out.append(name)
                                seen.add(name)
        return out

    out = get_neighbors_for_term(keyword)
    if out:
        return out[:max(1, limit_per_relation * 3)]

    if not use_fallback:
        return []

    words = keyword.split()
    if len(words) > 1:
        for i in range(1, len(words)):
            sub_term = " ".join(words[i:])
            out = get_neighbors_for_term(sub_term)
            if out:
                return out[:max(1, limit_per_relation * 3)]

    return []


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GradCAM Sparse Segmentation Benchmark')
    
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Sparse settings
    parser.add_argument('--atoms', type=int, default=8, help='Number of atoms for OMP')
    parser.add_argument('--dict_include_prompts', type=int, default=1, help='Include other class prompts (0/1)')
    parser.add_argument('--wn_use_synonyms', type=int, default=0, help='WordNet synonyms (0/1)')
    parser.add_argument('--wn_use_hypernyms', type=int, default=0, help='WordNet hypernyms (0/1)')
    parser.add_argument('--wn_use_hyponyms', type=int, default=0, help='WordNet hyponyms (0/1)')
    parser.add_argument('--wn_use_siblings', type=int, default=1, help='WordNet siblings (0/1)')
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.9, help='Max cosine similarity filter')
    
    # Thresholds
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--sparse_threshold', type=float, default=0.5, help='Threshold for sparse GradCAM')
    
    # GradCAM layer
    parser.add_argument('--gradcam_layer', type=int, default=8, help='Layer index for GradCAM')
    
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
        print(f"[imagenet] Loaded {len(wnid_to_label)} wnids.")
    except Exception as e:
        print(f"[imagenet] Warning: {e}")
        wnid_to_label = {}
    
    # Load dataset
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
    
    # Precompute all wnids and embeddings
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
        'gradcam_original': {'iou': [], 'acc': [], 'ap': []},
        'gradcam_sparse': {'iou': [], 'acc': [], 'ap': []},
    }
    
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
            
            # === ORIGINAL GRADCAM ===
            heatmap_gradcam_orig = compute_gradcam_for_embedding(
                model, img_t, original_1x, layer_index=args.gradcam_layer
            )
            heatmap_gradcam_orig_resized = F.interpolate(
                heatmap_gradcam_orig.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # === BUILD SPARSE DICTIONARY ===
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
                    use_fallback=True,
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
                if args.max_dict_cos_sim is not None and 0.0 < args.max_dict_cos_sim < 1.0:
                    sim = (D @ original_1x.t()).squeeze(-1).abs()
                    keep = sim < args.max_dict_cos_sim
                    D = D[keep]
            else:
                D = original_1x.new_zeros((0, original_1x.shape[-1]))
            
            # === OMP SPARSE RESIDUAL ===
            sparse_1x = omp_sparse_residual(original_1x, D, max_atoms=args.atoms)
            
            # === SPARSE GRADCAM ===
            heatmap_gradcam_sparse = compute_gradcam_for_embedding(
                model, img_t, sparse_1x, layer_index=args.gradcam_layer
            )
            heatmap_gradcam_sparse_resized = F.interpolate(
                heatmap_gradcam_sparse.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # === METRICS ===
            # Original GradCAM
            iou_o, acc_o = compute_iou_acc(heatmap_gradcam_orig_resized, gt_mask, threshold=args.threshold)
            ap_o = compute_map_score(heatmap_gradcam_orig_resized, gt_mask)
            results['gradcam_original']['iou'].append(iou_o)
            results['gradcam_original']['acc'].append(acc_o)
            results['gradcam_original']['ap'].append(ap_o)
            
            # Sparse GradCAM
            iou_s, acc_s = compute_iou_acc(heatmap_gradcam_sparse_resized, gt_mask, threshold=args.sparse_threshold)
            ap_s = compute_map_score(heatmap_gradcam_sparse_resized, gt_mask)
            results['gradcam_sparse']['iou'].append(iou_s)
            results['gradcam_sparse']['acc'].append(acc_s)
            results['gradcam_sparse']['ap'].append(ap_s)
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Print Results
    print("\n" + "=" * 60)
    print("GradCAM Sparse Segmentation Results")
    print("=" * 60)
    
    for method in ['gradcam_original', 'gradcam_sparse']:
        miou = np.mean(results[method]['iou']) * 100 if results[method]['iou'] else 0.0
        macc = np.mean(results[method]['acc']) * 100 if results[method]['acc'] else 0.0
        map_score = np.mean(results[method]['ap']) * 100 if results[method]['ap'] else 0.0
        
        label = "GradCAM (Original)" if method == 'gradcam_original' else "GradCAM (Sparse)"
        print(f"{label}: PixelAcc={macc:.2f}, mIoU={miou:.2f}, mAP={map_score:.2f}")
    
    # Improvement
    orig_miou = np.mean(results['gradcam_original']['iou']) * 100
    sparse_miou = np.mean(results['gradcam_sparse']['iou']) * 100
    improvement = sparse_miou - orig_miou
    
    print(f"\nImprovement (Sparse - Original): {improvement:+.2f} mIoU")
    print(f"Processed {limit} images in {elapsed/60.0:.2f} minutes.")
    
    # Summary with sparse settings
    print(f"\n--- Sparse Settings ---")
    print(f"  atoms: {args.atoms}")
    print(f"  dict_include_prompts: {args.dict_include_prompts}")
    print(f"  wn_use_siblings: {args.wn_use_siblings}")
    print(f"  max_dict_cos_sim: {args.max_dict_cos_sim}")
    print(f"  sparse_threshold: {args.sparse_threshold}")


if __name__ == '__main__':
    main()
