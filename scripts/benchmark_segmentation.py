import sys
import os
import argparse
import json
import re
import time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import requests

# Add project root to path (ensure local legrad overrides any installed package)
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
    wn = None  # type: ignore

from legrad import LeWrapper, LePreprocess
import open_clip

# Constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def download_imagenet_class_index(dst_path: str):
    """
    Download the standard ImageNet class index JSON.
    Returns dict: idx_str -> [wnid, human_label].
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(dst_path, "w") as f:
        json.dump(data, f)
    return data


def load_imagenet_class_index(path: str):
    """
    Load (or download) the ImageNet class index JSON.
    """
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    print(f"[imagenet] class index not found at {path}, downloading from {IMAGENET_CLASS_INDEX_URL} ...")
    return download_imagenet_class_index(path)


def build_wnid_to_label_map(class_index):
    """
    Convert imagenet_class_index mapping into wnid -> primary label (lowercase).
    Each entry: idx_str -> [wnid, \"label1, label2, ...\"].
    """
    wnid_to_label = {}
    for _, (wnid, label_str) in class_index.items():
        primary = label_str.split(",")[0].replace("_", " ").strip()
        wnid_to_label[wnid] = primary.lower()
    return wnid_to_label


def get_synset_name(wnid):
    """
    Fallback: derive a name from WordNet if JSON mapping is unavailable.
    """
    if wn is None:
        return wnid
    try:
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        name = synset.lemmas()[0].name().replace('_', ' ')
        return name
    except Exception:
        return wnid

def dynamic_preprocess(img: Image.Image, target_size: int = 448, patch_size: int = 16) -> torch.Tensor:
    """
    Resize image such that short side is target_size, 
    and dimensions are multiples of patch_size.
    Preserves aspect ratio. No cropping.
    """
    w, h = img.size
    scale = target_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Round to nearest multiple of patch_size
    new_w = round(new_w / patch_size) * patch_size
    new_h = round(new_h / patch_size) * patch_size
    
    # Resize
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Convert to tensor
    t = TF.to_tensor(img_resized) # [3, H, W], 0-1
    
    # Normalize
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    t = (t - mean) / std
    
    return t

def compute_iou_acc(heatmap, gt_mask, threshold: float):
    """Compute 2-class mIoU and pixel accuracy for a given heatmap and GT mask.
    
    IMPORTANT: This follows the reference ImageNet-Segmentation evaluation protocol
    from clip_text_span (batch_intersection_union), which computes the mean IoU
    over BOTH background and foreground classes, not just foreground IoU.
    
    mIoU = (IoU_background + IoU_foreground) / 2
    """
    pred_mask = (heatmap > threshold).astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    # Foreground IoU (class 1)
    fg_intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fg_union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
    iou_fg = fg_intersection / (fg_union + 1e-6)

    # Background IoU (class 0)
    bg_intersection = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
    bg_union = np.logical_or(pred_mask == 0, gt_mask == 0).sum()
    iou_bg = bg_intersection / (bg_union + 1e-6)

    # 2-class mean IoU (as per ImageNet-Segmentation protocol)
    iou = (iou_bg + iou_fg) / 2.0

    # Pixel accuracy
    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size
    acc = correct / total
    
    return iou, acc

def compute_map_score(heatmap, gt_mask):
    """Compute 2-class mean average precision for a continuous heatmap vs. GT mask.
    
    IMPORTANT: This follows the reference ImageNet-Segmentation evaluation protocol
    from clip_text_span (get_ap_scores), which computes AP over BOTH classes by:
    1. Creating 2-channel predictions: [1-heatmap, heatmap] for [bg_prob, fg_prob]
    2. One-hot encoding the target
    3. Computing AP over all predictions and targets combined
    
    This gives a 2-class mAP that averages performance on both background and foreground.
    """
    gt_mask = gt_mask.astype(int)
    
    # Create 2-channel predictions: background and foreground probabilities
    fg_prob = heatmap.flatten()
    bg_prob = 1.0 - fg_prob
    
    # One-hot encode the target
    fg_target = (gt_mask.flatten() == 1).astype(int)
    bg_target = (gt_mask.flatten() == 0).astype(int)
    
    # Combine both classes into single arrays (as in reference get_ap_scores)
    all_scores = np.concatenate([bg_prob, fg_prob])
    all_targets = np.concatenate([bg_target, fg_target])
    
    if all_targets.sum() == 0:
        return 0.0
    
    return average_precision_score(all_targets, all_scores)


def compute_gradcam_for_embedding(model, image, text_emb_1x, layer_index: int = 8):
    """
    Compute a GradCAM heatmap (normalized to [0, 1]) for a single text embedding.
    Returns: 2D tensor [H, W] on CPU.
    """
    # Ensure we use a valid layer for this wrapped model
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap

# --- Sparse Encoding Logic Copies ---
def omp_sparse_residual(
    x_1x: torch.Tensor,
    D: torch.Tensor,
    max_atoms: int = 8,
    tol: float = 1e-6,
    return_num_selected: bool = False,
):
    """
    Orthogonal Matching Pursuit residual (optionally returns number of selected atoms).
    If max_atoms <= 0 or the dictionary is empty, this is a no-op and just returns x_1x.
    """
    # If user requests zero (or negative) atoms, or dictionary is empty, do nothing.
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

def build_wordlist_neighbors_embedding(tokenizer, model, words, device):
    if words is None or len(words) == 0:
        return None
    tok = tokenizer(words).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb

def wordnet_neighbors_configured(
    keyword: str,
    use_synonyms: bool,
    use_hypernyms: bool,
    use_hyponyms: bool,
    use_siblings: bool,
    use_fallback: bool = True,
    limit_per_relation: int = 8
):
    """
    Configurable WordNet neighbors with optional fallback for multi-word terms.
    Mirrors the logic in imagenet_zero_shot_sparse.py.
    """
    try:
        import nltk  # type: ignore
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception as e:
        print(f"[WordNet] Warning: Failed to load NLTK/WordNet: {e}")
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
                        out.append(name); seen.add(name)
            if use_hypernyms:
                for h in s.hypernyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name); seen.add(name)
            if use_hyponyms:
                for h in s.hyponyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name); seen.add(name)
            if use_siblings:
                for h in s.hypernyms()[:limit_per_relation]:
                    for sib in h.hyponyms()[:limit_per_relation]:
                        for l in sib.lemmas()[:limit_per_relation]:
                            name = l.name().replace('_', ' ').lower()
                            if name != key_low and name not in seen:
                                out.append(name); seen.add(name)
        return out

    # 1. Try full keyword
    out = get_neighbors_for_term(keyword)
    if out:
        return out[: max(1, limit_per_relation * 3)]

    if not use_fallback:
        return []

    # 2. Fallback: suffixes for multi-word terms (e.g. "tiger cub" -> "cub")
    words = keyword.split()
    if len(words) > 1:
        for i in range(1, len(words)):
            sub_term = " ".join(words[i:])
            out = get_neighbors_for_term(sub_term)
            if out:
                return out[: max(1, limit_per_relation * 3)]

    return []

def compute_map_for_embedding(model, image, text_emb_1x):
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits

def main():
    parser = argparse.ArgumentParser(description='Benchmark LeGrad Segmentation')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    # image_size is no longer used when we rely on CLIP's own preprocess,
    # but we keep the arg for compatibility.
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument(
        '--class_index_path',
        type=str,
        default='resources/imagenet_class_index.json',
        help='Path to imagenet_class_index.json (downloaded automatically if missing).'
    )
    # Sparse settings (mirroring sparse_encoding.py / imagenet_zero_shot_sparse.py)
    parser.add_argument(
        '--atoms',
        type=int,
        default=8,
        help='Number of atoms for OMP sparse residual.'
    )
    parser.add_argument(
        '--dict_include_prompts',
        type=int,
        default=1,
        help='Include other class prompts in the dictionary D (0/1).'
    )
    parser.add_argument('--wn_use_synonyms', type=int, default=0, help='WordNet: include synonyms (0/1).')
    parser.add_argument('--wn_use_hypernyms', type=int, default=0, help='WordNet: include hypernyms (0/1).')
    parser.add_argument('--wn_use_hyponyms', type=int, default=0, help='WordNet: include hyponyms (0/1).')
    parser.add_argument('--wn_use_siblings', type=int, default=1, help='WordNet: include co-hyponyms/siblings (0/1).')
    parser.add_argument(
        '--wn_fallback_search',
        type=int,
        default=1,
        help='Enable fallback search for multi-word class names in WordNet (0/1).'
    )
    parser.add_argument(
        '--max_dict_cos_sim',
        type=float,
        default=0.9,
        help='Maximum allowed absolute cosine similarity between target prompt and any atom in D. '
             'Atoms with |cos| >= this are dropped; set >=1.0 or <=0 to disable.'
    )
    parser.add_argument(
        '--sparse_threshold',
        type=float,
        default=0.5,
        help='Binarization threshold for sparse-encoding heatmaps (LeGrad stays fixed at 0.5).'
    )
    parser.add_argument(
        '--optimize_thresholds',
        type=int,
        default=0,
        help='If 1, sweep segmentation thresholds to find per-method best mIoU.'
    )
    parser.add_argument(
        '--opt_threshold_min',
        type=float,
        default=0.05,
        help='Minimum threshold value for optimization sweep.'
    )
    parser.add_argument(
        '--opt_threshold_max',
        type=float,
        default=0.95,
        help='Maximum threshold value for optimization sweep.'
    )
    parser.add_argument(
        '--opt_threshold_steps',
        type=int,
        default=19,
        help='Number of threshold values (linspace) between min and max.'
    )
    parser.add_argument(
        '--vis_first_k',
        type=int,
        default=0,
        help='If >0, save visualization grids (GT / LeGrad / Sparse) for the first K images.'
    )
    parser.add_argument(
        '--vis_output_dir',
        type=str,
        default='outputs/segmentation_vis',
        help='Directory to save visualization grids when --vis_first_k > 0.'
    )
    
    # ------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------
    parser.add_argument(
        '--methods',
        type=str,
        default='original,gradcam,sparse',
        help="Comma-separated methods to run from: original,gradcam,sparse. "
             "Example: --methods gradcam,sparse"
    )
    parser.add_argument(
        '--sparse_target',
        type=str,
        default='legrad',
        choices=['legrad', 'gradcam'],
        help="Which heatmap function to use for the sparse embedding. "
             "'legrad' -> compute_map_for_embedding (LeGrad); "
             "'gradcam' -> compute_gradcam_for_embedding (Sparse-GradCAM)."
    )
    
    args = parser.parse_args()

    # Normalize and validate methods list
    methods = [m.strip().lower() for m in str(getattr(args, "methods", "")).split(",") if m.strip()]
    allowed_methods = {'original', 'gradcam', 'sparse'}
    methods = [m for m in methods if m in allowed_methods]
    if not methods:
        raise ValueError("No valid methods selected. Use --methods with any of: original,gradcam,sparse")
    
    # Load Model (use CLIP's preprocess, then wrap with LePreprocess for higher-res input)
    print(f"Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    # Use default layer_index as in original LeGrad (-2 = second-to-last layer)
    model = LeWrapper(model, layer_index=-2)
    # Match official LeGrad usage: wrap preprocess for e.g. 448x448 input
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)
    
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

    # Load ImageNet wnid -> label mapping
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"[imagenet] Loaded class index with {len(wnid_to_label)} wnids.")
    except Exception as e:
        print(f"[imagenet] Warning: failed to load class index ({e}); falling back to WordNet names only.")
        wnid_to_label = {}

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
    print(f"Processing {limit} images with dynamic aspect-ratio preprocessing...")

    # Create visualization directory if requested
    if args.vis_first_k > 0:
        os.makedirs(args.vis_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Precompute prompts and embeddings for all unique wnids in the seg set
    # so we can optionally include other prompts in the sparse dictionary.
    # ------------------------------------------------------------------
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = ''.join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)
    unique_wnids = sorted(set(wnids_in_seg))

    wnid_to_prompt = {}
    for wnid in unique_wnids:
        class_label = wnid_to_label.get(wnid)
        if class_label is None:
            class_label = get_synset_name(wnid)
        wnid_to_prompt[wnid] = f"a photo of a {class_label}."

    all_prompts = [wnid_to_prompt[w] for w in unique_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}

    print(f"[prompts] Built {len(all_prompts)} unique class prompts for segmentation set.")
    tok_all = tokenizer(all_prompts).to(args.device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tok_all, normalize=True)  # [C, d]

    # Track which classes we've logged debug info for (limit to first 4)
    debugged_classes = set()

    results = {
        'original': {'iou': [], 'acc': [], 'ap': []},   # LeGrad
        'gradcam': {'iou': [], 'acc': [], 'ap': []},    # Grad-CAM
        'sparse': {'iou': [], 'acc': [], 'ap': []},     # Sparse LeGrad (kept for completeness)
    }
    # Keep only requested methods
    results = {k: v for k, v in results.items() if k in methods}

    # Optional threshold-sweep structures (for mIoU / pixel-acc optimization)
    optimize_thresholds = bool(getattr(args, "optimize_thresholds", 0))
    if optimize_thresholds:
        thr_min = float(getattr(args, "opt_threshold_min", 0.05))
        thr_max = float(getattr(args, "opt_threshold_max", 0.95))
        thr_steps = int(getattr(args, "opt_threshold_steps", 19))
        if thr_steps < 2:
            thr_steps = 2
        opt_thresholds = np.linspace(thr_min, thr_max, thr_steps).tolist()
        opt_results = {method: {thr: {'iou': [], 'acc': []} for thr in opt_thresholds} for method in methods}
        print(f"[threshold-opt] Enabled sweep over {len(opt_thresholds)} thresholds "
              f"from {thr_min:.3f} to {thr_max:.3f}.")
    else:
        opt_thresholds = []
        opt_results = {}

    for idx in tqdm(range(limit)):
        try:
            # Load Image
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            
            # Use CLIP's official validation preprocess (no custom resizing here)
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            H_feat, W_feat = img_t.shape[-2:]
            
            # Load GT
            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0,0]
                real_gt = np.array(f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)

            # ------------------------------------------------------------------
            # IMPORTANT: match reference ImageNet-Seg processing
            # Resize GT mask to (image_size, image_size) using nearest-neighbor,
            # just like clip_text_span/utils/imagenet_segmentation.py does via
            # target_transform = Resize((image_size, image_size), Image.NEAREST)
            # ------------------------------------------------------------------
            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            target_resize = transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            gt_pil = target_resize(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.uint8)
            H_gt, W_gt = gt_mask.shape
            
            # Get Class Name & prompt / embedding from precomputed tables
            target_ref = targets_refs[idx, 0]
            target_data = np.array(f[target_ref])
            wnid = ''.join([chr(c) for c in target_data.flatten()])

            # Prefer official ImageNet label if available, otherwise fall back to WordNet
            class_label = wnid_to_label.get(wnid)
            if class_label is None:
                class_label = get_synset_name(wnid)
            class_name = class_label

            prompt = wnid_to_prompt[wnid]

            # Precomputed embedding for this class
            cls_idx = wnid_to_idx[wnid]
            original_1x = all_text_embs[cls_idx:cls_idx + 1]  # [1, d]
            
            # --- ORIGINAL (LeGrad) ---
            if 'original' in methods:
                heatmap_orig = compute_map_for_embedding(model, img_t, original_1x)  # [H_feat, W_feat]

            # --- GRAD-CAM ---
            if 'gradcam' in methods:
                heatmap_gradcam = compute_gradcam_for_embedding(model, img_t, original_1x)  # [H_feat, W_feat]

            # Resize to original size
            # Note: F.interpolate expects [B, C, H, W]
            heatmap_orig_resized = None
            heatmap_gradcam_resized = None
            if 'original' in methods:
                heatmap_orig_resized = F.interpolate(
                    heatmap_orig.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
            if 'gradcam' in methods:
                heatmap_gradcam_resized = F.interpolate(
                    heatmap_gradcam.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()

            # --- SPARSE: build dictionary from other prompts + WordNet neighbors ---
            parts = []
            d_labels = []
            prompt_atoms = 0
            neighbor_atoms = 0

            # 1) Other class prompts in the segmentation set (if enabled)
            if bool(args.dict_include_prompts) and len(unique_wnids) > 1:
                if cls_idx > 0:
                    other = all_text_embs[:cls_idx]
                    parts.append(other)
                    prompt_atoms += other.shape[0]
                    d_labels.extend([f"class_{i}" for i in range(cls_idx)])
                if cls_idx + 1 < len(unique_wnids):
                    other = all_text_embs[cls_idx + 1:]
                    parts.append(other)
                    prompt_atoms += other.shape[0]
                    d_labels.extend([f"class_{i}" for i in range(cls_idx + 1, len(unique_wnids))])

            # 2) WordNet neighbors of the class label
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
                    neighbor_atoms += n_emb.shape[0]
                    d_labels.extend(neighbor_prompts)

            dict_raw_size = prompt_atoms + neighbor_atoms

            if len(parts) > 0:
                D = torch.cat(parts, dim=0)
                D = F.normalize(D, dim=-1)
                sim = (D @ original_1x.t()).squeeze(-1).abs()

                # Filter by max cosine similarity if requested
                dropped = 0
                if args.max_dict_cos_sim is not None and 0.0 < float(args.max_dict_cos_sim) < 1.0:
                    keep = sim < float(args.max_dict_cos_sim)
                    dropped = int((~keep).sum().item())
                    D = D[keep]
            else:
                D = original_1x.new_zeros((0, original_1x.shape[-1]))
                dict_raw_size = 0
                dropped = 0

            sparse_1x, num_selected = omp_sparse_residual(
                original_1x, D, max_atoms=args.atoms, return_num_selected=True
            )

            # Debug logging for first few classes
            if cls_idx not in debugged_classes and len(debugged_classes) < 4:
                debugged_classes.add(cls_idx)
                print(f"\n[sparse-debug] class_idx={cls_idx}, wnid={wnid}")
                print(f"  prompt: '{prompt}'")
                print(f"  dict_raw_size={dict_raw_size} (prompts={prompt_atoms}, neighbors={neighbor_atoms})")
                if dict_raw_size > 0:
                    print(f"  max_dict_cos_sim={args.max_dict_cos_sim}, dropped={dropped}")
                    print(f"  dict_final_size={D.shape[0]}")
                    print(f"  atoms_selected={num_selected}")
                else:
                    print("  Dictionary empty; sparse residual falls back to original embedding.")

            if args.sparse_target == 'gradcam':
                heatmap_sparse = compute_gradcam_for_embedding(model, img_t, sparse_1x)
            else:
                heatmap_sparse = compute_map_for_embedding(model, img_t, sparse_1x)
            heatmap_sparse_resized = F.interpolate(
                heatmap_sparse.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            # --- METRICS ---
            # Original LeGrad: fixed threshold 0.5 on normalized heatmap (as in paper)
            if 'original' in methods and heatmap_orig_resized is not None:
                iou_o, acc_o = compute_iou_acc(heatmap_orig_resized, gt_mask, threshold=0.5)
                ap_o = compute_map_score(heatmap_orig_resized, gt_mask)
                
                results['original']['iou'].append(iou_o)
                results['original']['acc'].append(acc_o)
                results['original']['ap'].append(ap_o)

            # Grad-CAM: same fixed threshold 0.5 for fair comparison
            if 'gradcam' in methods and heatmap_gradcam_resized is not None:
                iou_g, acc_g = compute_iou_acc(heatmap_gradcam_resized, gt_mask, threshold=0.5)
                ap_g = compute_map_score(heatmap_gradcam_resized, gt_mask)

                results['gradcam']['iou'].append(iou_g)
                results['gradcam']['acc'].append(acc_g)
                results['gradcam']['ap'].append(ap_g)

            # Sparse encoding: user-controllable fixed threshold (default 0.5)
            if 'sparse' in methods:
                iou_s, acc_s = compute_iou_acc(heatmap_sparse_resized, gt_mask, threshold=args.sparse_threshold)
                ap_s = compute_map_score(heatmap_sparse_resized, gt_mask)
                
                results['sparse']['iou'].append(iou_s)
                results['sparse']['acc'].append(acc_s)
                results['sparse']['ap'].append(ap_s)

            # --- THRESHOLD SWEEP (OPTIONAL) ---
            if optimize_thresholds and opt_thresholds:
                for thr in opt_thresholds:
                    thr_val = float(thr)
                    if 'original' in methods and heatmap_orig_resized is not None:
                        iou_o_t, acc_o_t = compute_iou_acc(heatmap_orig_resized, gt_mask, threshold=thr_val)
                        opt_results['original'][thr]['iou'].append(iou_o_t)
                        opt_results['original'][thr]['acc'].append(acc_o_t)

                    if 'gradcam' in methods and heatmap_gradcam_resized is not None:
                        iou_g_t, acc_g_t = compute_iou_acc(heatmap_gradcam_resized, gt_mask, threshold=thr_val)
                        opt_results['gradcam'][thr]['iou'].append(iou_g_t)
                        opt_results['gradcam'][thr]['acc'].append(acc_g_t)

                    if 'sparse' in methods:
                        iou_s_t, acc_s_t = compute_iou_acc(heatmap_sparse_resized, gt_mask, threshold=thr_val)
                        opt_results['sparse'][thr]['iou'].append(iou_s_t)
                        opt_results['sparse'][thr]['acc'].append(acc_s_t)

            # --- OPTIONAL VISUALIZATION GRID ---
            if idx < args.vis_first_k:
                # Resize original image to GT size
                vis_img = base_img.resize((W_gt, H_gt))
                # Binary masks
                gt_bin = gt_mask.astype(np.uint8)
                legrad_bin = (heatmap_orig_resized > 0.5).astype(np.uint8)
                sparse_bin = (heatmap_sparse_resized > args.sparse_threshold).astype(np.uint8)

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(vis_img)
                axes[0].set_title('Image')
                axes[0].axis('off')
                axes[1].imshow(gt_bin, cmap='gray')
                arches1 = axes[1]
                arches1.set_title('GT')
                arches1.axis('off')
                axes[2].imshow(legrad_bin, cmap='gray')
                axes[2].set_title('LeGrad (0.5)')
                axes[2].axis('off')
                axes[3].imshow(sparse_bin, cmap='gray')
                axes[3].set_title(f'Sparse ({args.sparse_threshold:.2f})')
                axes[3].axis('off')
                plt.tight_layout()

                out_name = f"seg_vis_{idx:04d}.png"
                out_path = os.path.join(args.vis_output_dir, out_name)
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[vis] Saved {out_path}")
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue

    print("\n--- Results ---")
    for method in methods:
        miou = np.mean(results[method]['iou']) * 100
        macc = np.mean(results[method]['acc']) * 100
        map_score = np.mean(results[method]['ap']) * 100
        print(f"{method.capitalize()}: PixelAcc={macc:.2f}, mIoU={miou:.2f}, mAP={map_score:.2f}")

    # Report best thresholds per method if optimization was enabled
    if optimize_thresholds and opt_thresholds:
        print("\n--- Threshold optimization (best mIoU per method) ---")
        for method in methods:
            best_thr = None
            best_miou = -1.0
            best_macc = 0.0
            for thr in opt_thresholds:
                ious = opt_results[method][thr]['iou']
                accs = opt_results[method][thr]['acc']
                if not ious:
                    continue
                miou = np.mean(ious) * 100
                macc = np.mean(accs) * 100
                if miou > best_miou:
                    best_miou = miou
                    best_macc = macc
                    best_thr = thr
            if best_thr is not None:
                print(
                    f"{method.capitalize()}: best_threshold={float(best_thr):.3f}, "
                    f"PixelAcc={best_macc:.2f}, mIoU={best_miou:.2f}"
                )

        # Note: mAP is threshold-independent (it uses the continuous heatmaps),
        # so there is no separate "best threshold" for mAP. We still summarize
        # the achieved mAP here next to the optimized mIoU thresholds.
        print("\n--- mAP (threshold-independent) per method ---")
        for method in methods:
            map_score = np.mean(results[method]['ap']) * 100
            print(f"{method.capitalize()}: mAP={map_score:.2f}")

if __name__ == '__main__':
    main()
