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
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
    pred_mask = (heatmap > threshold).astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    iou = intersection / (union + 1e-6)
    
    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size
    acc = correct / total
    
    return iou, acc

def compute_map_score(heatmap, gt_mask):
    y_true = gt_mask.flatten().astype(int)
    y_scores = heatmap.flatten()
    if y_true.sum() == 0:
        return 0.0
    return average_precision_score(y_true, y_scores)

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
    """
    if D is None or D.numel() == 0:
        r_norm = F.normalize(x_1x, dim=-1)
        return (r_norm, 0) if return_num_selected else r_norm
    x = x_1x.clone()
    K = D.shape[0]
    max_atoms = int(max(1, min(max_atoms, K)))
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
    
    args = parser.parse_args()
    
    # Load Model (use CLIP's preprocess, then wrap with LePreprocess for higher-res input)
    print(f"Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    # Use default layer_index as in original LeGrad (typically -2)
    model = LeWrapper(model)
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
        'original': {'iou': [], 'acc': [], 'ap': []},
        'sparse': {'iou': [], 'acc': [], 'ap': []}
    }

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
                gt_mask = np.zeros((base_img.height, base_img.width))
            
            H_orig, W_orig = gt_mask.shape
            
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
            
            # --- ORIGINAL ---
            heatmap_orig = compute_map_for_embedding(model, img_t, original_1x)  # [H_feat, W_feat]

            # Resize to original size
            # Note: F.interpolate expects [B, C, H, W]
            heatmap_orig_resized = F.interpolate(
                heatmap_orig.view(1, 1, H_feat, W_feat),
                size=(H_orig, W_orig),
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

            heatmap_sparse = compute_map_for_embedding(model, img_t, sparse_1x)
            heatmap_sparse_resized = F.interpolate(
                heatmap_sparse.view(1, 1, H_feat, W_feat),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            # --- METRICS ---
            # Original LeGrad: fixed threshold 0.5
            iou_o, acc_o = compute_iou_acc(heatmap_orig_resized, gt_mask, threshold=0.5)
            ap_o = compute_map_score(heatmap_orig_resized, gt_mask)
            
            results['original']['iou'].append(iou_o)
            results['original']['acc'].append(acc_o)
            results['original']['ap'].append(ap_o)

            # Sparse encoding: user-controllable threshold
            iou_s, acc_s = compute_iou_acc(heatmap_sparse_resized, gt_mask, threshold=args.sparse_threshold)
            ap_s = compute_map_score(heatmap_sparse_resized, gt_mask)
            
            results['sparse']['iou'].append(iou_s)
            results['sparse']['acc'].append(acc_s)
            results['sparse']['ap'].append(ap_s)
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue

    print("\n--- Results ---")
    for method in ['original', 'sparse']:
        miou = np.mean(results[method]['iou']) * 100
        macc = np.mean(results[method]['acc']) * 100
        map_score = np.mean(results[method]['ap']) * 100
        print(f"{method.capitalize()}: mIoU={miou:.2f}, PixelAcc={macc:.2f}, mAP={map_score:.2f}")

if __name__ == '__main__':
    main()
