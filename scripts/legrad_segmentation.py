import sys
import os
import argparse
import json
import time
from typing import Dict, Any

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import requests

# Add project root to path (ensure local legrad overrides any installed package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    import nltk  # type: ignore
    from nltk.corpus import wordnet as wn  # type: ignore
except ImportError:
    wn = None  # type: ignore

from legrad import LeWrapper, LePreprocess
import open_clip

# -----------------------------------------------------------------------------
# Constants & helpers shared with the main segmentation benchmark, but this
# script is **LeGrad-only** (no Grad-CAM, no sparse residuals).
# -----------------------------------------------------------------------------
IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def download_imagenet_class_index(dst_path: str) -> Dict[str, Any]:
    """Download the standard ImageNet class index JSON.

    Returns dict: idx_str -> [wnid, human_label].
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(dst_path, "w") as f:
        json.dump(data, f)
    return data


def load_imagenet_class_index(path: str) -> Dict[str, Any]:
    """Load (or download) the ImageNet class index JSON."""
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    print(
        f"[imagenet] class index not found at {path}, downloading from {IMAGENET_CLASS_INDEX_URL} ..."
    )
    return download_imagenet_class_index(path)


def build_wnid_to_label_map(class_index: Dict[str, Any]) -> Dict[str, str]:
    """Convert imagenet_class_index mapping into wnid -> primary label (lowercase).

    Each entry: idx_str -> [wnid, "label1, label2, ..."].
    """
    wnid_to_label: Dict[str, str] = {}
    for _, (wnid, label_str) in class_index.items():
        primary = label_str.split(",")[0].replace("_", " ").strip()
        wnid_to_label[wnid] = primary.lower()
    return wnid_to_label


def get_synset_name(wnid: str) -> str:
    """Fallback: derive a name from WordNet if JSON mapping is unavailable."""
    if wn is None:
        return wnid
    try:
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset("n", offset)
        name = synset.lemmas()[0].name().replace("_", " ")
        return name
    except Exception:
        return wnid


def compute_iou_acc(heatmap: np.ndarray, gt_mask: np.ndarray, threshold: float):
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


def compute_map_score(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
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
    # fg_target[i] = 1 if gt_mask[i] == 1 (foreground)
    # bg_target[i] = 1 if gt_mask[i] == 0 (background)
    gt_flat = gt_mask.flatten()
    fg_target = (gt_flat == 1).astype(int)
    bg_target = (gt_flat == 0).astype(int)
    
    # Combine both classes into single arrays (as in reference get_ap_scores)
    # This concatenates predictions and targets for both classes
    all_scores = np.concatenate([bg_prob, fg_prob])
    all_targets = np.concatenate([bg_target, fg_target])
    
    if all_targets.sum() == 0:
        return 0.0
    
    return average_precision_score(all_targets, all_scores)


def compute_map_for_embedding(model, image: torch.Tensor, text_emb_1x: torch.Tensor) -> torch.Tensor:
    """Run LeGrad and return normalized heatmap [H, W] on CPU in [0,1]."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


# -----------------------------------------------------------------------------
# Main: LeGrad-only segmentation benchmark
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LeGrad-only ImageNet-Segmentation benchmark")

    parser.add_argument("--mat_file", type=str, default="scripts/data/gtsegs_ijcv.mat")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 for all)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    # image_size is kept for compatibility with the main benchmark.
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument(
        "--class_index_path",
        type=str,
        default="resources/imagenet_class_index.json",
        help="Path to imagenet_class_index.json (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for LeGrad heatmaps (default 0.5, as in the paper).",
    )
    parser.add_argument(
        "--vis_first_k",
        type=int,
        default=0,
        help="If >0, save visualization grids (GT / LeGrad) for the first K images.",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="outputs/legrad_segmentation_vis",
        help="Directory to save visualization grids when --vis_first_k > 0.",
    )

    args = parser.parse_args()

    # Load Model (use CLIP's preprocess, then wrap with LePreprocess for higher-res input)
    print(f"[LeGrad] Loading model {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # Use default layer_index as in original LeGrad (-2 = second-to-last layer)
    model = LeWrapper(model, layer_index=-2)
    # Match official LeGrad usage: wrap preprocess for e.g. 448x448 input
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # Ensure WordNet is available for fallback names, but don't crash if it fails
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass

    # Load ImageNet wnid -> label mapping
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"[imagenet] Loaded class index with {len(wnid_to_label)} wnids.")
    except Exception as e:  # pragma: no cover - network / file errors
        print(
            f"[imagenet] Warning: failed to load class index ({e}); "
            "falling back to WordNet names only."
        )
        wnid_to_label = {}

    print(f"[LeGrad] Opening dataset {args.mat_file}...")
    try:
        f = h5py.File(args.mat_file, "r")
        imgs_refs = f["value/img"]
        gts_refs = f["value/gt"]
        targets_refs = f["value/target"]
        num_images = imgs_refs.shape[0]
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    limit = args.limit if args.limit > 0 else num_images
    limit = min(limit, num_images)
    print(f"[LeGrad] Processing {limit} images...")

    # Create visualization directory if requested
    if args.vis_first_k > 0:
        os.makedirs(args.vis_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Precompute prompts and embeddings for all unique wnids in the seg set
    # ------------------------------------------------------------------
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = "".join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)
    unique_wnids = sorted(set(wnids_in_seg))

    wnid_to_prompt: Dict[str, str] = {}
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

    results = {"legrad": {"iou": [], "acc": [], "ap": []}}

    start_time = time.time()

    for idx in tqdm(range(limit)):
        try:
            # Load Image
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)

            # Use CLIP's official validation preprocess
            img_t = preprocess(base_img).unsqueeze(0).to(args.device)
            H_feat, W_feat = img_t.shape[-2:]

            # Load GT
            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == "object":
                real_gt_ref = gt_wrapper[0, 0]
                real_gt = np.array(f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)

            # Match reference ImageNet-Seg processing: resize GT mask to
            # (image_size, image_size) using nearest-neighbor.
            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            target_resize = transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            gt_pil = target_resize(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.uint8)
            H_gt, W_gt = gt_mask.shape

            # Get class wnid & corresponding prompt / embedding
            target_ref = targets_refs[idx, 0]
            target_data = np.array(f[target_ref])
            wnid = "".join([chr(c) for c in target_data.flatten()])

            # Prefer official ImageNet label if available, otherwise fall back to WordNet
            class_label = wnid_to_label.get(wnid)
            if class_label is None:
                class_label = get_synset_name(wnid)

            prompt = wnid_to_prompt[wnid]

            # Precomputed embedding for this class
            cls_idx = wnid_to_idx[wnid]
            text_emb_1x = all_text_embs[cls_idx : cls_idx + 1]  # [1, d]

            # --- LeGrad ---
            heatmap_legrad = compute_map_for_embedding(model, img_t, text_emb_1x)  # [H_feat, W_feat]

            # Resize to GT size for evaluation
            heatmap_legrad_resized = F.interpolate(
                heatmap_legrad.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            # --- METRICS (LeGrad-only) ---
            iou_l, acc_l = compute_iou_acc(
                heatmap_legrad_resized, gt_mask, threshold=args.threshold
            )
            ap_l = compute_map_score(heatmap_legrad_resized, gt_mask)

            results["legrad"]["iou"].append(iou_l)
            results["legrad"]["acc"].append(acc_l)
            results["legrad"]["ap"].append(ap_l)

            # --- OPTIONAL VISUALIZATION GRID ---
            if idx < args.vis_first_k:
                vis_img = base_img.resize((W_gt, H_gt))
                gt_bin = gt_mask.astype(np.uint8)
                legrad_bin = (heatmap_legrad_resized > args.threshold).astype(np.uint8)

                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(vis_img)
                axes[0].set_title("Image")
                axes[0].axis("off")
                axes[1].imshow(gt_bin, cmap="gray")
                axes[1].set_title("GT")
                axes[1].axis("off")
                axes[2].imshow(legrad_bin, cmap="gray")
                axes[2].set_title(f"LeGrad ({args.threshold:.2f})")
                axes[2].axis("off")

                plt.tight_layout()
                out_name = f"legrad_seg_{idx:04d}.png"
                out_path = os.path.join(args.vis_output_dir, out_name)
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[vis] Saved {out_path}")

        except Exception as e:  # pragma: no cover - robust to dataset quirks
            print(f"Error processing idx {idx}: {e}")
            continue

    elapsed = time.time() - start_time

    print("\n--- LeGrad-only Segmentation Results ---")
    miou = float(np.mean(results["legrad"]["iou"]) * 100) if results["legrad"]["iou"] else 0.0
    macc = float(np.mean(results["legrad"]["acc"]) * 100) if results["legrad"]["acc"] else 0.0
    map_score = float(np.mean(results["legrad"]["ap"]) * 100) if results["legrad"]["ap"] else 0.0

    print(f"LeGrad: PixelAcc={macc:.2f}, mIoU={miou:.2f}, mAP={map_score:.2f}")
    print(f"Processed {limit} images in {elapsed/60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
