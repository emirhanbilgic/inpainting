#!/usr/bin/env python3
"""
Compute positive/negative baselines on MS COCO 2017 two-object images.

Each image has exactly two unique objects.  One object is randomly chosen and
its segmentation mask is kept.

  * Positive baseline:  prompt = chosen object  →  evaluate against chosen mask
  * Negative baseline:  prompt = OTHER object   →  evaluate against chosen mask

Usage:
    python scripts/compute_coco_baseline.py --limit 50
    python scripts/compute_coco_baseline.py --use_gradcam
    python scripts/compute_coco_baseline.py --use_siglip
"""

import sys
import os

# Enable MPS CPU fallback for unsupported ops (e.g. bicubic interpolation)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
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

from legrad import LeWrapper, LePreprocess
import open_clip

# Reuse heatmap computation functions from the existing baseline script
from compute_positive_negative_baseline import (
    compute_legrad_heatmap,
    compute_gradcam_heatmap,
    compute_lrp_heatmap,
    compute_chefercam,
    compute_transformer_attribution,
    batch_intersection_union,
)

from benchmark_segmentation_v2 import (
    get_ap_scores,
    batch_pix_accuracy,
)

try:
    from daam_segmentation import DAAMSegmenter
except ImportError:
    DAAMSegmenter = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coco_metadata(json_path):
    """Load the selected_images_2_unique.json metadata."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_mask(mask_path, target_size):
    """Load a binary mask PNG, resize, and return as uint8 numpy (0/1)."""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    arr = np.array(mask)
    return (arr > 128).astype(np.uint8)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CocoBaselineEvaluator:
    """Evaluate heatmap methods on the COCO two‑object dataset."""

    def __init__(
        self,
        model,
        tokenizer,
        preprocess,
        coco_dir,
        metadata,
        device,
        image_size=448,
        limit=0,
        method='legrad',
        gradcam_layer=8,
        chefercam_method='transformer_attribution',
        transformer_attribution_start_layer=1,
        threshold_mode='mean',
        fixed_threshold=0.5,
        use_daam=False,
        daam_model_id='Manojb/stable-diffusion-2-base',
        lrp_start_layer=1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.method = method              # 'legrad', 'gradcam', 'chefercam', 'lrp', 'daam'
        self.gradcam_layer = gradcam_layer
        self.chefercam_method = chefercam_method
        self.transformer_attribution_start_layer = transformer_attribution_start_layer
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.lrp_start_layer = lrp_start_layer

        self.coco_dir = coco_dir
        self.image_dir = os.path.join(coco_dir, 'val2017')
        self.mask_dir = os.path.join(coco_dir, 'val2017_2_objects_masks')

        # Build ordered list of entries
        self.entries = list(metadata.values())
        if limit > 0:
            self.entries = self.entries[:limit]

        # DAAM
        self.daam_segmenter = None
        if use_daam:
            if DAAMSegmenter is None:
                raise ImportError("DAAMSegmenter not available.")
            print(f"[coco] Initializing DAAMSegmenter ({daam_model_id})...")
            self.daam_segmenter = DAAMSegmenter(model_id=daam_model_id, device=device)

        # Precompute text embeddings for all unique object names
        all_objects = set()
        for e in self.entries:
            all_objects.update(e['objects'])
        self.unique_objects = sorted(all_objects)
        self.obj_to_idx = {o: i for i, o in enumerate(self.unique_objects)}

        prompts = [f"a photo of a {obj}." for obj in self.unique_objects]
        tok = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            self.all_text_embs = self.model.encode_text(tok, normalize=True)

        print(f"[coco] {len(self.entries)} images, {len(self.unique_objects)} unique objects")

    # ----- heatmap dispatch -----
    def _compute_heatmap(self, img_t, text_emb_1x, base_img=None, prompt_text=None):
        """Dispatch to the selected method."""
        if self.method == 'daam':
            return self.daam_segmenter.predict(base_img, prompt_text, size=512)
        elif self.method == 'lrp':
            return compute_lrp_heatmap(self.model, img_t, text_emb_1x)
        elif self.method == 'chefercam':
            if self.chefercam_method == 'transformer_attribution':
                return compute_transformer_attribution(
                    self.model, img_t, text_emb_1x,
                    start_layer=self.transformer_attribution_start_layer,
                )
            return compute_chefercam(self.model, img_t, text_emb_1x)
        elif self.method == 'gradcam':
            return compute_gradcam_heatmap(self.model, img_t, text_emb_1x, layer_index=self.gradcam_layer)
        else:
            return compute_legrad_heatmap(self.model, img_t, text_emb_1x)

    # ----- metric computation for one (heatmap, mask) pair -----
    def _metrics_from_heatmap(self, heatmap, gt_mask, H_gt, W_gt):
        """Resize heatmap → threshold → compute IOU / AP / accuracy."""
        H_hm, W_hm = heatmap.shape[-2], heatmap.shape[-1]
        heatmap_resized = F.interpolate(
            heatmap.view(1, 1, H_hm, W_hm),
            size=(H_gt, W_gt),
            mode='bilinear',
            align_corners=False,
        ).squeeze()

        # Normalise to [0,1]
        if self.method == 'legrad':
            heatmap_norm = heatmap_resized  # already [0,1]
        else:
            heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
                heatmap_resized.max() - heatmap_resized.min() + 1e-8
            )

        # Threshold
        if self.method == 'legrad':
            thr = 0.5
        elif self.threshold_mode == 'mean':
            thr = heatmap_norm.mean().item()
        else:
            thr = self.fixed_threshold

        Res_1 = (heatmap_norm > thr).float()
        Res_0 = (heatmap_norm <= thr).float()
        output_tensor = torch.stack([Res_0, Res_1], dim=0)       # [2,H,W]
        output_AP     = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)

        gt_tensor = torch.from_numpy(gt_mask).long()

        inter, union = batch_intersection_union(output_tensor, gt_tensor, nclass=2)
        ap_list = get_ap_scores(output_AP, gt_tensor)
        ap = ap_list[0] if ap_list else 0.0

        correct_pixels, labeled_pixels = batch_pix_accuracy(output_tensor, gt_tensor)

        heatmap_np = heatmap_norm.detach().cpu().numpy()
        return inter, union, correct_pixels, labeled_pixels, ap, heatmap_np

    # ----- main loop -----
    def evaluate(self, show_progress=True):
        """Run positive & negative evaluation over all entries."""
        # accumulators
        def _empty():
            return {
                'inter': np.zeros(2), 'union': np.zeros(2),
                'pixel_correct': 0, 'pixel_label': 0,
                'ap': [], 'auroc': [],
            }
        pos = _empty()
        neg = _empty()

        iterator = self.entries
        if show_progress:
            iterator = tqdm(iterator, desc="COCO baseline")

        for entry in iterator:
            try:
                file_name = entry['file_name']
                objects = entry['objects']
                chosen = entry['chosen']
                other = [o for o in objects if o != chosen][0]

                # ---- load image ----
                img_path = os.path.join(self.image_dir, file_name)
                if not os.path.exists(img_path):
                    continue
                base_img = Image.open(img_path).convert('RGB')
                img_t = self.preprocess(base_img).unsqueeze(0).to(self.device)

                # ---- load mask ----
                image_id = file_name.replace('.jpg', '')
                mask_name = f"{image_id}_{chosen.replace(' ', '_')}.png"
                mask_path = os.path.join(self.mask_dir, mask_name)
                if not os.path.exists(mask_path):
                    continue
                gt_mask = load_mask(mask_path, (self.image_size, self.image_size))
                H_gt, W_gt = gt_mask.shape

                # ---- POSITIVE: prompt = chosen ----
                pos_idx = self.obj_to_idx[chosen]
                pos_emb = self.all_text_embs[pos_idx:pos_idx + 1]
                pos_prompt = f"a photo of a {chosen}."
                heatmap_pos = self._compute_heatmap(img_t, pos_emb, base_img=base_img, prompt_text=pos_prompt)
                i_p, u_p, c_p, l_p, ap_p, hm_p = self._metrics_from_heatmap(heatmap_pos, gt_mask, H_gt, W_gt)
                pos['inter'] += i_p
                pos['union'] += u_p
                pos['pixel_correct'] += c_p
                pos['pixel_label'] += l_p
                pos['ap'].append(ap_p)

                # ---- NEGATIVE: prompt = other, mask = same chosen mask ----
                neg_idx = self.obj_to_idx[other]
                neg_emb = self.all_text_embs[neg_idx:neg_idx + 1]
                neg_prompt = f"a photo of a {other}."
                heatmap_neg = self._compute_heatmap(img_t, neg_emb, base_img=base_img, prompt_text=neg_prompt)
                i_n, u_n, c_n, l_n, ap_n, hm_n = self._metrics_from_heatmap(heatmap_neg, gt_mask, H_gt, W_gt)
                neg['inter'] += i_n
                neg['union'] += u_n
                neg['pixel_correct'] += c_n
                neg['pixel_label'] += l_n
                neg['ap'].append(ap_n)

                # ---- Paired AUROC ----
                gt_binary = (gt_mask > 0).astype(int).flatten()
                gt_zeros  = np.zeros_like(gt_binary)
                paired_gt   = np.concatenate([gt_binary, gt_zeros])
                paired_pred = np.concatenate([hm_p.flatten(), hm_n.flatten()])
                if len(np.unique(paired_gt)) > 1:
                    auroc = roc_auc_score(paired_gt, paired_pred)
                    pos['auroc'].append(auroc)
                    neg['auroc'].append(auroc)

            except Exception as e:
                print(f"[warn] {file_name}: {e}")
                continue

        return self._aggregate(pos, neg)

    # ----- aggregate -----
    @staticmethod
    def _aggregate(pos, neg):
        def _metrics(d):
            iou = d['inter'].astype(np.float64) / (d['union'].astype(np.float64) + 1e-10)
            miou = 100.0 * iou.mean()
            acc = 100.0 * d['pixel_correct'] / (d['pixel_label'] + 1e-10)
            map_score = np.mean(d['ap']) * 100 if d['ap'] else 0.0
            auroc_list = [a for a in d['auroc'] if not np.isnan(a)]
            auroc = np.mean(auroc_list) * 100 if auroc_list else 0.0
            return {
                'miou': float(miou),
                'acc': float(acc),
                'map': float(map_score),
                'auroc': float(auroc),
                'n_samples': len(d['ap']),
            }
        return {'correct': _metrics(pos), 'wrong': _metrics(neg)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='COCO positive/negative baseline')

    # paths
    parser.add_argument('--coco_dir', type=str,
                        default=os.path.join(project_root, 'ms_coco_2017'),
                        help='Path to ms_coco_2017 directory')
    parser.add_argument('--metadata_json', type=str, default=None,
                        help='Path to selected_images_2_unique.json (default: <coco_dir>/selected_images_2_unique.json)')

    # model
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--use_siglip', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--image_size', type=int, default=448)

    # method
    parser.add_argument('--use_gradcam', action='store_true')
    parser.add_argument('--gradcam_layer', type=int, default=8)
    parser.add_argument('--use_chefercam', action='store_true')
    parser.add_argument('--chefercam_method', type=str, default='transformer_attribution',
                        choices=['attn_gradcam', 'transformer_attribution'])
    parser.add_argument('--transformer_attribution_start_layer', type=int, default=1)
    parser.add_argument('--use_lrp', action='store_true')
    parser.add_argument('--lrp_start_layer', type=int, default=1)
    parser.add_argument('--use_daam', action='store_true')
    parser.add_argument('--daam_model_id', type=str, default='Manojb/stable-diffusion-2-base')

    # thresholding
    parser.add_argument('--threshold_mode', type=str, default='mean', choices=['mean', 'fixed'])
    parser.add_argument('--fixed_threshold', type=float, default=0.5)

    # control
    parser.add_argument('--limit', type=int, default=0, help='Max images (0 = all)')
    parser.add_argument('--output_json', type=str, default='coco_baseline.json')

    args = parser.parse_args()

    # Resolve metadata path
    if args.metadata_json is None:
        args.metadata_json = os.path.join(args.coco_dir, 'selected_images_2_unique.json')

    # Model defaults
    if args.use_siglip:
        args.model_name = args.model_name or 'ViT-B-16-SigLIP'
        args.pretrained = args.pretrained or 'webli'
        model_type = 'SigLIP'
    else:
        args.model_name = args.model_name or 'ViT-B-16'
        args.pretrained = args.pretrained or 'laion2b_s34b_b88k'
        model_type = 'CLIP'

    # Determine method string
    if args.use_daam:
        method = 'daam'
    elif args.use_lrp:
        method = 'lrp'
    elif args.use_chefercam:
        method = 'chefercam'
    elif args.use_gradcam:
        method = 'gradcam'
    else:
        method = 'legrad'

    # Load model
    print(f"Loading {model_type}: {args.model_name} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # Load metadata
    metadata = load_coco_metadata(args.metadata_json)
    print(f"Loaded {len(metadata)} entries from {args.metadata_json}")

    # Create evaluator
    evaluator = CocoBaselineEvaluator(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        coco_dir=args.coco_dir,
        metadata=metadata,
        device=args.device,
        image_size=args.image_size,
        limit=args.limit,
        method=method,
        gradcam_layer=args.gradcam_layer,
        chefercam_method=args.chefercam_method,
        transformer_attribution_start_layer=args.transformer_attribution_start_layer,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        use_daam=args.use_daam,
        daam_model_id=args.daam_model_id,
        lrp_start_layer=args.lrp_start_layer,
    )

    # Run
    print(f"\n{'='*60}")
    print(f"COCO Positive/Negative Baseline  –  {method.upper()}  ({model_type})")
    print(f"{'='*60}\n")

    results = evaluator.evaluate()

    # Print
    for label, key in [('POSITIVE (correct prompt)', 'correct'),
                        ('NEGATIVE (wrong prompt)',   'wrong')]:
        r = results[key]
        print(f"\n=== {label} ===")
        print(f"  mIoU:     {r['miou']:.2f}")
        print(f"  Accuracy: {r['acc']:.2f}")
        print(f"  mAP:      {r['map']:.2f}")
        print(f"  AUROC:    {r['auroc']:.2f}")
        print(f"  Samples:  {r['n_samples']}")

    composite = results['correct']['miou'] - 0.5 * results['wrong']['miou']
    print(f"\n  Composite (correct − 0.5×wrong): {composite:.2f}")

    # Save
    output = {
        'correct': results['correct'],
        'wrong': results['wrong'],
        'composite': float(composite),
        'settings': {
            'method': method,
            'model_name': args.model_name,
            'pretrained': args.pretrained,
            'model_type': model_type,
            'image_size': args.image_size,
            'threshold_mode': args.threshold_mode,
            'fixed_threshold': args.fixed_threshold,
            'limit': args.limit if args.limit > 0 else 'all',
        },
    }
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[output] Saved to {args.output_json}")


if __name__ == '__main__':
    main()
