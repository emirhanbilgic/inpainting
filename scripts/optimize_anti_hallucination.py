#!/usr/bin/env python3
"""
Dual-objective hyperparameter optimization for anti-hallucination segmentation.

Goal: Find hyperparameters that:
1. MAXIMIZE segmentation metrics for CORRECT class prompts
2. MINIMIZE segmentation metrics for WRONG class prompts (hallucinations)

This prevents the model from activating on unrelated class prompts,
achieving class-specific segmentation.

Optimization Approaches:
1. Multi-objective Pareto optimization (--multi_objective)
2. Composite score: correct_mIoU - λ × wrong_mIoU (default)

Usage:
    # Composite score optimization (single-objective)
    python scripts/optimize_anti_hallucination.py --n_trials 100 --limit 100

    # Multi-objective Pareto optimization
    python scripts/optimize_anti_hallucination.py --n_trials 100 --multi_objective

    # Adjust hallucination penalty weight
    python scripts/optimize_anti_hallucination.py --n_trials 50 --composite_lambda 1.0
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

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None

from legrad import LeWrapper, LePreprocess
import open_clip

# Import functions from benchmark_segmentation
from benchmark_segmentation import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    compute_iou_acc,
    compute_map_score,
    compute_map_for_embedding,
    omp_sparse_residual,
    wordnet_neighbors_configured,
)


# Define semantic superclasses for distant negative sampling
IMAGENET_SUPERCLASSES = {
    # Animals
    'mammal': list(range(151, 294)),  # dogs, cats, bears, etc.
    'bird': list(range(80, 101)) + list(range(127, 147)),
    'reptile': list(range(33, 69)),
    'amphibian': list(range(25, 33)),
    'fish': list(range(0, 7)) + list(range(389, 398)),
    'invertebrate': list(range(69, 80)) + list(range(107, 127)) + list(range(300, 328)),
    
    # Vehicles & transport
    'vehicle': [403, 404, 405, 407, 408, 436, 468, 511, 573, 575, 609, 627, 654, 656, 670, 671, 717, 751, 757, 779, 802, 803, 817, 829, 864, 866, 867],
    
    # Food
    'food': list(range(924, 970)),
    
    # Household objects
    'furniture': [423, 453, 492, 493, 495, 516, 520, 526, 532, 559, 564, 765, 831, 857],
    
    # Nature/scenes
    'nature': list(range(970, 981)),
    
    # Clothing
    'clothing': [399, 400, 411, 445, 451, 452, 474, 501, 568, 601, 608, 610, 614, 655, 697, 735, 775, 841, 869],
}


def get_superclass(class_idx: int) -> str:
    """Get the superclass for a given ImageNet class index."""
    for superclass, indices in IMAGENET_SUPERCLASSES.items():
        if class_idx in indices:
            return superclass
    return 'other'


def get_distant_class_indices(class_idx: int, all_wnids: list, wnid_to_idx: dict) -> list:
    """Get indices of classes from different superclasses (semantically distant)."""
    current_superclass = get_superclass(class_idx)
    distant_indices = []
    
    for superclass, indices in IMAGENET_SUPERCLASSES.items():
        if superclass != current_superclass:
            # Filter to only include indices that are in our dataset
            for idx in indices:
                # Map ImageNet index to our local index if available
                if idx < len(all_wnids):
                    distant_indices.append(idx)
    
    return distant_indices


class AntiHallucinationObjective:
    """
    Optuna objective for anti-hallucination hyperparameter optimization.
    
    Evaluates both:
    - Correct prompts (maximize): prompt matches actual class
    - Wrong prompts (minimize): prompt is for a different class
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        preprocess,
        dataset_file,
        wnid_to_label,
        device,
        image_size=448,
        limit=0,
        num_negatives=1,
        negative_strategy='random',
        composite_lambda=0.5,
        multi_objective=False,
        seed=42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.num_negatives = num_negatives
        self.negative_strategy = negative_strategy
        self.composite_lambda = composite_lambda
        self.multi_objective = multi_objective
        self.rng = random.Random(seed)
        
        # Load dataset
        self.f = h5py.File(dataset_file, 'r')
        self.imgs_refs = self.f['value/img']
        self.gts_refs = self.f['value/gt']
        self.targets_refs = self.f['value/target']
        self.num_images = self.imgs_refs.shape[0]
        
        self.limit = limit if limit > 0 else self.num_images
        self.limit = min(self.limit, self.num_images)
        
        self.wnid_to_label = wnid_to_label
        
        # Precompute wnids and prompts
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        """Precompute all class embeddings once."""
        print("[optuna] Precomputing class embeddings...")
        
        wnids_in_seg = []
        for idx in range(self.num_images):
            target_ref = self.targets_refs[idx, 0]
            target_data = np.array(self.f[target_ref])
            wnid = ''.join([chr(c) for c in target_data.flatten()])
            wnids_in_seg.append(wnid)
        
        self.wnids_in_seg = wnids_in_seg
        self.unique_wnids = sorted(set(wnids_in_seg))
        
        self.wnid_to_prompt = {}
        self.wnid_to_classname = {}
        for wnid in self.unique_wnids:
            class_label = self.wnid_to_label.get(wnid)
            if class_label is None:
                class_label = get_synset_name(wnid)
            self.wnid_to_classname[wnid] = class_label
            self.wnid_to_prompt[wnid] = f"a photo of a {class_label}."
        
        all_prompts = [self.wnid_to_prompt[w] for w in self.unique_wnids]
        self.wnid_to_idx = {w: i for i, w in enumerate(self.unique_wnids)}
        self.idx_to_wnid = {i: w for w, i in self.wnid_to_idx.items()}
        
        tok_all = self.tokenizer(all_prompts).to(self.device)
        with torch.no_grad():
            self.all_text_embs = self.model.encode_text(tok_all, normalize=True)
        
        print(f"[optuna] Precomputed {len(all_prompts)} class embeddings.")
        print(f"[optuna] Negative strategy: {self.negative_strategy}, num_negatives: {self.num_negatives}")
    
    def _sample_negative_indices(self, correct_idx: int) -> list:
        """Sample negative class indices different from the correct class."""
        all_indices = list(range(len(self.unique_wnids)))
        all_indices.remove(correct_idx)
        
        if self.negative_strategy == 'distant':
            # Try to get semantically distant classes
            distant = get_distant_class_indices(correct_idx, self.unique_wnids, self.wnid_to_idx)
            # Filter to what's available in our dataset
            distant = [d for d in distant if d < len(self.unique_wnids) and d != correct_idx]
            if len(distant) >= self.num_negatives:
                return self.rng.sample(distant, self.num_negatives)
            # Fall back to random if not enough distant classes
            return self.rng.sample(all_indices, min(self.num_negatives, len(all_indices)))
        else:
            # Random sampling
            return self.rng.sample(all_indices, min(self.num_negatives, len(all_indices)))
    
    def evaluate_sparse_config(
        self,
        wn_use_synonyms: bool,
        wn_use_hypernyms: bool,
        wn_use_hyponyms: bool,
        wn_use_siblings: bool,
        dict_include_prompts: bool,
        sparse_threshold: float,
        atoms: int,
        max_dict_cos_sim: float,
        show_progress: bool = False,
    ):
        """
        Evaluate a specific sparse configuration.
        Returns: (correct_miou, wrong_miou, correct_acc, wrong_acc, correct_map, wrong_map,
                  correct_auroc, wrong_auroc, correct_stats, wrong_stats)
        """
        correct_results = {
            'iou': [], 'acc': [], 'ap': [], 'auroc': [],
            'max': [], 'mean': [], 'median': [], 'min': []
        }
        wrong_results = {
            'iou': [], 'acc': [], 'ap': [], 'auroc': [],
            'max': [], 'mean': [], 'median': [], 'min': []
        }
        
        iterator = range(self.limit)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for idx in iterator:
            try:
                # Load Image
                img_ref = self.imgs_refs[idx, 0]
                img_obj = np.array(self.f[img_ref])
                img_np = img_obj.transpose(2, 1, 0)
                base_img = Image.fromarray(img_np)
                
                img_t = self.preprocess(base_img).unsqueeze(0).to(self.device)
                H_feat, W_feat = img_t.shape[-2:]
                
                # Load GT
                gt_ref = self.gts_refs[idx, 0]
                gt_wrapper = self.f[gt_ref]
                if gt_wrapper.dtype == 'object':
                    real_gt_ref = gt_wrapper[0, 0]
                    real_gt = np.array(self.f[real_gt_ref])
                    gt_mask = real_gt.transpose(1, 0)
                else:
                    gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                
                # Resize GT
                gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
                target_resize = transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                )
                gt_pil = target_resize(gt_pil)
                gt_mask = np.array(gt_pil).astype(np.uint8)
                H_gt, W_gt = gt_mask.shape
                
                # Get class info
                wnid = self.wnids_in_seg[idx]
                class_name = self.wnid_to_classname[wnid]
                prompt = self.wnid_to_prompt[wnid]
                cls_idx = self.wnid_to_idx[wnid]
                original_1x = self.all_text_embs[cls_idx:cls_idx + 1]
                
                # Build sparse dictionary
                def build_sparse_embedding(text_emb_1x, target_class_name):
                    """Build sparse residual embedding for a given text embedding."""
                    parts = []
                    
                    # 1) Other class prompts
                    emb_idx = None
                    for i, w in enumerate(self.unique_wnids):
                        if self.wnid_to_classname[w] == target_class_name:
                            emb_idx = i
                            break
                    
                    if dict_include_prompts and len(self.unique_wnids) > 1:
                        if emb_idx is not None:
                            if emb_idx > 0:
                                parts.append(self.all_text_embs[:emb_idx])
                            if emb_idx + 1 < len(self.unique_wnids):
                                parts.append(self.all_text_embs[emb_idx + 1:])
                        else:
                            parts.append(self.all_text_embs)
                    
                    # 2) WordNet neighbors
                    use_wn = any([wn_use_synonyms, wn_use_hypernyms, wn_use_hyponyms, wn_use_siblings])
                    if use_wn:
                        target_prompt = f"a photo of a {target_class_name}."
                        raw_neighbors = wordnet_neighbors_configured(
                            target_class_name,
                            use_synonyms=wn_use_synonyms,
                            use_hypernyms=wn_use_hypernyms,
                            use_hyponyms=wn_use_hyponyms,
                            use_siblings=wn_use_siblings,
                            use_fallback=True,
                            limit_per_relation=8,
                        )
                        if raw_neighbors:
                            neighbor_prompts = [target_prompt.replace(target_class_name, w) for w in raw_neighbors]
                            n_tok = self.tokenizer(neighbor_prompts).to(self.device)
                            with torch.no_grad():
                                n_emb = self.model.encode_text(n_tok)
                                n_emb = F.normalize(n_emb, dim=-1)
                            parts.append(n_emb)
                    
                    # Combine dictionary
                    if len(parts) > 0:
                        D = torch.cat(parts, dim=0)
                        D = F.normalize(D, dim=-1)
                        
                        # Filter by cosine similarity
                        if 0.0 < max_dict_cos_sim < 1.0:
                            sim = (D @ text_emb_1x.t()).squeeze(-1).abs()
                            keep = sim < max_dict_cos_sim
                            D = D[keep]
                    else:
                        D = text_emb_1x.new_zeros((0, text_emb_1x.shape[-1]))
                    
                    # OMP sparse residual
                    sparse_1x = omp_sparse_residual(text_emb_1x, D, max_atoms=atoms)
                    return sparse_1x
                
                def compute_metrics(text_emb_1x, target_class_name):
                    """Compute heatmap and metrics for a given embedding."""
                    sparse_1x = build_sparse_embedding(text_emb_1x, target_class_name)
                    
                    heatmap = compute_map_for_embedding(self.model, img_t, sparse_1x)
                    heatmap_resized = F.interpolate(
                        heatmap.view(1, 1, H_feat, W_feat),
                        size=(H_gt, W_gt),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().numpy()
                    
                    iou, acc = compute_iou_acc(heatmap_resized, gt_mask, threshold=sparse_threshold)
                    ap = compute_map_score(heatmap_resized, gt_mask)
                    
                    # Statistics
                    max_val = float(np.max(heatmap_resized))
                    mean_val = float(np.mean(heatmap_resized))
                    median_val = float(np.median(heatmap_resized))
                    min_val = float(np.min(heatmap_resized))
                    
                    # AUROC
                    gt_binary = (gt_mask > 0).astype(int).flatten()
                    pred_flat = heatmap_resized.flatten()
                    if len(np.unique(gt_binary)) > 1:
                        auroc = roc_auc_score(gt_binary, pred_flat)
                    else:
                        auroc = np.nan
                    
                    return iou, acc, ap, auroc, max_val, mean_val, median_val, min_val
                
                # === CORRECT PROMPT ===
                iou_c, acc_c, ap_c, auroc_c, mx_c, mn_c, md_c, mi_c = compute_metrics(original_1x, class_name)
                correct_results['iou'].append(iou_c)
                correct_results['acc'].append(acc_c)
                correct_results['ap'].append(ap_c)
                if not np.isnan(auroc_c):
                    correct_results['auroc'].append(auroc_c)
                correct_results['max'].append(mx_c)
                correct_results['mean'].append(mn_c)
                correct_results['median'].append(md_c)
                correct_results['min'].append(mi_c)
                
                # === WRONG PROMPTS ===
                neg_indices = self._sample_negative_indices(cls_idx)
                for neg_idx in neg_indices:
                    neg_wnid = self.idx_to_wnid[neg_idx]
                    neg_class_name = self.wnid_to_classname[neg_wnid]
                    neg_emb = self.all_text_embs[neg_idx:neg_idx + 1]
                    
                    iou_w, acc_w, ap_w, auroc_w, mx_w, mn_w, md_w, mi_w = compute_metrics(neg_emb, neg_class_name)
                    wrong_results['iou'].append(iou_w)
                    wrong_results['acc'].append(acc_w)
                    wrong_results['ap'].append(ap_w)
                    if not np.isnan(auroc_w):
                        wrong_results['auroc'].append(auroc_w)
                    wrong_results['max'].append(mx_w)
                    wrong_results['mean'].append(mn_w)
                    wrong_results['median'].append(md_w)
                    wrong_results['min'].append(mi_w)
                
            except Exception as e:
                continue
        
        # Compute averages
        correct_miou = np.mean(correct_results['iou']) * 100 if correct_results['iou'] else 0.0
        correct_macc = np.mean(correct_results['acc']) * 100 if correct_results['acc'] else 0.0
        correct_map = np.mean(correct_results['ap']) * 100 if correct_results['ap'] else 0.0
        correct_auroc = np.mean(correct_results['auroc']) * 100 if correct_results['auroc'] else 0.0
        
        wrong_miou = np.mean(wrong_results['iou']) * 100 if wrong_results['iou'] else 0.0
        wrong_macc = np.mean(wrong_results['acc']) * 100 if wrong_results['acc'] else 0.0
        wrong_map = np.mean(wrong_results['ap']) * 100 if wrong_results['ap'] else 0.0
        wrong_auroc = np.mean(wrong_results['auroc']) * 100 if wrong_results['auroc'] else 0.0
        
        # Statistics
        correct_stats = {
            'max': np.mean(correct_results['max']) if correct_results['max'] else 0.0,
            'mean': np.mean(correct_results['mean']) if correct_results['mean'] else 0.0,
            'median': np.mean(correct_results['median']) if correct_results['median'] else 0.0,
            'min': np.mean(correct_results['min']) if correct_results['min'] else 0.0,
            'n_samples': len(correct_results['iou']),
        }
        wrong_stats = {
            'max': np.mean(wrong_results['max']) if wrong_results['max'] else 0.0,
            'mean': np.mean(wrong_results['mean']) if wrong_results['mean'] else 0.0,
            'median': np.mean(wrong_results['median']) if wrong_results['median'] else 0.0,
            'min': np.mean(wrong_results['min']) if wrong_results['min'] else 0.0,
            'n_samples': len(wrong_results['iou']),
        }
        
        return (correct_miou, wrong_miou, correct_macc, wrong_macc, correct_map, wrong_map,
                correct_auroc, wrong_auroc, correct_stats, wrong_stats)
    
    def __call__(self, trial: optuna.Trial):
        """Optuna objective function."""
        
        # Sample hyperparameters
        wn_use_synonyms = trial.suggest_categorical('wn_use_synonyms', [True, False])
        wn_use_hypernyms = trial.suggest_categorical('wn_use_hypernyms', [True, False])
        wn_use_hyponyms = trial.suggest_categorical('wn_use_hyponyms', [True, False])
        wn_use_siblings = trial.suggest_categorical('wn_use_siblings', [True, False])
        dict_include_prompts = trial.suggest_categorical('dict_include_prompts', [True, False])
        
        sparse_threshold = trial.suggest_float('sparse_threshold', 0.1, 0.9, step=0.025)
        atoms = trial.suggest_int('atoms', 1, 32)
        max_dict_cos_sim = trial.suggest_float('max_dict_cos_sim', 0.5, 1.0, step=0.05)
        
        # Evaluate
        (correct_miou, wrong_miou, correct_acc, wrong_acc, correct_map, wrong_map,
         correct_auroc, wrong_auroc, correct_stats, wrong_stats) = self.evaluate_sparse_config(
            wn_use_synonyms=wn_use_synonyms,
            wn_use_hypernyms=wn_use_hypernyms,
            wn_use_hyponyms=wn_use_hyponyms,
            wn_use_siblings=wn_use_siblings,
            dict_include_prompts=dict_include_prompts,
            sparse_threshold=sparse_threshold,
            atoms=atoms,
            max_dict_cos_sim=max_dict_cos_sim,
            show_progress=False,
        )
        
        # Log all metrics
        trial.set_user_attr('correct_miou', correct_miou)
        trial.set_user_attr('wrong_miou', wrong_miou)
        trial.set_user_attr('correct_acc', correct_acc)
        trial.set_user_attr('wrong_acc', wrong_acc)
        trial.set_user_attr('correct_map', correct_map)
        trial.set_user_attr('wrong_map', wrong_map)
        trial.set_user_attr('correct_auroc', correct_auroc)
        trial.set_user_attr('wrong_auroc', wrong_auroc)
        trial.set_user_attr('correct_stats', correct_stats)
        trial.set_user_attr('wrong_stats', wrong_stats)
        
        # Compute multi-metric composite score for single-objective optimization
        # Objective: maximize correct metrics, minimize wrong metrics
        composite_miou = correct_miou - self.composite_lambda * wrong_miou
        composite_acc = correct_acc - self.composite_lambda * wrong_acc
        composite_map = correct_map - self.composite_lambda * wrong_map
        composite_auroc = correct_auroc - self.composite_lambda * wrong_auroc
        
        # Sum all composite scores (each weighted equally)
        composite = composite_miou + composite_acc + composite_map + composite_auroc
        trial.set_user_attr('composite_score', composite)
        trial.set_user_attr('composite_miou', composite_miou)
        trial.set_user_attr('composite_acc', composite_acc)
        trial.set_user_attr('composite_map', composite_map)
        trial.set_user_attr('composite_auroc', composite_auroc)
        
        # Report for pruning
        trial.report(composite, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if self.multi_objective:
            # Return tuple for multi-objective: all 4 correct and wrong metrics
            # Directions: maximize correct, minimize wrong
            return (correct_miou, correct_acc, correct_map, correct_auroc,
                    wrong_miou, wrong_acc, wrong_map, wrong_auroc)
        else:
            # Return composite score for single-objective
            return composite


def main():
    parser = argparse.ArgumentParser(description='Anti-Hallucination Hyperparameter Optimization')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights (auto-set based on --use_siglip if not provided)')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--use_siglip', action='store_true',
                        help='Use SigLIP instead of CLIP for optimization')
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Anti-hallucination settings
    parser.add_argument('--num_negatives', type=int, default=1, 
                        help='Number of negative (wrong) prompts per image')
    parser.add_argument('--negative_strategy', type=str, default='random', 
                        choices=['random', 'distant'],
                        help='Strategy for sampling negative prompts')
    parser.add_argument('--composite_lambda', type=float, default=0.5,
                        help='Weight for wrong-prompt penalty in composite score')
    parser.add_argument('--multi_objective', action='store_true',
                        help='Use multi-objective Pareto optimization instead of composite score')
    
    # Baseline comparison (from compute_legrad_negative_baseline.py)
    parser.add_argument('--baseline_json', type=str, default=None,
                        help='Path to baseline JSON from compute_legrad_negative_baseline.py')
    parser.add_argument('--baseline_correct_miou', type=float, default=None,
                        help='Manual baseline correct mIoU (overrides --baseline_json)')
    parser.add_argument('--baseline_wrong_miou', type=float, default=None,
                        help='Manual baseline wrong mIoU (overrides --baseline_json)')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='anti_hallucination_optimization')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--output_json', type=str, default='anti_hallucination_results.json')
    
    args = parser.parse_args()
    
    # Load baseline if provided
    baseline_correct_miou = args.baseline_correct_miou
    baseline_wrong_miou = args.baseline_wrong_miou
    baseline_composite = None
    
    if args.baseline_json and os.path.exists(args.baseline_json):
        print(f"Loading baseline from {args.baseline_json}...")
        with open(args.baseline_json, 'r') as f:
            baseline_data = json.load(f)
        if baseline_correct_miou is None:
            baseline_correct_miou = baseline_data.get('correct', {}).get('miou')
        if baseline_wrong_miou is None:
            baseline_wrong_miou = baseline_data.get('wrong', {}).get('miou')
        baseline_composite = baseline_data.get('composite')
        print(f"  Baseline correct mIoU: {baseline_correct_miou:.2f}")
        print(f"  Baseline wrong mIoU: {baseline_wrong_miou:.2f}")
        if baseline_composite is not None:
            print(f"  Baseline composite: {baseline_composite:.2f}")
    
    if baseline_correct_miou is not None and baseline_composite is None:
        baseline_composite = baseline_correct_miou - args.composite_lambda * (baseline_wrong_miou or 0)
    
    # Set model defaults based on --use_siglip
    if args.use_siglip:
        if args.model_name is None:
            args.model_name = 'ViT-B-16-SigLIP'
        if args.pretrained is None:
            args.pretrained = 'webli'
        model_type = 'SigLIP'
    else:
        if args.model_name is None:
            args.model_name = 'ViT-B-16'
        if args.pretrained is None:
            args.pretrained = 'laion2b_s34b_b88k'
        model_type = 'CLIP'
    
    # Load model
    print(f"Loading {model_type} model: {args.model_name} ({args.pretrained})...")
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
    
    # Create objective
    objective = AntiHallucinationObjective(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        dataset_file=args.mat_file,
        wnid_to_label=wnid_to_label,
        device=args.device,
        image_size=args.image_size,
        limit=args.limit,
        num_negatives=args.num_negatives,
        negative_strategy=args.negative_strategy,
        composite_lambda=args.composite_lambda,
        multi_objective=args.multi_objective,
        seed=args.seed,
    )
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
    if args.multi_objective:
        # Multi-objective: maximize all correct metrics, minimize all wrong metrics
        directions = ['maximize', 'maximize', 'maximize', 'maximize',  # correct: mIoU, Acc, mAP, AUROC
                      'minimize', 'minimize', 'minimize', 'minimize']  # wrong: mIoU, Acc, mAP, AUROC
        if args.storage:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                directions=directions,
                sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
            )
        else:
            study = optuna.create_study(
                study_name=args.study_name,
                directions=directions,
                sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
            )
        print(f"\n{'='*60}")
        print("Multi-Objective Optimization Mode (8 objectives)")
        print("MAXIMIZE: Correct mIoU, Accuracy, mAP, AUROC")
        print("MINIMIZE: Wrong mIoU, Accuracy, mAP, AUROC")
        print(f"{'='*60}\n")
    else:
        # Single-objective with composite score
        if args.storage:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                study_name=args.study_name,
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
            )
        print(f"\n{'='*60}")
        print("Multi-Metric Composite Score Optimization")
        print(f"Score = Σ(correct_metric - {args.composite_lambda} × wrong_metric) for mIoU, Acc, mAP, AUROC")
        print(f"{'='*60}\n")
    
    # Custom callback for detailed logging
    def trial_callback(study, trial):
        """Log detailed metrics after each trial."""
        if trial.state != TrialState.COMPLETE:
            return
        
        correct_miou = trial.user_attrs.get('correct_miou', 0)
        wrong_miou = trial.user_attrs.get('wrong_miou', 0)
        correct_auroc = trial.user_attrs.get('correct_auroc', 0)
        wrong_auroc = trial.user_attrs.get('wrong_auroc', 0)
        composite = trial.user_attrs.get('composite_score', trial.value)
        
        # Find best trial so far (for single-objective)
        if not args.multi_objective:
            best_trial = study.best_trial
            best_correct = best_trial.user_attrs.get('correct_miou', 0)
            best_wrong = best_trial.user_attrs.get('wrong_miou', 0)
            best_composite = best_trial.value
            
            # Print detailed info
            print(f"\n  Trial {trial.number}: "
                  f"mIoU(C/W)={correct_miou:.1f}/{wrong_miou:.1f} | "
                  f"AUROC(C/W)={correct_auroc:.1f}/{wrong_auroc:.1f} | "
                  f"Composite={composite:.2f}")
            print(f"  Best so far (Trial {best_trial.number}): "
                  f"Correct mIoU={best_correct:.2f} | Wrong mIoU={best_wrong:.2f} | "
                  f"Composite={best_composite:.2f}")
            
            # Show baseline comparison if available
            if baseline_composite is not None:
                improvement = best_composite - baseline_composite
                print(f"  vs Baseline: Composite improvement = {improvement:+.2f}")
    
    # Run optimization
    print(f"Starting Optuna optimization with {args.n_trials} trials")
    print(f"Model: {args.model_name} ({args.pretrained}) [{model_type}]")
    print(f"Negative strategy: {args.negative_strategy}, Num negatives: {args.num_negatives}")
    if baseline_composite is not None:
        print(f"Baseline to beat: Correct={baseline_correct_miou:.2f} | "
              f"Wrong={baseline_wrong_miou:.2f} | Composite={baseline_composite:.2f}")
    print(f"{'='*60}\n")
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        callbacks=[trial_callback],
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    if args.multi_objective:
        # Multi-objective results: show Pareto front
        pareto_trials = study.best_trials
        print(f"\nPareto-optimal trials: {len(pareto_trials)}")
        
        print(f"\n{'='*60}")
        print("Pareto Front (Best Trade-offs):")
        print(f"{'='*60}")
        
        for i, trial in enumerate(pareto_trials[:10], 1):
            vals = trial.values
            print(f"\n#{i} Trial {trial.number}:")
            print(f"  Correct: mIoU={vals[0]:.2f} | Acc={vals[1]:.2f} | mAP={vals[2]:.2f} | AUROC={vals[3]:.2f}")
            print(f"  Wrong:   mIoU={vals[4]:.2f} | Acc={vals[5]:.2f} | mAP={vals[6]:.2f} | AUROC={vals[7]:.2f}")
            print(f"  Params: {trial.params}")
        
        results = {
            'mode': 'multi_objective',
            'n_trials': len(study.trials),
            'pareto_trials': [
                {
                    'trial_number': t.number,
                    'correct': {'miou': t.values[0], 'acc': t.values[1], 'map': t.values[2], 'auroc': t.values[3]},
                    'wrong': {'miou': t.values[4], 'acc': t.values[5], 'map': t.values[6], 'auroc': t.values[7]},
                    'params': t.params,
                }
                for t in pareto_trials
            ],
        }
    else:
        # Single-objective results
        best_trial = study.best_trial
        
        # Extract all metrics from best trial
        correct_metrics = {
            'miou': best_trial.user_attrs.get('correct_miou', 0),
            'acc': best_trial.user_attrs.get('correct_acc', 0),
            'map': best_trial.user_attrs.get('correct_map', 0),
            'auroc': best_trial.user_attrs.get('correct_auroc', 0),
        }
        wrong_metrics = {
            'miou': best_trial.user_attrs.get('wrong_miou', 0),
            'acc': best_trial.user_attrs.get('wrong_acc', 0),
            'map': best_trial.user_attrs.get('wrong_map', 0),
            'auroc': best_trial.user_attrs.get('wrong_auroc', 0),
        }
        correct_stats = best_trial.user_attrs.get('correct_stats', {})
        wrong_stats = best_trial.user_attrs.get('wrong_stats', {})
        
        print(f"\n{'='*60}")
        print("BEST TRIAL RESULTS")
        print(f"{'='*60}")
        print(f"\nComposite Score: {best_trial.value:.2f}")
        
        print(f"\n=== CORRECT PROMPTS (image class = text prompt class) ===")
        print(f"  mIoU:     {correct_metrics['miou']:.2f}")
        print(f"  Accuracy: {correct_metrics['acc']:.2f}")
        print(f"  mAP:      {correct_metrics['map']:.2f}")
        print(f"  AUROC:    {correct_metrics['auroc']:.2f}")
        if correct_stats:
            print(f"  Max Val:  {correct_stats.get('max', 0):.4f}")
            print(f"  Mean Val: {correct_stats.get('mean', 0):.4f}")
            print(f"  Median:   {correct_stats.get('median', 0):.4f}")
            print(f"  Min Val:  {correct_stats.get('min', 0):.4f}")
            print(f"  Samples:  {correct_stats.get('n_samples', 0)}")
        
        print(f"\n=== WRONG PROMPTS (image class ≠ text prompt class) ===")
        print(f"  mIoU:     {wrong_metrics['miou']:.2f}")
        print(f"  Accuracy: {wrong_metrics['acc']:.2f}")
        print(f"  mAP:      {wrong_metrics['map']:.2f}")
        print(f"  AUROC:    {wrong_metrics['auroc']:.2f}")
        if wrong_stats:
            print(f"  Max Val:  {wrong_stats.get('max', 0):.4f}")
            print(f"  Mean Val: {wrong_stats.get('mean', 0):.4f}")
            print(f"  Median:   {wrong_stats.get('median', 0):.4f}")
            print(f"  Min Val:  {wrong_stats.get('min', 0):.4f}")
            print(f"  Samples:  {wrong_stats.get('n_samples', 0)}")
        
        print(f"\n=== COMPOSITE BREAKDOWN ===")
        print(f"  mIoU:     {best_trial.user_attrs.get('composite_miou', 0):.2f}")
        print(f"  Accuracy: {best_trial.user_attrs.get('composite_acc', 0):.2f}")
        print(f"  mAP:      {best_trial.user_attrs.get('composite_map', 0):.2f}")
        print(f"  AUROC:    {best_trial.user_attrs.get('composite_auroc', 0):.2f}")
        
        print(f"\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Top 5 by composite score
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
        
        print(f"\n{'='*60}")
        print("Top 5 trials by composite score:")
        print(f"{'='*60}")
        
        for i, trial in enumerate(sorted_trials, 1):
            print(f"\n#{i} Trial {trial.number}:")
            print(f"  Composite: {trial.value:.2f}")
            print(f"  Correct: mIoU={trial.user_attrs.get('correct_miou', 0):.1f} | "
                  f"Acc={trial.user_attrs.get('correct_acc', 0):.1f} | "
                  f"mAP={trial.user_attrs.get('correct_map', 0):.1f} | "
                  f"AUROC={trial.user_attrs.get('correct_auroc', 0):.1f}")
            print(f"  Wrong:   mIoU={trial.user_attrs.get('wrong_miou', 0):.1f} | "
                  f"Acc={trial.user_attrs.get('wrong_acc', 0):.1f} | "
                  f"mAP={trial.user_attrs.get('wrong_map', 0):.1f} | "
                  f"AUROC={trial.user_attrs.get('wrong_auroc', 0):.1f}")
        
        results = {
            'mode': 'composite',
            'composite_lambda': args.composite_lambda,
            'best_composite_score': best_trial.value,
            'best_correct': correct_metrics,
            'best_wrong': wrong_metrics,
            'best_correct_stats': correct_stats,
            'best_wrong_stats': wrong_stats,
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'model_settings': {
                'model_name': args.model_name,
                'pretrained': args.pretrained,
                'model_type': model_type,
                'use_siglip': args.use_siglip,
            },
            'top_5_trials': [
                {
                    'trial_number': t.number,
                    'composite_score': t.value,
                    'correct': {
                        'miou': t.user_attrs.get('correct_miou'),
                        'acc': t.user_attrs.get('correct_acc'),
                        'map': t.user_attrs.get('correct_map'),
                        'auroc': t.user_attrs.get('correct_auroc'),
                    },
                    'wrong': {
                        'miou': t.user_attrs.get('wrong_miou'),
                        'acc': t.user_attrs.get('wrong_acc'),
                        'map': t.user_attrs.get('wrong_map'),
                        'auroc': t.user_attrs.get('wrong_auroc'),
                    },
                    'params': t.params,
                }
                for t in sorted_trials
            ],
        }
        
        # Print command to reproduce best result
        best_params = best_trial.params
        print(f"\n{'='*60}")
        print("Command to reproduce best result:")
        print(f"{'='*60}")
        cmd = f"""python scripts/benchmark_segmentation.py \\
    --wn_use_synonyms {1 if best_params['wn_use_synonyms'] else 0} \\
    --wn_use_hypernyms {1 if best_params['wn_use_hypernyms'] else 0} \\
    --wn_use_hyponyms {1 if best_params['wn_use_hyponyms'] else 0} \\
    --wn_use_siblings {1 if best_params['wn_use_siblings'] else 0} \\
    --dict_include_prompts {1 if best_params['dict_include_prompts'] else 0} \\
    --sparse_threshold {best_params['sparse_threshold']:.3f} \\
    --atoms {best_params['atoms']} \\
    --max_dict_cos_sim {best_params['max_dict_cos_sim']:.2f}"""
        print(cmd)
    
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
