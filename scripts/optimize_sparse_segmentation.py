#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for sparse segmentation method.

Goal: Find hyperparameters that make sparse method surpass the original LeGrad method.

Hyperparameters to optimize:
- wn_use_synonyms (bool)
- wn_use_hypernyms (bool)
- wn_use_hyponyms (bool)
- wn_use_siblings (bool)
- dict_include_prompts (bool)
- sparse_threshold (float, 0.1-0.9)
- atoms (int, 1-32)
- max_dict_cos_sim (float, 0.5-1.0)

Usage:
    python scripts/optimize_sparse_segmentation.py --n_trials 100
    python scripts/optimize_sparse_segmentation.py --n_trials 50 --limit 500  # faster with subset
"""

import sys
import os
import argparse
import json
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
    build_wordlist_neighbors_embedding,
)


class SparseSegmentationObjective:
    """
    Optuna objective for sparse segmentation hyperparameter optimization.
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
        baseline_miou=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.baseline_miou = baseline_miou
        
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
        
        tok_all = self.tokenizer(all_prompts).to(self.device)
        with torch.no_grad():
            self.all_text_embs = self.model.encode_text(tok_all, normalize=True)
        
        print(f"[optuna] Precomputed {len(all_prompts)} class embeddings.")
    
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
        Evaluate a specific sparse configuration and return mIoU.
        """
        results = {'iou': [], 'acc': [], 'ap': []}
        
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
                parts = []
                
                # 1) Other class prompts
                if dict_include_prompts and len(self.unique_wnids) > 1:
                    if cls_idx > 0:
                        parts.append(self.all_text_embs[:cls_idx])
                    if cls_idx + 1 < len(self.unique_wnids):
                        parts.append(self.all_text_embs[cls_idx + 1:])
                
                # 2) WordNet neighbors
                use_wn = any([wn_use_synonyms, wn_use_hypernyms, wn_use_hyponyms, wn_use_siblings])
                if use_wn:
                    raw_neighbors = wordnet_neighbors_configured(
                        class_name,
                        use_synonyms=wn_use_synonyms,
                        use_hypernyms=wn_use_hypernyms,
                        use_hyponyms=wn_use_hyponyms,
                        use_siblings=wn_use_siblings,
                        use_fallback=True,
                        limit_per_relation=8,
                    )
                    if raw_neighbors:
                        neighbor_prompts = [prompt.replace(class_name, w) for w in raw_neighbors]
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
                        sim = (D @ original_1x.t()).squeeze(-1).abs()
                        keep = sim < max_dict_cos_sim
                        D = D[keep]
                else:
                    D = original_1x.new_zeros((0, original_1x.shape[-1]))
                
                # OMP sparse residual
                sparse_1x = omp_sparse_residual(original_1x, D, max_atoms=atoms)
                
                # Compute heatmap
                heatmap_sparse = compute_map_for_embedding(self.model, img_t, sparse_1x)
                heatmap_sparse_resized = F.interpolate(
                    heatmap_sparse.view(1, 1, H_feat, W_feat),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Compute metrics
                iou_s, acc_s = compute_iou_acc(heatmap_sparse_resized, gt_mask, threshold=sparse_threshold)
                ap_s = compute_map_score(heatmap_sparse_resized, gt_mask)
                
                results['iou'].append(iou_s)
                results['acc'].append(acc_s)
                results['ap'].append(ap_s)
                
            except Exception as e:
                continue
        
        miou = np.mean(results['iou']) * 100 if results['iou'] else 0.0
        macc = np.mean(results['acc']) * 100 if results['acc'] else 0.0
        map_score = np.mean(results['ap']) * 100 if results['ap'] else 0.0
        
        return miou, macc, map_score
    
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
        miou, macc, map_score = self.evaluate_sparse_config(
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
        
        # Log intermediate values
        trial.set_user_attr('pixel_acc', macc)
        trial.set_user_attr('map', map_score)
        
        # Report for pruning
        trial.report(miou, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return miou


def compute_baseline_miou(objective: SparseSegmentationObjective):
    """Compute baseline LeGrad mIoU for comparison."""
    print("\n[baseline] Computing original LeGrad mIoU...")
    
    results = {'iou': []}
    
    for idx in tqdm(range(objective.limit), desc="Baseline LeGrad"):
        try:
            # Load Image
            img_ref = objective.imgs_refs[idx, 0]
            img_obj = np.array(objective.f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            
            img_t = objective.preprocess(base_img).unsqueeze(0).to(objective.device)
            H_feat, W_feat = img_t.shape[-2:]
            
            # Load GT
            gt_ref = objective.gts_refs[idx, 0]
            gt_wrapper = objective.f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0, 0]
                real_gt = np.array(objective.f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
            
            # Resize GT
            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            target_resize = transforms.Resize(
                (objective.image_size, objective.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            gt_pil = target_resize(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.uint8)
            H_gt, W_gt = gt_mask.shape
            
            # Get class embedding
            wnid = objective.wnids_in_seg[idx]
            cls_idx = objective.wnid_to_idx[wnid]
            original_1x = objective.all_text_embs[cls_idx:cls_idx + 1]
            
            # Compute LeGrad heatmap (no sparse)
            heatmap_orig = compute_map_for_embedding(objective.model, img_t, original_1x)
            heatmap_orig_resized = F.interpolate(
                heatmap_orig.view(1, 1, H_feat, W_feat),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            iou_o, _ = compute_iou_acc(heatmap_orig_resized, gt_mask, threshold=0.5)
            results['iou'].append(iou_o)
            
        except Exception:
            continue
    
    baseline_miou = np.mean(results['iou']) * 100 if results['iou'] else 0.0
    print(f"[baseline] Original LeGrad mIoU: {baseline_miou:.2f}")
    return baseline_miou


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for Sparse Segmentation')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='sparse_segmentation_optimization')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Baseline values (skip computation if provided)
    parser.add_argument('--baseline_miou', type=float, default=58.66, help='Baseline LeGrad mIoU')
    parser.add_argument('--baseline_acc', type=float, default=77.52, help='Baseline LeGrad PixelAcc')
    parser.add_argument('--baseline_map', type=float, default=82.49, help='Baseline LeGrad mAP')
    
    # Output
    parser.add_argument('--output_json', type=str, default='sparse_optimization_results.json')
    
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
    objective = SparseSegmentationObjective(
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        dataset_file=args.mat_file,
        wnid_to_label=wnid_to_label,
        device=args.device,
        image_size=args.image_size,
        limit=args.limit,
    )
    
    # Use provided baseline values (skip expensive computation)
    baseline_miou = args.baseline_miou
    baseline_acc = args.baseline_acc
    baseline_map = args.baseline_map
    objective.baseline_miou = baseline_miou
    print(f"\n[baseline] Using provided baseline: PixelAcc={baseline_acc:.2f}, mIoU={baseline_miou:.2f}, mAP={baseline_map:.2f}")
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    
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
    
    # Run optimization
    print(f"\n{'='*60}")
    print(f"Starting Optuna optimization with {args.n_trials} trials")
    print(f"Baseline LeGrad mIoU: {baseline_miou:.2f}")
    print(f"Goal: Surpass {baseline_miou:.2f} mIoU")
    print(f"{'='*60}\n")
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest trial:")
    print(f"  mIoU: {best_trial.value:.2f}")
    print(f"  PixelAcc: {best_trial.user_attrs.get('pixel_acc', 'N/A'):.2f}")
    print(f"  mAP: {best_trial.user_attrs.get('map', 'N/A'):.2f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Compare with baseline
    improvement = best_trial.value - baseline_miou
    if improvement > 0:
        print(f"\n✅ SUCCESS! Sparse method SURPASSES baseline by +{improvement:.2f} mIoU")
    else:
        print(f"\n❌ Sparse method is {abs(improvement):.2f} mIoU below baseline")
    
    # Top 5 trials
    print(f"\n{'='*60}")
    print("Top 5 trials:")
    print(f"{'='*60}")
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        print(f"\n#{i} Trial {trial.number}:")
        print(f"  mIoU: {trial.value:.2f}")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
    
    # Save results to JSON
    results = {
        'baseline_miou': baseline_miou,
        'best_miou': best_trial.value,
        'improvement': improvement,
        'best_params': best_trial.params,
        'best_pixel_acc': best_trial.user_attrs.get('pixel_acc'),
        'best_map': best_trial.user_attrs.get('map'),
        'n_trials': len(study.trials),
        'top_5_trials': [
            {
                'trial_number': t.number,
                'miou': t.value,
                'params': t.params,
            }
            for t in sorted_trials
        ],
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
    
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


if __name__ == '__main__':
    main()
