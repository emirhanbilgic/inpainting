#!/usr/bin/env python3
"""
Constrained DAAM hyperparameter optimization (20 trials).

Searches within tight bounds:
    sparse_threshold : 0.20 – 0.40
    atoms            : 16 – 24
    max_dict_cos_sim : 0.60 – 0.95
    omp_beta         : 0.10 – 0.30

Dictionary flags are fixed (LLM/GPT/WordNet chosen via CLI, same as the
full optimizer), so only the four numerical hyper-parameters are explored.

Usage (minimal):
    python scripts/optimize_daam_constrained.py

Usage (with LLM dictionary):
    python scripts/optimize_daam_constrained.py \
        --use_llm_dictionary \
        --llm_dictionary_path scripts/visual_concept_dictionary_445.json

All other flags (--limit, --device, --fix_dictionary, …) are forwarded to
the underlying AntiHallucinationObjective unchanged.
"""
import argparse
import json
import os
import sys
import random
import math

import torch
import numpy as np
import h5py
from PIL import Image

# ── path setup (same as the parent script) ─────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir  = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import optuna
from optuna.trial import TrialState
import open_clip
import nltk
import torch.nn.functional as F

# LeGrad components
from legrad import LeWrapper, LePreprocess
from benchmark_segmentation_v2 import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
)

# Import the heavy objective class from the full optimizer
from optimize_anti_hallucination import (
    AntiHallucinationObjective,
    BASELINES,
)

# ── constrained objective wrapper ──────────────────────────────────────

class ConstrainedDAAMObjective:
    """
    Thin wrapper around AntiHallucinationObjective.__call__ that narrows
    the Optuna search space to the four user-specified ranges.

    The dictionary flags (WordNet/LLM/GPT) are still decided by the
    underlying objective's constructor flags – no trial sampling for those.
    """

    # Constrained ranges
    SPARSE_THRESHOLD_RANGE = (0.20, 0.40)
    ATOMS_RANGE            = (16, 24)
    MAX_DICT_COS_SIM_RANGE = (0.60, 0.95)
    OMP_BETA_RANGE         = (0.10, 0.30)

    def __init__(self, inner: AntiHallucinationObjective):
        self.inner = inner

    # ─────────────────────────────────────────────────────────────────
    def __call__(self, trial: optuna.Trial):
        """Sample within constrained ranges, then delegate to inner eval."""

        # ── dictionary flags (fixed, determined by CLI) ────────────
        if self.inner.use_llm_dictionary or self.inner.use_gpt_dictionary:
            wn_use_synonyms    = False
            wn_use_hypernyms   = False
            wn_use_hyponyms    = False
            wn_use_siblings    = False
            dict_include_prompts = False
        elif self.inner.fix_dictionary:
            wn_use_synonyms    = True
            wn_use_hypernyms   = True
            wn_use_hyponyms    = True
            wn_use_siblings    = False
            dict_include_prompts = True
        elif self.inner.fix_dictionary_wordnet_only:
            wn_use_synonyms    = False
            wn_use_hypernyms   = True
            wn_use_hyponyms    = True
            wn_use_siblings    = True
            dict_include_prompts = False
        elif self.inner.fix_dictionary_prompts_only:
            wn_use_synonyms    = False
            wn_use_hypernyms   = False
            wn_use_hyponyms    = False
            wn_use_siblings    = False
            dict_include_prompts = True
        else:
            # Default: full WordNet + prompts
            wn_use_synonyms    = True
            wn_use_hypernyms   = True
            wn_use_hyponyms    = True
            wn_use_siblings    = False
            dict_include_prompts = True

        # ── constrained hyper-parameters ───────────────────────────
        lo, hi = self.SPARSE_THRESHOLD_RANGE
        sparse_threshold = trial.suggest_float(
            'sparse_threshold', lo, hi, step=0.025,
        )

        lo, hi = self.ATOMS_RANGE
        atoms = trial.suggest_int('atoms', lo, hi)

        lo, hi = self.MAX_DICT_COS_SIM_RANGE
        max_dict_cos_sim = trial.suggest_float(
            'max_dict_cos_sim', lo, hi, step=0.05,
        )

        lo, hi = self.OMP_BETA_RANGE
        omp_beta = trial.suggest_float(
            'omp_beta', lo, hi, step=0.025,
        )

        # ── evaluate ───────────────────────────────────────────────
        (correct_miou, wrong_miou,
         correct_acc,  wrong_acc,
         correct_map,  wrong_map,
         correct_auroc, wrong_auroc,
         correct_auroc_max, correct_auroc_min, correct_auroc_median,
         wrong_auroc_max,   wrong_auroc_min,   wrong_auroc_median,
         correct_stats, wrong_stats) = self.inner.evaluate_sparse_config(
            wn_use_synonyms=wn_use_synonyms,
            wn_use_hypernyms=wn_use_hypernyms,
            wn_use_hyponyms=wn_use_hyponyms,
            wn_use_siblings=wn_use_siblings,
            dict_include_prompts=dict_include_prompts,
            sparse_threshold=sparse_threshold,
            atoms=atoms,
            max_dict_cos_sim=max_dict_cos_sim,
            omp_beta=omp_beta,
            show_progress=False,
        )

        # ── log metrics ────────────────────────────────────────────
        trial.set_user_attr('correct_miou', correct_miou)
        trial.set_user_attr('wrong_miou',   wrong_miou)
        trial.set_user_attr('correct_acc',  correct_acc)
        trial.set_user_attr('wrong_acc',    wrong_acc)
        trial.set_user_attr('correct_map',  correct_map)
        trial.set_user_attr('wrong_map',    wrong_map)
        trial.set_user_attr('correct_auroc', correct_auroc)
        trial.set_user_attr('wrong_auroc',   wrong_auroc)
        trial.set_user_attr('correct_auroc_max',    correct_auroc_max)
        trial.set_user_attr('correct_auroc_min',    correct_auroc_min)
        trial.set_user_attr('correct_auroc_median', correct_auroc_median)
        trial.set_user_attr('wrong_auroc_max',    wrong_auroc_max)
        trial.set_user_attr('wrong_auroc_min',    wrong_auroc_min)
        trial.set_user_attr('wrong_auroc_median', wrong_auroc_median)
        trial.set_user_attr('correct_stats', correct_stats)
        trial.set_user_attr('wrong_stats',   wrong_stats)

        # ── composite score (gap-improvement) ──────────────────────
        base = self.inner.baseline_metrics
        if base:
            base_c, base_w = base['correct'], base['wrong']

            gap_imp_miou  = (correct_miou - wrong_miou) - (base_c['miou'] - base_w['miou'])
            gap_imp_acc   = (correct_acc  - wrong_acc)  - (base_c['acc']  - base_w['acc'])
            gap_imp_map   = (correct_map  - wrong_map)  - (base_c['map']  - base_w['map'])
            gap_imp_auroc = correct_auroc - base_c.get('auroc', correct_auroc)

            d_c_miou = correct_miou - base_c['miou']
            d_c_acc  = correct_acc  - base_c['acc']
            d_c_map  = correct_map  - base_c['map']
            d_c_auroc = correct_auroc - base_c.get('auroc', correct_auroc)

            d_w_miou = base_w['miou'] - wrong_miou
            d_w_acc  = base_w['acc']  - wrong_acc
            d_w_map  = base_w['map']  - wrong_map
            d_w_auroc = base_w.get('auroc', wrong_auroc) - wrong_auroc

            trial.set_user_attr('delta_correct_miou', d_c_miou)
            trial.set_user_attr('delta_correct_acc',  d_c_acc)
            trial.set_user_attr('delta_correct_map',  d_c_map)
            trial.set_user_attr('delta_correct_auroc', d_c_auroc)
            trial.set_user_attr('delta_wrong_miou', d_w_miou)
            trial.set_user_attr('delta_wrong_acc',  d_w_acc)
            trial.set_user_attr('delta_wrong_map',  d_w_map)
            trial.set_user_attr('delta_wrong_auroc', d_w_auroc)

            trial.set_user_attr('gap_improvement_miou',  gap_imp_miou)
            trial.set_user_attr('gap_improvement_acc',   gap_imp_acc)
            trial.set_user_attr('gap_improvement_map',   gap_imp_map)
            trial.set_user_attr('gap_improvement_auroc', gap_imp_auroc)

            composite = gap_imp_miou + gap_imp_acc + gap_imp_map + gap_imp_auroc
        else:
            composite = (correct_miou - wrong_miou) + \
                        (correct_acc  - wrong_acc)  + \
                        (correct_map  - wrong_map)  + \
                        correct_auroc

        trial.set_user_attr('composite_score', composite)
        trial.report(composite, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return composite


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Constrained DAAM hyperparameter optimization (20 trials)',
    )

    # Dataset / model
    parser.add_argument('--mat_file', type=str,
                        default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of images (0 = all)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--class_index_path', type=str,
                        default='resources/imagenet_class_index.json')

    # Anti-hallucination
    parser.add_argument('--num_negatives', type=int, default=1)
    parser.add_argument('--negative_strategy', type=str, default='random',
                        choices=['random', 'distant'])

    # DAAM settings
    parser.add_argument('--daam_model_id', type=str,
                        default='Manojb/stable-diffusion-2-base')

    # Dictionary selection (mutually exclusive)
    parser.add_argument('--fix_dictionary', action='store_true',
                        help='WordNet + prompts (default if no flag)')
    parser.add_argument('--fix_dictionary_wordnet_only', action='store_true')
    parser.add_argument('--fix_dictionary_prompts_only', action='store_true')
    parser.add_argument('--use_llm_dictionary', action='store_true')
    parser.add_argument('--llm_dictionary_path', type=str, default=None)
    parser.add_argument('--use_gpt_dictionary', action='store_true')
    parser.add_argument('--gpt_dictionary_path', type=str, default=None)

    # Optuna / output
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of Optuna trials (default 20)')
    parser.add_argument('--study_name', type=str,
                        default='daam_constrained_optimization')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_json', type=str,
                        default='daam_constrained_results.json')

    args = parser.parse_args()

    # ── baseline ───────────────────────────────────────────────────
    baseline_metrics = BASELINES['DAAM']['DAAM']
    print(f"DAAM baseline:")
    print(f"  Correct: mIoU={baseline_metrics['correct']['miou']}, "
          f"Acc={baseline_metrics['correct']['acc']}, "
          f"mAP={baseline_metrics['correct']['map']}")
    print(f"  Wrong:   mIoU={baseline_metrics['wrong']['miou']}, "
          f"Acc={baseline_metrics['wrong']['acc']}, "
          f"mAP={baseline_metrics['wrong']['map']}")

    # ── load CLIP model (needed by AntiHallucinationObjective) ─────
    model_name = 'ViT-B-16'
    pretrained = 'laion2b_s34b_b88k'
    print(f"\nLoading CLIP model: {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # ── NLTK ───────────────────────────────────────────────────────
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception:
        pass

    # ── ImageNet class index ───────────────────────────────────────
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"[imagenet] {len(wnid_to_label)} wnids loaded.")
    except Exception as e:
        print(f"[imagenet] Warning: {e}")
        wnid_to_label = {}

    # ── inner objective ────────────────────────────────────────────
    inner = AntiHallucinationObjective(
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
        composite_lambda=0.5,
        gap_only=False,
        multi_objective=False,
        seed=args.seed,
        use_gradcam=False,
        gradcam_layer=8,
        use_chefercam=False,
        threshold_mode='fixed',
        fixed_threshold=0.5,
        baseline_metrics=baseline_metrics,
        use_attentioncam=False,
        lrp_start_layer=1,
        use_daam=True,
        use_daam_keyspace_omp=True,
        daam_model_id=args.daam_model_id,
        fix_dictionary=args.fix_dictionary,
        fix_dictionary_wordnet_only=args.fix_dictionary_wordnet_only,
        fix_dictionary_prompts_only=args.fix_dictionary_prompts_only,
        use_llm_dictionary=args.use_llm_dictionary,
        llm_dictionary_path=args.llm_dictionary_path,
        use_gpt_dictionary=args.use_gpt_dictionary,
        gpt_dictionary_path=args.gpt_dictionary_path,
    )

    objective = ConstrainedDAAMObjective(inner)

    # ── Optuna study ───────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study   = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    # ── callback ───────────────────────────────────────────────────
    def trial_callback(study, trial):
        if trial.state != TrialState.COMPLETE:
            return

        c_miou = trial.user_attrs.get('correct_miou', 0)
        w_miou = trial.user_attrs.get('wrong_miou', 0)
        c_acc  = trial.user_attrs.get('correct_acc', 0)
        w_acc  = trial.user_attrs.get('wrong_acc', 0)
        c_map  = trial.user_attrs.get('correct_map', 0)
        w_map  = trial.user_attrs.get('wrong_map', 0)
        c_auroc = trial.user_attrs.get('correct_auroc', 0)
        w_auroc = trial.user_attrs.get('wrong_auroc', 0)

        dc_miou = trial.user_attrs.get('delta_correct_miou', 0)
        dc_acc  = trial.user_attrs.get('delta_correct_acc', 0)
        dc_map  = trial.user_attrs.get('delta_correct_map', 0)
        dc_auroc = trial.user_attrs.get('delta_correct_auroc', 0)
        dw_miou = trial.user_attrs.get('delta_wrong_miou', 0)
        dw_acc  = trial.user_attrs.get('delta_wrong_acc', 0)
        dw_map  = trial.user_attrs.get('delta_wrong_map', 0)
        dw_auroc = trial.user_attrs.get('delta_wrong_auroc', 0)

        composite = trial.user_attrs.get('composite_score', trial.value)

        gap_imp_miou  = trial.user_attrs.get('gap_improvement_miou', 0)
        gap_imp_acc   = trial.user_attrs.get('gap_improvement_acc', 0)
        gap_imp_map   = trial.user_attrs.get('gap_improvement_map', 0)
        gap_imp_auroc = trial.user_attrs.get('gap_improvement_auroc', 0)

        print(f"\n{'─'*70}")
        print(f"  Trial {trial.number} │ Score={composite:.2f}")
        print(f"{'─'*70}")
        print(f"  {'':\u003e10} {'mIoU':\u003e8} {'Acc':\u003e8} {'mAP':\u003e8} {'AUROC':\u003e8}")
        print(f"  {'Correct':\u003e10} {c_miou:\u003e7.2f}% {c_acc:\u003e7.2f}% {c_map:\u003e7.2f}% {c_auroc:\u003e7.2f}%")
        print(f"  {'  Δbase':\u003e10} {dc_miou:\u003e+7.2f}  {dc_acc:\u003e+7.2f}  {dc_map:\u003e+7.2f}  {dc_auroc:\u003e+7.2f}")
        print(f"  {'Wrong':\u003e10} {w_miou:\u003e7.2f}% {w_acc:\u003e7.2f}% {w_map:\u003e7.2f}% {w_auroc:\u003e7.2f}%")
        print(f"  {'  Δbase':\u003e10} {dw_miou:\u003e+7.2f}  {dw_acc:\u003e+7.2f}  {dw_map:\u003e+7.2f}  {dw_auroc:\u003e+7.2f}")
        print(f"  {'Gap Imprv':\u003e10} {gap_imp_miou:\u003e+7.2f}  {gap_imp_acc:\u003e+7.2f}  {gap_imp_map:\u003e+7.2f}  {gap_imp_auroc:\u003e+7.2f}")
        print(f"  Params: {trial.params}")

        best = study.best_trial
        b_comp = best.user_attrs.get('composite_score', best.value)
        print(f"  Best so far: Trial {best.number} (Score={b_comp:.2f})")
        print(f"{'─'*70}")

    # ── run ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CONSTRAINED DAAM OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  sparse_threshold : {ConstrainedDAAMObjective.SPARSE_THRESHOLD_RANGE}")
    print(f"  atoms            : {ConstrainedDAAMObjective.ATOMS_RANGE}")
    print(f"  max_dict_cos_sim : {ConstrainedDAAMObjective.MAX_DICT_COS_SIM_RANGE}")
    print(f"  omp_beta         : {ConstrainedDAAMObjective.OMP_BETA_RANGE}")
    print(f"  n_trials         : {args.n_trials}")
    print(f"  DAAM model       : {args.daam_model_id}")
    print(f"{'='*60}\n")

    # DAAM is NOT thread-safe → n_jobs=1
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[trial_callback],
    )

    # ── print results ──────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    best = study.best_trial

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Completed trials: {len(completed)}")

    c = best.user_attrs
    print(f"\nBest Trial #{best.number}  (composite = {best.value:.2f})")
    print(f"  Correct  mIoU={c.get('correct_miou',0):.2f}  "
          f"Acc={c.get('correct_acc',0):.2f}  "
          f"mAP={c.get('correct_map',0):.2f}  "
          f"AUROC={c.get('correct_auroc',0):.2f}")
    print(f"  Wrong    mIoU={c.get('wrong_miou',0):.2f}  "
          f"Acc={c.get('wrong_acc',0):.2f}  "
          f"mAP={c.get('wrong_map',0):.2f}  "
          f"AUROC={c.get('wrong_auroc',0):.2f}")
    print(f"\n  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Top-5
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n{'='*60}")
    print("Top 5 trials:")
    print(f"{'='*60}")
    for i, t in enumerate(sorted_trials, 1):
        a = t.user_attrs
        print(f"  #{i}  Trial {t.number}  score={t.value:.2f}")
        print(f"       Correct mIoU={a.get('correct_miou',0):.2f} | "
              f"Acc={a.get('correct_acc',0):.2f} | "
              f"mAP={a.get('correct_map',0):.2f} | "
              f"AUROC={a.get('correct_auroc',0):.2f}")
        print(f"       Wrong   mIoU={a.get('wrong_miou',0):.2f} | "
              f"Acc={a.get('wrong_acc',0):.2f} | "
              f"mAP={a.get('wrong_map',0):.2f} | "
              f"AUROC={a.get('wrong_auroc',0):.2f}")
        print(f"       Params: {t.params}")

    # ── save JSON ──────────────────────────────────────────────────
    results = {
        'mode': 'daam_constrained',
        'search_ranges': {
            'sparse_threshold': list(ConstrainedDAAMObjective.SPARSE_THRESHOLD_RANGE),
            'atoms':            list(ConstrainedDAAMObjective.ATOMS_RANGE),
            'max_dict_cos_sim': list(ConstrainedDAAMObjective.MAX_DICT_COS_SIM_RANGE),
            'omp_beta':         list(ConstrainedDAAMObjective.OMP_BETA_RANGE),
        },
        'n_trials': len(study.trials),
        'best_composite_score': best.value,
        'best_correct': {
            'miou':  c.get('correct_miou'),
            'acc':   c.get('correct_acc'),
            'map':   c.get('correct_map'),
            'auroc': c.get('correct_auroc'),
            'auroc_max':    c.get('correct_auroc_max'),
            'auroc_min':    c.get('correct_auroc_min'),
            'auroc_median': c.get('correct_auroc_median'),
        },
        'best_wrong': {
            'miou':  c.get('wrong_miou'),
            'acc':   c.get('wrong_acc'),
            'map':   c.get('wrong_map'),
            'auroc': c.get('wrong_auroc'),
            'auroc_max':    c.get('wrong_auroc_max'),
            'auroc_min':    c.get('wrong_auroc_min'),
            'auroc_median': c.get('wrong_auroc_median'),
        },
        'best_params': best.params,
        'best_gap_analysis': {
            'gap_improvement_miou':  c.get('gap_improvement_miou'),
            'gap_improvement_acc':   c.get('gap_improvement_acc'),
            'gap_improvement_map':   c.get('gap_improvement_map'),
            'gap_improvement_auroc': c.get('gap_improvement_auroc'),
        },
        'baseline_metrics': baseline_metrics,
        'daam_model_id': args.daam_model_id,
        'top_5_trials': [
            {
                'trial_number': t.number,
                'composite_score': t.value,
                'correct': {
                    'miou':  t.user_attrs.get('correct_miou'),
                    'acc':   t.user_attrs.get('correct_acc'),
                    'map':   t.user_attrs.get('correct_map'),
                    'auroc': t.user_attrs.get('correct_auroc'),
                },
                'wrong': {
                    'miou':  t.user_attrs.get('wrong_miou'),
                    'acc':   t.user_attrs.get('wrong_acc'),
                    'map':   t.user_attrs.get('wrong_map'),
                    'auroc': t.user_attrs.get('wrong_auroc'),
                },
                'params': t.params,
            }
            for t in sorted_trials
        ],
    }

    try:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")
    except (PermissionError, OSError) as e:
        fallback = os.path.join(os.getcwd(), os.path.basename(args.output_json))
        print(f"[Warning] {e} – trying {fallback}")
        with open(fallback, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {fallback}")


if __name__ == '__main__':
    main()
