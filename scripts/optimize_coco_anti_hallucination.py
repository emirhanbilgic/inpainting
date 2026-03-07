#!/usr/bin/env python3
"""
GAP-based hyperparameter optimization for anti-hallucination segmentation on MS COCO.

Usage:
    python scripts/optimize_coco_anti_hallucination.py --n_trials 50
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import time
import sys
import math
import argparse
import json
import random
from typing import List
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

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

from legrad import LeWrapper, LePreprocess
import open_clip

# Use existing heatmap functions from baseline to save space
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
from sparse_encoding import (
    omp_sparse_residual,
    wordnet_neighbors_configured,
)

try:
    from daam_segmentation import DAAMSegmenter
except ImportError:
    DAAMSegmenter = None

try:
    from diffusers.models.attention_processor import Attention
    from daam.hook import UNetCrossAttentionLocator
    from daam.heatmap import RawHeatMapCollection
    from daam.utils import auto_autocast
except ImportError:
    pass

# ==============================================================================
# Helper functions for datasets
# ==============================================================================
COCO_CLASSES = [
    "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear",
    "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus",
    "cake", "car", "carrot", "cat", "cell phone", "chair", "clock", "couch", "cow", "cup",
    "dining table", "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee", "giraffe",
    "hair drier", "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop",
    "microwave", "motorcycle", "mouse", "orange", "oven", "parking meter", "person", "pizza",
    "potted plant", "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink",
    "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
    "surfboard", "teddy bear", "tennis racket", "tie", "toilet", "toothbrush", "traffic light",
    "train", "truck", "tv", "umbrella", "vase", "wine glass", "zebra"
]

def load_coco_metadata(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_mask(mask_path, target_size):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    arr = np.array(mask)
    return (arr > 128).astype(np.uint8)

BASELINES = {
    'CLIP': {
        'LeGrad': {
            'correct': {'miou': 59.40, 'acc': 86.68, 'map': 90.78, 'auroc': 87.45},
            'wrong': {'miou': 39.04, 'acc': 75.06, 'map': 77.00, 'auroc': 87.45}
        },
        'GradCAM': {
            'correct': {'miou': 46.98, 'acc': 75.45, 'map': 88.69, 'auroc': 77.82},
            'wrong': {'miou': 35.75, 'acc': 66.50, 'map': 81.91, 'auroc': 77.82}
        },
        'CheferCAM': {
            'correct': {'miou': 45.36, 'acc': 71.41, 'map': 90.52, 'auroc': 83.16},
            'wrong': {'miou': 40.61, 'acc': 67.56, 'map': 88.40, 'auroc': 83.16}
        },
        'AttentionCAM': {
            'correct': {'miou': 44.65, 'acc': 78.10, 'map': 85.13, 'auroc': 53.84},
            'wrong': {'miou': 40.24, 'acc': 74.66, 'map': 83.49, 'auroc': 53.84}
        }
    },
    'SigLIP': {
        'LeGrad': {
            'correct': {'miou': 47.30, 'acc': 85.16, 'map': 89.37, 'auroc': 77.81},
            'wrong': {'miou': 41.82, 'acc': 82.57, 'map': 80.21, 'auroc': 77.81}
        },
        'GradCAM': {
            'correct': {'miou': 37.69, 'acc': 61.73, 'map': 87.08, 'auroc': 69.47},
            'wrong': {'miou': 30.15, 'acc': 54.23, 'map': 80.45, 'auroc': 69.47}
        },
        'CheferCAM': {
            'correct': {'miou': 39.92, 'acc': 68.19, 'map': 87.33, 'auroc': 71.77},
            'wrong': {'miou': 37.86, 'acc': 66.39, 'map': 85.56, 'auroc': 71.77}
        },
        'AttentionCAM': {
            'correct': {'miou': 45.68, 'acc': 71.28, 'map': 90.16, 'auroc': 81.01},
            'wrong': {'miou': 33.50, 'acc': 60.86, 'map': 81.45, 'auroc': 81.01}
        }
    },
    'DAAM': {
        'DAAM': {
            'correct': {'miou': 43.58, 'acc': 66.72, 'map': 91.43, 'auroc': 86.47},
            'wrong': {'miou': 37.86, 'acc': 61.47, 'map': 87.10, 'auroc': 86.47}
        }
    }
}

# ==============================================================================
# True Key-Space OMP Components for DAAM
# ==============================================================================
class KeySpaceOMPProcessor:
    def __init__(self, target_token_indices, distractor_token_indices, beta=1.0, heat_maps=None, layer_idx=0, latent_hw=4096, context_size=77, parent_trace=None):
        self.target_token_indices = target_token_indices
        self.distractor_token_indices = distractor_token_indices
        self.beta = beta
        self.heat_maps = heat_maps
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw
        self.context_size = context_size
        self.parent_trace = parent_trace

    def _orthogonalize_keys(self, key, n_heads):
        key = key.clone()
        for target_idx in self.target_token_indices:
            target_key = key[:, target_idx, :]
            for dist_indices in self.distractor_token_indices:
                for dist_idx in dist_indices:
                    dist_key = key[:, dist_idx, :]
                    dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                    projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                    target_key = target_key - self.beta * projection
            key[:, target_idx, :] = target_key
        return key

    def _unravel_attn(self, x):
        factor = int(math.sqrt(self.latent_hw // x.shape[1]))
        if factor == 0: factor = 1
        hw = int(math.sqrt(x.shape[1]))
        maps = x.reshape(x.shape[0], hw, hw, x.shape[2]).permute(0, 3, 1, 2)
        return maps

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None: encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query, key, value = attn.head_to_batch_dim(query), attn.head_to_batch_dim(key), attn.head_to_batch_dim(value)
        
        if key.shape[1] == self.context_size and self.beta > 0:
            key = self._orthogonalize_keys(key, attn.heads)
            
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1])) if attention_probs.shape[1] > 0 else 8
        if self.parent_trace is not None: self.parent_trace._gen_idx += 1
        if self.heat_maps is not None and attention_probs.shape[-1] == self.context_size and factor != 8:
            maps = self._unravel_attn(attention_probs)
            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)
                
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.to_out[1](attn.to_out[0](attn.batch_to_head_dim(hidden_states)))
        return hidden_states

def get_token_indices(tokenizer, prompt: str, concept: str) -> List[int]:
    tokens = tokenizer.tokenize(prompt)
    concept_tokens = tokenizer.tokenize(concept)
    indices = []
    concept_len = len(concept_tokens)
    for i in range(len(tokens) - concept_len + 1):
        if tokens[i:i + concept_len] == concept_tokens:
            indices.extend(range(i + 1, i + 1 + concept_len))
    if not indices:
        for i, tok in enumerate(tokens):
            if concept.lower() in tok.lower().replace('</w>', ''):
                indices.append(i + 1)
    return indices

def run_daam_with_key_space_omp(segmenter, image_pil, target_concept, competing_concepts, beta=1.0, size=512):
    pipeline, tokenizer, text_encoder, vae, unet, scheduler, device = segmenter.pipeline, segmenter.tokenizer, segmenter.text_encoder, segmenter.vae, segmenter.unet, segmenter.scheduler, segmenter.device
    w, h = image_pil.size
    dtype = next(vae.parameters()).dtype
    
    img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0 * 2.0 - 1.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    
    with torch.no_grad(): latents = vae.encode(img_tensor).latent_dist.sample() * 0.18215
    
    all_concepts = [target_concept] + competing_concepts
    combined_prompt = f"a photo of a {', a '.join(all_concepts)}."
    
    context_size = tokenizer.model_max_length
    target_indices = [i for i in get_token_indices(tokenizer, combined_prompt, target_concept) if i < context_size]
    distractor_indices = [[i for i in get_token_indices(tokenizer, combined_prompt, comp) if i < context_size] for comp in competing_concepts]
    distractor_indices = [g for g in distractor_indices if g]
    
    if not target_indices: return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    text_input = tokenizer(combined_prompt, padding="max_length", max_length=context_size, truncation=True, return_tensors="pt")
    with torch.no_grad(): text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    scheduler.set_timesteps(50, device=device)
    noise = torch.randn_like(latents)
    timestep = torch.tensor([21], device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    
    heat_maps = RawHeatMapCollection()
    locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
    cross_attn_modules = locator.locate(unet)
    
    latent_hw = 4096
    class GenIdxTracker: _gen_idx = 0
    tracker = GenIdxTracker()
    
    original_processors = {}
    for idx, module in enumerate(cross_attn_modules):
        original_processors[idx] = module.processor
        module.set_processor(KeySpaceOMPProcessor(target_indices, distractor_indices, beta, heat_maps, idx, latent_hw, 77, tracker))
    
    try:
        with torch.no_grad(): _ = unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
    finally:
        for idx, module in enumerate(cross_attn_modules): module.set_processor(original_processors[idx])
    
    x, factors, all_merges = int(np.sqrt(latent_hw)), {0, 1, 2, 4, 8, 16, 32, 64}, []
    with auto_autocast(dtype=torch.float32):
        for (factor, layer, head), heat_map in heat_maps:
            if factor in factors and factor != 8:
                all_merges.append(F.interpolate(heat_map.unsqueeze(1), size=(x, x), mode='bicubic').clamp_(min=0))
                
    if not all_merges: return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    maps = torch.stack(all_merges, dim=0).mean(0)[:, 0]
    target_maps = [maps[tidx] for tidx in target_indices if tidx < maps.shape[0]]
    if not target_maps: return segmenter.predict(image_pil, f"a photo of a {target_concept}.", size=size)
    
    heatmap = torch.stack(target_maps).mean(0).unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False).squeeze()
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8).cpu()

# ==============================================================================
# Evaluator Class
# ==============================================================================
class CocoAntiHallucinationObjective:
    def __init__(
        self, model, tokenizer, preprocess, coco_dir, metadata, device,
        image_size=448, limit=0, composite_lambda=0.5, multi_objective=False, seed=42,
        method='legrad', gradcam_layer=8, chefercam_method='transformer_attribution',
        transformer_attribution_start_layer=1, threshold_mode='fixed', fixed_threshold=0.5,
        baseline_metrics=None, lrp_start_layer=1, use_daam_keyspace_omp=False, daam_model_id=None,
        use_llm_dictionary=False, llm_dictionary_path=None, use_gpt_dictionary=False, gpt_dictionary_path=None, negative_strategy='dynamic'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.image_size = image_size
        self.composite_lambda = composite_lambda
        self.multi_objective = multi_objective
        self.rng = random.Random(seed)
        self.method = method
        self.gradcam_layer = gradcam_layer
        self.chefercam_method = chefercam_method
        self.transformer_attribution_start_layer = transformer_attribution_start_layer
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.baseline_metrics = baseline_metrics
        self.lrp_start_layer = lrp_start_layer
        self.use_daam_keyspace_omp = use_daam_keyspace_omp
        self.negative_strategy = negative_strategy
        
        self.coco_dir = coco_dir
        self.image_dir = os.path.join(coco_dir, 'val2017')
        self.mask_dir = os.path.join(coco_dir, 'val2017_2_objects_masks')
        
        # Build ordered list of entries
        self.entries = list(metadata.values())
        if limit > 0:
            self.entries = self.entries[:limit]

        # DAAM
        self.daam_segmenter = None
        if self.method == 'daam':
            if DAAMSegmenter is None:
                raise ImportError("DAAMSegmenter not available.")
            print(f"[coco opt] Initializing DAAMSegmenter ({daam_model_id})...")
            self.daam_segmenter = DAAMSegmenter(model_id=daam_model_id, device=device)

        # Dictionary
        self.llm_dictionary = None
        self.gpt_dictionary = None
        
        if use_llm_dictionary:
            if llm_dictionary_path is None: 
                llm_dictionary_path = os.path.join(scripts_dir, 'visual_concept_dictionary_coco.json')
            
            if os.path.exists(llm_dictionary_path):
                print(f"[coco opt] Loading LLM dictionary from {llm_dictionary_path}...")
                with open(llm_dictionary_path, 'r') as f: 
                    self.llm_dictionary = json.load(f)
            else:
                print(f"[coco opt] Warning: LLM dictionary not found at {llm_dictionary_path}")

        if use_gpt_dictionary:
            if gpt_dictionary_path is None: 
                gpt_dictionary_path = os.path.join(scripts_dir, 'visual_concept_dictionary_coco_gpt.json')
            
            if os.path.exists(gpt_dictionary_path):
                print(f"[coco opt] Loading GPT dictionary from {gpt_dictionary_path}...")
                with open(gpt_dictionary_path, 'r') as f: 
                    self.gpt_dictionary = json.load(f)
            else:
                print(f"[coco opt] Warning: GPT dictionary not found at {gpt_dictionary_path}")

        # Text Embeddings
        all_objects = set()
        for e in self.entries:
            all_objects.update(e['objects'])
        self.unique_objects = sorted(all_objects)
        self.obj_to_idx = {o: i for i, o in enumerate(self.unique_objects)}
        prompts = [f"a photo of a {obj}." for obj in self.unique_objects]
        tok = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            self.all_text_embs = self.model.encode_text(tok, normalize=True)

        # Fixed Dictionary Embs
        self.fixed_dictionary_embs = None
        self.fixed_dictionary_words = None

        if self.negative_strategy in ['fix_dictionary', 'fix_dictionary_prompts_only']:
            all_words = list(COCO_CLASSES)
            if self.negative_strategy == 'fix_dictionary':
                print("[coco opt] Building fixed wordnet dictionary based on COCO classes...")
                seen_words = set(all_words)
                for c in COCO_CLASSES:
                    nbs = wordnet_neighbors_configured(
                        c, use_synonyms=False, use_hypernyms=True, use_hyponyms=True, use_siblings=True, limit_per_relation=8
                    )
                    for n in nbs:
                        if n not in seen_words:
                            all_words.append(n)
                            seen_words.add(n)
            
            print(f"[coco opt] Fixed dictionary size: {len(all_words)} words")
            self.fixed_dictionary_words = all_words

            # If not DAAM, compute CLIP embeddings
            if self.method != 'daam':
                all_tok = self.tokenizer([f"a photo of a {w}." for w in all_words]).to(self.device)
                with torch.no_grad():
                    embs = []
                    bs = 256
                    for i in range(0, len(all_tok), bs):
                        b_tok = all_tok[i:i+bs]
                        embs.append(self.model.encode_text(b_tok, normalize=True))
                    self.fixed_dictionary_embs = torch.cat(embs, dim=0)

            
    def _compute_heatmap(self, img_t, text_emb, base_img=None, target_concept=None, competing_concepts=None, omp_beta=1.0):
        if self.method == 'daam':
            prompt_text = f"a photo of a {target_concept}."
            if competing_concepts:
                if self.use_daam_keyspace_omp:
                    return run_daam_with_key_space_omp(self.daam_segmenter, base_img, target_concept, competing_concepts, omp_beta)
                else:
                    return self.daam_segmenter.predict_key_space_omp(base_img, prompt_text, target_concept, competing_concepts, size=512, omp_beta=omp_beta)
            return self.daam_segmenter.predict(base_img, prompt_text, size=512)
        elif self.method == 'lrp':
            return compute_lrp_heatmap(self.model, img_t, text_emb)
        elif self.method == 'chefercam':
            if self.chefercam_method == 'transformer_attribution':
                return compute_transformer_attribution(self.model, img_t, text_emb, start_layer=self.transformer_attribution_start_layer)
            return compute_chefercam(self.model, img_t, text_emb)
        elif self.method == 'gradcam':
            return compute_gradcam_heatmap(self.model, img_t, text_emb, layer_index=self.gradcam_layer)
        else:
            return compute_legrad_heatmap(self.model, img_t, text_emb)

    def evaluate_sparse_config(self, wn_flags, dict_include_prompts, sparse_threshold, atoms, max_dict_cos_sim, omp_beta):
        def _empty(): return {'inter': np.zeros(2), 'union': np.zeros(2), 'pixel_correct': 0, 'pixel_label': 0, 'ap': [], 'auroc': []}
        pos, neg = _empty(), _empty()

        for entry in self.entries:
            file_name, objects, chosen = entry['file_name'], entry['objects'], entry['chosen']
            other = [o for o in objects if o != chosen][0]
            
            img_path = os.path.join(self.image_dir, file_name)
            if not os.path.exists(img_path): continue
            base_img = Image.open(img_path).convert('RGB')
            img_t = self.preprocess(base_img).unsqueeze(0).to(self.device)

            mask_name = f"{file_name.replace('.jpg', '')}_{chosen.replace(' ', '_')}.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path): continue
            gt_mask = load_mask(mask_path, (self.image_size, self.image_size))
            H_gt, W_gt = gt_mask.shape

            def process_prompt(target_obj, competing_obj_for_daam=None):
                target_idx = self.obj_to_idx[target_obj]
                orig_emb = self.all_text_embs[target_idx:target_idx+1]
                
                # Dictionary Builder
                competing_daam, sparse_emb = [], orig_emb
                if self.method == 'daam':
                    if self.negative_strategy in ['fix_dictionary', 'fix_dictionary_prompts_only']:
                        competing_daam = [c for c in self.fixed_dictionary_words if c.lower() != target_obj.lower()]
                    elif self.gpt_dictionary and target_obj in self.gpt_dictionary:
                        gpt_dict = self.gpt_dictionary[target_obj]
                        competing_daam = gpt_dict.get('visual_confusers', []) + gpt_dict.get('co_occurring_context', []) + gpt_dict.get('semantic_hierarchy', [])
                    elif self.llm_dictionary and target_obj in self.llm_dictionary:
                        llm = self.llm_dictionary[target_obj]
                        competing_daam = llm.get('visual_confusers', []) + llm.get('co_occurring_context', []) + llm.get('semantic_hierarchy', [])
                    else:
                        if dict_include_prompts and competing_obj_for_daam: competing_daam.append(competing_obj_for_daam)
                        if any(wn_flags.values()):
                            nbs = wordnet_neighbors_configured(target_obj, **wn_flags, limit_per_relation=8)
                            competing_daam.extend([n for n in nbs if n.lower() != target_obj.lower()])
                    
                    seen, uniq = set(), []
                    for c in competing_daam:
                        if c.lower() not in seen and c.lower() != target_obj.lower(): seen.add(c.lower()); uniq.append(c)
                    if atoms > 0 and len(uniq) > atoms: uniq = uniq[:atoms]
                    competing_daam = uniq
                else:
                    parts = []
                    if self.negative_strategy in ['fix_dictionary', 'fix_dictionary_prompts_only']:
                        parts.append(self.fixed_dictionary_embs)
                    elif self.gpt_dictionary and target_obj in self.gpt_dictionary:
                        gpt_dict = self.gpt_dictionary[target_obj]
                        all_c = gpt_dict.get('visual_confusers', []) + gpt_dict.get('co_occurring_context', []) + gpt_dict.get('semantic_hierarchy', [])
                        if all_c:
                            c_tok = self.tokenizer([f"a photo of a {c}." for c in all_c]).to(self.device)
                            with torch.no_grad(): parts.append(F.normalize(self.model.encode_text(c_tok), dim=-1))
                    elif self.llm_dictionary and target_obj in self.llm_dictionary:
                        llm = self.llm_dictionary[target_obj]
                        all_c = llm.get('visual_confusers', []) + llm.get('co_occurring_context', []) + llm.get('semantic_hierarchy', [])
                        if all_c:
                            c_tok = self.tokenizer([f"a photo of a {c}." for c in all_c]).to(self.device)
                            with torch.no_grad(): parts.append(F.normalize(self.model.encode_text(c_tok), dim=-1))
                    else:
                        if dict_include_prompts and len(self.unique_objects) > 1:
                            if target_idx > 0: parts.append(self.all_text_embs[:target_idx])
                            if target_idx + 1 < len(self.unique_objects): parts.append(self.all_text_embs[target_idx+1:])
                        if any(wn_flags.values()):
                            nbs = wordnet_neighbors_configured(target_obj, **wn_flags, limit_per_relation=8)
                            if nbs:
                                n_tok = self.tokenizer([f"a photo of a {w}." for w in nbs]).to(self.device)
                                with torch.no_grad(): parts.append(F.normalize(self.model.encode_text(n_tok), dim=-1))
                    
                    if parts:
                        D = F.normalize(torch.cat(parts, dim=0), dim=-1)
                        if 0.0 < max_dict_cos_sim < 1.0:
                            D = D[(D @ orig_emb.t()).squeeze(-1).abs() < max_dict_cos_sim]
                        sparse_emb = omp_sparse_residual(orig_emb, D, max_atoms=atoms)

                heatmap = self._compute_heatmap(img_t, sparse_emb, base_img, target_obj, competing_daam, omp_beta)
                
                heatmap_resized = F.interpolate(heatmap.view(1, 1, heatmap.shape[-2], heatmap.shape[-1]), size=(H_gt, W_gt), mode='bilinear', align_corners=False).squeeze()
                heatmap_norm = heatmap_resized if self.method == 'legrad' else (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                
                thr = sparse_threshold if self.method == 'legrad' or self.threshold_mode == 'fixed' else heatmap_norm.mean().item()
                Res_1, Res_0 = (heatmap_norm > thr).float(), (heatmap_norm <= thr).float()
                out_tensor = torch.stack([Res_0, Res_1], dim=0)
                out_AP = torch.stack([1.0 - heatmap_norm, heatmap_norm], dim=0)
                gt = torch.from_numpy(gt_mask).long()
                
                inter, union = batch_intersection_union(out_tensor, gt, nclass=2)
                corr, lab = batch_pix_accuracy(out_tensor, gt)
                ap_val = get_ap_scores(out_AP, gt)
                return inter, union, corr, lab, ap_val[0] if ap_val else 0.0, heatmap_norm.cpu().numpy()

            try:
                # Positive
                i_p, u_p, c_p, l_p, ap_p, hm_p = process_prompt(chosen, other)
                pos['inter'] += i_p
                pos['union'] += u_p
                pos['pixel_correct'] += c_p
                pos['pixel_label'] += l_p
                pos['ap'].append(ap_p)

                # Negative
                i_n, u_n, c_n, l_n, ap_n, hm_n = process_prompt(other, chosen)
                neg['inter'] += i_n
                neg['union'] += u_n
                neg['pixel_correct'] += c_n
                neg['pixel_label'] += l_n
                neg['ap'].append(ap_n)

                # Paired AUROC
                gt_b = (gt_mask > 0).astype(int).flatten()
                paired_gt = np.concatenate([gt_b, np.zeros_like(gt_b)])
                paired_pred = np.concatenate([hm_p.flatten(), hm_n.flatten()])
                if len(np.unique(paired_gt)) > 1:
                    auc = roc_auc_score(paired_gt, paired_pred)
                    pos['auroc'].append(auc); neg['auroc'].append(auc)
            except Exception as e:
                print(f"Error processing image {file_name}: {e}")
                pass
                
        def _metrics(d):
            iou = d['inter'].astype(np.float64) / (d['union'].astype(np.float64) + 1e-10)
            auroc_l = [a for a in d['auroc'] if not np.isnan(a)]
            return {
                'miou': float(100.0 * iou.mean()),
                'acc': float(100.0 * d['pixel_correct'] / (d['pixel_label'] + 1e-10)),
                'map': float(np.mean(d['ap']) * 100 if d['ap'] else 0.0),
                'auroc': float(np.mean(auroc_l) * 100 if auroc_l else 0.0)
            }
        
        return _metrics(pos), _metrics(neg)

    def __call__(self, trial: optuna.Trial):
        if self.use_llm_dictionary or self.gpt_dictionary:
            wn_flags = {'use_synonyms': False, 'use_hypernyms': False, 'use_hyponyms': False, 'use_siblings': False}
            dict_include_prompts = False
        elif self.negative_strategy == 'fix_dictionary':
            wn_flags = {'use_synonyms': True, 'use_hypernyms': True, 'use_hyponyms': True, 'use_siblings': False}
            dict_include_prompts = True
        else:
            wn_flags = {
                'use_synonyms': trial.suggest_categorical('use_synonyms', [True, False]),
                'use_hypernyms': trial.suggest_categorical('use_hypernyms', [True, False]),
                'use_hyponyms': trial.suggest_categorical('use_hyponyms', [True, False]),
                'use_siblings': trial.suggest_categorical('use_siblings', [True, False])
            }
            dict_include_prompts = trial.suggest_categorical('dict_include_prompts', [True, False])
            if not any(list(wn_flags.values()) + [dict_include_prompts]):
                raise optuna.TrialPruned()

        sparse_threshold = trial.suggest_float('sparse_threshold', 0.1, 0.9, step=0.025) if self.threshold_mode == 'fixed' or self.method == 'legrad' else 0.5
        atoms = trial.suggest_int('atoms', 1, 32)
        max_dict_cos_sim = trial.suggest_float('max_dict_cos_sim', 0.5, 1.0, step=0.05)
        omp_beta = trial.suggest_float('omp_beta', 0.0, 2.0, step=0.1) if self.method == 'daam' else 1.0

        pos_m, neg_m = self.evaluate_sparse_config(wn_flags, dict_include_prompts, sparse_threshold, atoms, max_dict_cos_sim, omp_beta)
        
        for k in ['miou', 'acc', 'map', 'auroc']:
            trial.set_user_attr(f'correct_{k}', pos_m[k])
            trial.set_user_attr(f'wrong_{k}', neg_m[k])

        if self.baseline_metrics:
            bc, bw = self.baseline_metrics['correct'], self.baseline_metrics['wrong']
            delta_c_miou = pos_m['miou'] - bc['miou']
            delta_c_acc = pos_m['acc'] - bc['acc']
            delta_c_map = pos_m['map'] - bc['map']
            delta_c_auroc = pos_m['auroc'] - bc['auroc']
            
            delta_w_miou = bw['miou'] - neg_m['miou']
            delta_w_acc = bw['acc'] - neg_m['acc']
            delta_w_map = bw['map'] - neg_m['map']
            delta_w_auroc = bw['auroc'] - neg_m['auroc']

            gap_baseline_miou = bc['miou'] - bw['miou']
            gap_baseline_acc = bc['acc'] - bw['acc']
            gap_baseline_map = bc['map'] - bw['map']

            gap_current_miou = pos_m['miou'] - neg_m['miou']
            gap_current_acc = pos_m['acc'] - neg_m['acc']
            gap_current_map = pos_m['map'] - neg_m['map']

            gap_imprv_miou = gap_current_miou - gap_baseline_miou
            gap_imprv_acc = gap_current_acc - gap_baseline_acc
            gap_imprv_map = gap_current_map - gap_baseline_map
            gap_imprv_auroc = pos_m['auroc'] - bc['auroc']
            
            trial.set_user_attr('delta_correct_miou', delta_c_miou)
            trial.set_user_attr('delta_correct_acc', delta_c_acc)
            trial.set_user_attr('delta_correct_map', delta_c_map)
            trial.set_user_attr('delta_correct_auroc', delta_c_auroc)

            trial.set_user_attr('delta_wrong_miou', delta_w_miou)
            trial.set_user_attr('delta_wrong_acc', delta_w_acc)
            trial.set_user_attr('delta_wrong_map', delta_w_map)
            trial.set_user_attr('delta_wrong_auroc', delta_w_auroc)

            trial.set_user_attr('gap_improvement_miou', gap_imprv_miou)
            trial.set_user_attr('gap_improvement_acc', gap_imprv_acc)
            trial.set_user_attr('gap_improvement_map', gap_imprv_map)
            trial.set_user_attr('gap_improvement_auroc', gap_imprv_auroc)

            # Default logic is direct gap
            comp_miou, comp_acc, comp_map, comp_auroc = gap_imprv_miou, gap_imprv_acc, gap_imprv_map, gap_imprv_auroc
            if self.composite_lambda != 0.5:
                comp_miou += self.composite_lambda * delta_c_miou
                comp_acc += self.composite_lambda * delta_c_acc
                comp_map += self.composite_lambda * delta_c_map
                comp_auroc += self.composite_lambda * delta_c_auroc
        else:
            comp_miou = pos_m['miou'] - neg_m['miou']
            comp_acc = pos_m['acc'] - neg_m['acc']
            comp_map = pos_m['map'] - neg_m['map']
            comp_auroc = 0.0

        composite = comp_miou + comp_acc + comp_map + comp_auroc
        trial.set_user_attr('composite_score', composite)
        trial.report(composite, step=0)
        if trial.should_prune(): raise optuna.TrialPruned()

        if self.multi_objective:
            return pos_m['miou'], pos_m['acc'], pos_m['map'], pos_m['auroc'], neg_m['miou'], neg_m['acc'], neg_m['map'], neg_m['auroc']
        return composite

# ==============================================================================
# main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='COCO Anti-Hallucination Optimization')
    parser.add_argument('--coco_dir', type=str, default=os.path.join(project_root, 'ms_coco_2017'))
    parser.add_argument('--metadata_json', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--use_siglip', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--image_size', type=int, default=448)

    parser.add_argument('--use_gradcam', action='store_true')
    parser.add_argument('--gradcam_layer', type=int, default=8)
    parser.add_argument('--use_chefercam', action='store_true')
    parser.add_argument('--chefercam_method', type=str, default='transformer_attribution')
    parser.add_argument('--transformer_attribution_start_layer', type=int, default=1)
    parser.add_argument('--use_attentioncam', action='store_true')
    parser.add_argument('--lrp_start_layer', type=int, default=1)
    parser.add_argument('--use_daam', action='store_true')
    parser.add_argument('--use_daam_keyspace_omp', action='store_true')
    parser.add_argument('--daam_model_id', type=str, default='Manojb/stable-diffusion-2-base')

    parser.add_argument('--threshold_mode', type=str, default='fixed', choices=['mean', 'fixed'])
    parser.add_argument('--fixed_threshold', type=float, default=0.5)

    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--study_name', type=str, default='coco_anti_hallucination')
    parser.add_argument('--multi_objective', action='store_true')
    parser.add_argument('--composite_lambda', type=float, default=0.5)
    parser.add_argument('--use_llm_dictionary', action='store_true')
    parser.add_argument('--llm_dictionary_path', type=str, default=None)
    parser.add_argument('--use_gpt_dictionary', action='store_true')
    parser.add_argument('--gpt_dictionary_path', type=str, default=None)
    parser.add_argument('--negative_strategy', type=str, default='dynamic', choices=['dynamic', 'fix_dictionary', 'fix_dictionary_prompts_only'])
    parser.add_argument('--output_json', type=str, default='coco_anti_hallucination_results.json')
    args = parser.parse_args()

    if args.metadata_json is None: args.metadata_json = os.path.join(args.coco_dir, 'selected_images_2_unique.json')

    if args.use_daam:
        model_type_key, method_key = 'DAAM', 'DAAM'
    else:
        model_type_key = 'SigLIP' if args.use_siglip else 'CLIP'
        if args.use_attentioncam: method_key = 'AttentionCAM'
        elif args.use_chefercam: method_key = 'CheferCAM'
        elif args.use_gradcam: method_key = 'GradCAM'
        else: method_key = 'LeGrad'

    baseline_metrics = BASELINES[model_type_key][method_key]
    print(f"Using {model_type_key} - {method_key} baseline.")

    if args.use_siglip:
        args.model_name, args.pretrained = args.model_name or 'ViT-B-16-SigLIP', args.pretrained or 'webli'
    else:
        args.model_name, args.pretrained = args.model_name or 'ViT-B-16', args.pretrained or 'laion2b_s34b_b88k'

    print(f"Loading {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    metadata = load_coco_metadata(args.metadata_json)

    objective = CocoAntiHallucinationObjective(
        model=model, tokenizer=tokenizer, preprocess=preprocess, coco_dir=args.coco_dir,
        metadata=metadata, device=args.device, image_size=args.image_size, limit=args.limit,
        composite_lambda=args.composite_lambda, multi_objective=args.multi_objective, seed=args.seed,
        method=method_key.lower(), gradcam_layer=args.gradcam_layer, chefercam_method=args.chefercam_method,
        transformer_attribution_start_layer=args.transformer_attribution_start_layer,
        threshold_mode=args.threshold_mode, fixed_threshold=args.fixed_threshold,
        baseline_metrics=baseline_metrics, lrp_start_layer=args.lrp_start_layer,
        use_daam_keyspace_omp=args.use_daam_keyspace_omp, daam_model_id=args.daam_model_id,
        use_llm_dictionary=args.use_llm_dictionary, llm_dictionary_path=args.llm_dictionary_path,
        use_gpt_dictionary=args.use_gpt_dictionary, gpt_dictionary_path=args.gpt_dictionary_path,
        negative_strategy=args.negative_strategy
    )

    if args.use_daam and args.n_jobs > 1:
        args.n_jobs = 1

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize' if not args.multi_objective else None,
        directions=['maximize', 'maximize', 'maximize', 'maximize', 'minimize', 'minimize', 'minimize', 'minimize'] if args.multi_objective else None,
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed) if args.multi_objective else optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10) if not args.multi_objective else None
    )

    def trial_callback(study, trial):
        if trial.state != TrialState.COMPLETE: return
        
        c_miou = trial.user_attrs.get('correct_miou', 0)
        c_acc = trial.user_attrs.get('correct_acc', 0)
        c_map = trial.user_attrs.get('correct_map', 0)
        c_auroc = trial.user_attrs.get('correct_auroc', 0)
        
        w_miou = trial.user_attrs.get('wrong_miou', 0)
        w_acc = trial.user_attrs.get('wrong_acc', 0)
        w_map = trial.user_attrs.get('wrong_map', 0)
        w_auroc = trial.user_attrs.get('wrong_auroc', 0)
        
        dc_miou = trial.user_attrs.get('delta_correct_miou', 0)
        dc_acc = trial.user_attrs.get('delta_correct_acc', 0)
        dc_map = trial.user_attrs.get('delta_correct_map', 0)
        dc_auroc = trial.user_attrs.get('delta_correct_auroc', 0)
        
        dw_miou = trial.user_attrs.get('delta_wrong_miou', 0)
        dw_acc = trial.user_attrs.get('delta_wrong_acc', 0)
        dw_map = trial.user_attrs.get('delta_wrong_map', 0)
        dw_auroc = trial.user_attrs.get('delta_wrong_auroc', 0)
        
        composite = trial.user_attrs.get('composite_score', trial.value)
        
        gap_imp_miou = trial.user_attrs.get('gap_improvement_miou', 0)
        gap_imp_acc = trial.user_attrs.get('gap_improvement_acc', 0)
        gap_imp_map = trial.user_attrs.get('gap_improvement_map', 0)
        gap_imp_auroc = trial.user_attrs.get('gap_improvement_auroc', 0)

        print(f"\n{'─'*70}")
        print(f"  Trial {trial.number} │ Score={composite:.2f}")
        print(f"{'─'*70}")
        print(f"  {'':>10} {'mIoU':>8} {'Acc':>8} {'mAP':>8} {'AUROC':>8}")
        print(f"  {'Correct':>10} {c_miou:>7.2f}% {c_acc:>7.2f}% {c_map:>7.2f}% {c_auroc:>7.2f}%")
        print(f"  {'  Δbase':>10} {dc_miou:>+7.2f}  {dc_acc:>+7.2f}  {dc_map:>+7.2f}  {dc_auroc:>+7.2f}")
        print(f"  {'Wrong':>10} {w_miou:>7.2f}% {w_acc:>7.2f}% {w_map:>7.2f}% {w_auroc:>7.2f}%")
        print(f"  {'  Δbase':>10} {dw_miou:>+7.2f}  {dw_acc:>+7.2f}  {dw_map:>+7.2f}  {dw_auroc:>+7.2f}")
        print(f"  {'Gap Imprv':>10} {gap_imp_miou:>+7.2f}  {gap_imp_acc:>+7.2f}  {gap_imp_map:>+7.2f}  {gap_imp_auroc:>+7.2f}")
        
        if not args.multi_objective:
            best = study.best_trial
            b_comp = best.user_attrs.get('composite_score', best.value)
            print(f"  Best: Trial {best.number} (Score={b_comp:.2f})")

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, callbacks=[trial_callback])

    results = {
        'best_params': study.best_trial.params if not args.multi_objective else None,
        'best_composite': study.best_trial.value if not args.multi_objective else None,
        'gap_improvement_miou': study.best_trial.user_attrs.get('gap_improvement_miou') if not args.multi_objective else None,
    }
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

if __name__ == '__main__':
    main()
