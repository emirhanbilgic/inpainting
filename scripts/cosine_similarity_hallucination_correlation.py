#!/usr/bin/env python3
"""
Cosine Similarity vs mIoU Correlation Analysis for Hallucination.

Validates the paper's claim: high cosine similarity between text embeddings
of different classes causes hallucination in XAI attribution maps.

For each image in ImageNet-Segmentation:
  1. Get the actual class embedding
  2. Sample N negative (wrong) class embeddings  
  3. Compute cosine similarity between each negative embedding and the actual class
  4. Compute mIoU of the negative prompt's heatmap against the GT segmentation
  5. Collect (cosine_similarity, mIoU) pairs

Then compute Pearson & Spearman correlations and generate a scatter plot.

Supports ALL methods: LeGrad, GradCAM, CheferCAM, AttentionCAM, DAAM.
Use --all_methods to run all CLIP-based methods, or --use_daam for DAAM.
"""

import sys
import os
import argparse
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root and scripts dir to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess
import open_clip

from benchmark_segmentation_v2 import (
    load_imagenet_class_index,
    build_wnid_to_label_map,
    get_synset_name,
    batch_intersection_union,
    get_ap_scores,
    batch_pix_accuracy,
)

from sparse_encoding import compute_map_for_embedding

# Import additional heatmap methods from optimize_anti_hallucination
from optimize_anti_hallucination import (
    compute_gradcam_heatmap,
    compute_lrp_heatmap,      # AttentionCAM
    compute_chefercam,         # CheferCAM
)

# Import DAAM segmenter (optional — only needed with --use_daam)
try:
    from daam_segmentation import DAAMSegmenter
    from diffusers import StableDiffusionPipeline as _SDPipeline
    from daam import trace as _daam_trace

    class Float32DAAMSegmenter(DAAMSegmenter):
        """DAAMSegmenter that uses float32 to avoid NaN on MPS."""
        def __init__(self, model_id="Manojb/stable-diffusion-2-base", device='mps'):
            if _SDPipeline is None:
                raise ImportError("Please install 'daam' and 'diffusers'")
            print(f"[DAAM-F32] Loading Stable Diffusion pipeline in float32: {model_id}...")
            self.device = device
            self.pipeline = _SDPipeline.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(device)
            self.pipeline.enable_attention_slicing()
            self.vae = self.pipeline.vae
            self.tokenizer = self.pipeline.tokenizer
            self.text_encoder = self.pipeline.text_encoder
            self.unet = self.pipeline.unet
            self.scheduler = self.pipeline.scheduler
            print("[DAAM-F32] Pipeline loaded (float32).")

        def predict(self, image_pil, prompt, size=512):
            """Override predict to use float32 tensors."""
            if self.device == 'mps':
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()

            w, h = image_pil.size
            img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
            img_arr = np.array(img_resized).astype(np.float32) / 255.0
            img_arr = img_arr * 2.0 - 1.0
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).float()

            with torch.no_grad():
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                latents = latents * 0.18215

            self.scheduler.set_timesteps(50, device=self.device)
            noise = torch.randn_like(latents)
            timestep = torch.tensor([21], device=self.device)
            noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

            concept = ""
            if prompt.startswith("a photo of a "):
                concept = prompt[len("a photo of a "):].strip(".").strip()
            elif prompt.startswith("a "):
                concept = prompt[2:].strip(".").strip()
            if not concept:
                concept = prompt.strip(".").strip()

            background_concepts = ["background", "floor", "tree", "person", "grass", "face"]
            background_str = ", ".join([f"a {bc}" for bc in background_concepts])
            augmented_prompt = f"{prompt}, a {concept}, {background_str}"

            text_input = self.tokenizer(
                prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            prompt_embeds = text_embeddings
            latent_model_input = noisy_latents

            with _daam_trace(self.pipeline) as tc:
                with torch.no_grad():
                    _ = self.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample
                global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)

                heatmap = None
                try:
                    word_heat_map = global_heat_map.compute_word_heat_map(concept)
                    heatmap = word_heat_map.heatmap
                except Exception:
                    sub_words = concept.split()
                    sub_heatmaps = []
                    for sw in sub_words:
                        try:
                            whm = global_heat_map.compute_word_heat_map(sw).heatmap
                            sub_heatmaps.append(whm)
                        except Exception:
                            pass
                    if sub_heatmaps:
                        heatmap = torch.stack(sub_heatmaps).mean(0)

                if heatmap is None and hasattr(global_heat_map, 'heat_maps'):
                    if global_heat_map.heat_maps.shape[0] > 6:
                        heatmap = global_heat_map.heat_maps[5:-1].mean(0)
                    else:
                        heatmap = global_heat_map.heat_maps.mean(0)

                if heatmap is None:
                    heatmap = torch.zeros((h, w))

            heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
            heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap.cpu()

except ImportError:
    DAAMSegmenter = None
    Float32DAAMSegmenter = None


def compute_heatmap(method_name, model, img_t, text_emb_1x, gradcam_layer=8):
    """Compute heatmap using the specified method."""
    if method_name == 'LeGrad':
        return compute_map_for_embedding(model, img_t, text_emb_1x)
    elif method_name == 'GradCAM':
        return compute_gradcam_heatmap(model, img_t, text_emb_1x, layer_index=gradcam_layer)
    elif method_name == 'CheferCAM':
        return compute_chefercam(model, img_t, text_emb_1x)
    elif method_name == 'AttentionCAM':
        return compute_lrp_heatmap(model, img_t, text_emb_1x)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def resize_heatmap(heatmap, gt_mask, needs_renorm=False):
    """Resize heatmap to GT size and optionally re-normalize."""
    H_gt, W_gt = gt_mask.shape
    H_hm, W_hm = heatmap.shape[-2], heatmap.shape[-1]
    heatmap_resized = F.interpolate(
        heatmap.view(1, 1, H_hm, W_hm),
        size=(H_gt, W_gt),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    if needs_renorm:
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    return heatmap_resized


def compute_per_image_miou(heatmap, gt_mask, threshold=0.5, needs_renorm=False):
    """
    Compute per-image mIoU from a heatmap and GT segmentation mask.
    Returns mIoU as a float in [0, 100].
    """
    heatmap_norm = resize_heatmap(heatmap, gt_mask, needs_renorm)

    Res_1 = (heatmap_norm > threshold).float()
    Res_0 = (heatmap_norm <= threshold).float()
    output = torch.stack([Res_0, Res_1], dim=0)
    gt_tensor = torch.from_numpy(gt_mask).long()
    inter, union = batch_intersection_union(output, gt_tensor, nclass=2)
    iou = inter.astype(np.float64) / (union.astype(np.float64) + 1e-10)
    return 100.0 * iou.mean()


def compute_heatmap_mass(heatmap, gt_mask, needs_renorm=False):
    """
    Continuous hallucination metric: fraction of heatmap energy inside the GT region.
    Returns value in [0, 100]. Higher = more hallucination (wrong prompt sees the object).
    """
    heatmap_norm = resize_heatmap(heatmap, gt_mask, needs_renorm)
    gt_binary = torch.from_numpy((gt_mask > 0).astype(np.float32))
    total_mass = heatmap_norm.sum().item()
    if total_mass < 1e-10:
        return 0.0
    mass_in_gt = (heatmap_norm * gt_binary).sum().item()
    return 100.0 * mass_in_gt / total_mass


def stratified_sample_negatives(cos_sim_row, cls_idx, num_negatives, n_bins=10, rng=None):
    """
    Sample negatives uniformly across cosine similarity bins.
    cos_sim_row: 1D array of cosine similarities of this class to all others.
    Returns list of indices (excluding cls_idx).
    """
    all_indices = [i for i in range(len(cos_sim_row)) if i != cls_idx]
    sims = np.array([cos_sim_row[i] for i in all_indices])
    
    bin_edges = np.linspace(sims.min(), sims.max() + 1e-8, n_bins + 1)
    bins = [[] for _ in range(n_bins)]
    for idx, sim in zip(all_indices, sims):
        b = min(int((sim - bin_edges[0]) / (bin_edges[1] - bin_edges[0])), n_bins - 1)
        bins[b].append(idx)
    
    # Round-robin sample from non-empty bins
    selected = []
    non_empty = [b for b in bins if b]
    if not non_empty:
        return []
    bi = 0
    while len(selected) < num_negatives and non_empty:
        bucket = non_empty[bi % len(non_empty)]
        pick = rng.choice(bucket) if rng else random.choice(bucket)
        if pick not in selected:
            selected.append(pick)
        bi += 1
        # Safety: prevent infinite loop if not enough unique items
        if bi > num_negatives * 10:
            break
    return selected


def run_single_method(method_name, model, preprocess, all_text_embs, cos_sim_matrix,
                      wnids_in_seg, unique_wnids, wnid_to_classname, wnid_to_idx,
                      idx_to_wnid, f, imgs_refs, gts_refs, limit, num_negatives,
                      threshold, rng, device, image_size, gradcam_layer=8,
                      stratified=False, use_heatmap_mass=False):
    """Run correlation analysis for a single CLIP-based method. Returns (cosine_sims, scores)."""
    cosine_sims = []
    scores = []
    
    needs_renorm = (method_name != 'LeGrad')
    metric_fn = compute_heatmap_mass if use_heatmap_mass else compute_per_image_miou

    for idx in tqdm(range(limit), desc=f"  {method_name}"):
        try:
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            img_t = preprocess(base_img).unsqueeze(0).to(device)

            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0, 0]
                real_gt = np.array(f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)

            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            gt_pil = transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST,
            )(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.int32)

            wnid = wnids_in_seg[idx]
            cls_idx = wnid_to_idx[wnid]

            # Sampling: stratified or random
            if stratified:
                neg_indices = stratified_sample_negatives(
                    cos_sim_matrix[cls_idx], cls_idx, num_negatives, n_bins=10, rng=rng
                )
            else:
                all_indices = list(range(len(unique_wnids)))
                all_indices.remove(cls_idx)
                neg_indices = rng.sample(all_indices, min(num_negatives, len(all_indices)))

            for neg_idx in neg_indices:
                neg_emb = all_text_embs[neg_idx:neg_idx + 1]
                cos_sim = cos_sim_matrix[neg_idx, cls_idx]

                heatmap = compute_heatmap(method_name, model, img_t, neg_emb, gradcam_layer)
                score = metric_fn(heatmap, gt_mask, threshold=threshold, needs_renorm=needs_renorm) \
                    if not use_heatmap_mass else metric_fn(heatmap, gt_mask, needs_renorm=needs_renorm)

                cosine_sims.append(cos_sim)
                scores.append(score)

        except Exception as e:
            print(f"  Error at idx {idx} ({method_name}): {e}")
            continue

    return np.array(cosine_sims), np.array(scores)


def run_daam_method(daam_segmenter, wnids_in_seg, unique_wnids, wnid_to_classname,
                    wnid_to_idx, idx_to_wnid, f, imgs_refs, gts_refs, limit,
                    num_negatives, threshold, rng, image_size,
                    stratified=False, use_heatmap_mass=False):
    """Run correlation analysis for DAAM. Uses SD's own text encoder for cosine similarity."""
    device = daam_segmenter.device
    text_encoder = daam_segmenter.text_encoder
    sd_tokenizer = daam_segmenter.tokenizer

    print("  Precomputing SD text encoder embeddings for cosine similarity...")
    all_prompts = [f"a photo of a {wnid_to_classname[w]}." for w in unique_wnids]
    sd_embs = []
    with torch.no_grad():
        for prompt in all_prompts:
            tok = sd_tokenizer(
                prompt, padding="max_length",
                max_length=sd_tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            enc_out = text_encoder(tok.input_ids.to(device))
            if hasattr(enc_out, 'pooler_output') and enc_out.pooler_output is not None:
                emb = enc_out.pooler_output
            else:
                emb = enc_out[0].mean(dim=1)
            sd_embs.append(F.normalize(emb, dim=-1))
    sd_embs = torch.cat(sd_embs, dim=0)
    cos_sim_matrix = (sd_embs @ sd_embs.T).cpu().numpy()
    print(f"  SD embeddings: {sd_embs.shape}, cos_sim range: [{cos_sim_matrix.min():.3f}, {cos_sim_matrix.max():.3f}]")

    metric_fn = compute_heatmap_mass if use_heatmap_mass else compute_per_image_miou
    cosine_sims = []
    scores = []

    for idx in tqdm(range(limit), desc="  DAAM"):
        try:
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref])
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)

            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0, 0]
                real_gt = np.array(f[real_gt_ref])
                gt_mask = real_gt.transpose(1, 0)
            else:
                gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)

            gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
            gt_pil = transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST,
            )(gt_pil)
            gt_mask = np.array(gt_pil).astype(np.int32)

            wnid = wnids_in_seg[idx]
            cls_idx = wnid_to_idx[wnid]

            if stratified:
                neg_indices = stratified_sample_negatives(
                    cos_sim_matrix[cls_idx], cls_idx, num_negatives, n_bins=10, rng=rng
                )
            else:
                all_indices = list(range(len(unique_wnids)))
                all_indices.remove(cls_idx)
                neg_indices = rng.sample(all_indices, min(num_negatives, len(all_indices)))

            for neg_idx in neg_indices:
                neg_class_name = wnid_to_classname[idx_to_wnid[neg_idx]]
                neg_prompt = f"a photo of a {neg_class_name}."
                cos_sim = cos_sim_matrix[neg_idx, cls_idx]

                heatmap = daam_segmenter.predict(base_img, neg_prompt, size=512)
                score = metric_fn(heatmap, gt_mask, threshold=threshold, needs_renorm=True) \
                    if not use_heatmap_mass else metric_fn(heatmap, gt_mask, needs_renorm=True)

                cosine_sims.append(cos_sim)
                scores.append(score)

        except Exception as e:
            print(f"  Error at idx {idx} (DAAM): {e}")
            continue

    return np.array(cosine_sims), np.array(scores)


def main():
    parser = argparse.ArgumentParser(
        description='Cosine Similarity vs mIoU Hallucination Correlation Analysis'
    )
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0=all)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--use_siglip', action='store_true', help='Use SigLIP instead of CLIP')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--class_index_path', type=str, default='resources/imagenet_class_index.json')
    parser.add_argument('--num_negatives', type=int, default=3, help='Negative prompts per image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradcam_layer', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='scripts/outputs')
    parser.add_argument('--output_name', type=str, default='cosine_vs_miou_correlation')
    
    # Method selection
    parser.add_argument('--all_methods', action='store_true', help='Run all CLIP-based methods (LeGrad, GradCAM, CheferCAM, AttentionCAM)')
    parser.add_argument('--use_gradcam', action='store_true', help='Use GradCAM instead of LeGrad')
    parser.add_argument('--use_chefercam', action='store_true', help='Use CheferCAM')
    parser.add_argument('--use_attentioncam', action='store_true', help='Use AttentionCAM')
    parser.add_argument('--use_daam', action='store_true', help='Include DAAM (Stable Diffusion)')
    parser.add_argument('--daam_model_id', type=str, default='Manojb/stable-diffusion-2-base')
    
    # Correlation boosting
    parser.add_argument('--stratified', action='store_true', help='Stratified cosine similarity sampling')
    parser.add_argument('--use_heatmap_mass', action='store_true', help='Use continuous heatmap mass metric instead of mIoU')
    parser.add_argument('--paired', action='store_true', help='Paired within-image analysis (closest vs farthest negative)')

    args = parser.parse_args()

    # ---- Determine methods to run ----
    # CLIP-based methods
    if args.all_methods:
        clip_methods = ['LeGrad', 'GradCAM', 'CheferCAM', 'AttentionCAM']
        run_daam = True  # --all_methods includes DAAM
    else:
        clip_methods = []
        if args.use_gradcam:
            clip_methods.append('GradCAM')
        if args.use_chefercam:
            clip_methods.append('CheferCAM')
        if args.use_attentioncam:
            clip_methods.append('AttentionCAM')
        if not clip_methods and not args.use_daam:
            clip_methods.append('LeGrad')
        run_daam = args.use_daam
    
    methods = clip_methods + (['DAAM'] if run_daam else [])

    # ---- Model setup ----
    if args.use_siglip:
        model_name = args.model_name or 'ViT-B-16-SigLIP'
        pretrained = args.pretrained or 'webli'
        model_type = 'SigLIP'
    else:
        model_name = args.model_name or 'ViT-B-16'
        pretrained = args.pretrained or 'laion2b_s34b_b88k'
        model_type = 'CLIP'

    model = None
    preprocess = None
    tokenizer = None
    if clip_methods:
        print(f"Loading {model_type} model: {model_name} ({pretrained})...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained, device=args.device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        model = LeWrapper(model, layer_index=-2)
        preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # ---- DAAM setup ----
    daam_segmenter = None
    if run_daam:
        if DAAMSegmenter is None:
            raise ImportError("DAAMSegmenter not available. Install daam and diffusers.")
        # Use Float32 variant on MPS to avoid NaN
        if args.device == 'mps' and Float32DAAMSegmenter is not None:
            print(f"Loading DAAM pipeline (float32 for MPS): {args.daam_model_id}...")
            daam_segmenter = Float32DAAMSegmenter(model_id=args.daam_model_id, device=args.device)
        else:
            print(f"Loading DAAM pipeline: {args.daam_model_id}...")
            daam_segmenter = DAAMSegmenter(model_id=args.daam_model_id, device=args.device)

    # ---- Class index ----
    try:
        class_index = load_imagenet_class_index(args.class_index_path)
        wnid_to_label = build_wnid_to_label_map(class_index)
        print(f"Loaded class index with {len(wnid_to_label)} wnids.")
    except Exception as e:
        print(f"Warning: failed to load class index ({e})")
        wnid_to_label = {}

    # ---- Dataset ----
    print(f"Loading dataset from {args.mat_file}...")
    f = h5py.File(args.mat_file, 'r')
    imgs_refs = f['value/img']
    gts_refs = f['value/gt']
    targets_refs = f['value/target']
    num_images = imgs_refs.shape[0]

    limit = args.limit if args.limit > 0 else num_images
    limit = min(limit, num_images)

    # ---- Precompute class embeddings ----
    print("Precomputing class embeddings...")
    wnids_in_seg = []
    for idx in range(num_images):
        target_ref = targets_refs[idx, 0]
        target_data = np.array(f[target_ref])
        wnid = ''.join([chr(c) for c in target_data.flatten()])
        wnids_in_seg.append(wnid)

    unique_wnids = sorted(set(wnids_in_seg))
    wnid_to_classname = {}
    wnid_to_prompt = {}
    for wnid in unique_wnids:
        cl = wnid_to_label.get(wnid)
        if cl is None:
            cl = get_synset_name(wnid)
        wnid_to_classname[wnid] = cl
        wnid_to_prompt[wnid] = f"a photo of a {cl}."

    all_prompts = [wnid_to_prompt[w] for w in unique_wnids]
    wnid_to_idx = {w: i for i, w in enumerate(unique_wnids)}
    idx_to_wnid = {i: w for w, i in wnid_to_idx.items()}

    all_text_embs = None
    cos_sim_matrix = None
    if clip_methods:
        tok_all = tokenizer(all_prompts).to(args.device)
        with torch.no_grad():
            all_text_embs = model.encode_text(tok_all, normalize=True)
        print(f"Precomputed {len(all_prompts)} class embeddings.")
        cos_sim_matrix = (all_text_embs @ all_text_embs.T).cpu().numpy()

    metric_label = 'Heatmap Mass (%)' if args.use_heatmap_mass else 'mIoU (%)'
    sampling_label = 'stratified' if args.stratified else 'random'
    print(f"\nMetric: {metric_label}, Sampling: {sampling_label}")

    # ---- Run each method ----
    all_results = {}  # method_name -> (cosine_sims, mious, pearson, spearman)
    
    print(f"\nMethods to run: {methods}")
    print(f"Images: {limit}, Negatives per image: {args.num_negatives}\n")

    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"  Running: {method_name}")
        print(f"{'='*60}")
        
        # Each method gets its own deterministic RNG so they all use the same negatives
        method_rng = random.Random(args.seed)
        
        if method_name == 'DAAM':
            cosine_sims, scores = run_daam_method(
                daam_segmenter=daam_segmenter,
                wnids_in_seg=wnids_in_seg,
                unique_wnids=unique_wnids,
                wnid_to_classname=wnid_to_classname,
                wnid_to_idx=wnid_to_idx,
                idx_to_wnid=idx_to_wnid,
                f=f,
                imgs_refs=imgs_refs,
                gts_refs=gts_refs,
                limit=limit,
                num_negatives=args.num_negatives,
                threshold=args.threshold,
                rng=method_rng,
                image_size=args.image_size,
                stratified=args.stratified,
                use_heatmap_mass=args.use_heatmap_mass,
            )
        else:
            cosine_sims, scores = run_single_method(
                method_name=method_name,
                model=model,
                preprocess=preprocess,
                all_text_embs=all_text_embs,
                cos_sim_matrix=cos_sim_matrix,
                wnids_in_seg=wnids_in_seg,
                unique_wnids=unique_wnids,
                wnid_to_classname=wnid_to_classname,
                wnid_to_idx=wnid_to_idx,
                idx_to_wnid=idx_to_wnid,
                f=f,
                imgs_refs=imgs_refs,
                gts_refs=gts_refs,
                limit=limit,
                num_negatives=args.num_negatives,
                threshold=args.threshold,
                rng=method_rng,
                device=args.device,
                image_size=args.image_size,
                gradcam_layer=args.gradcam_layer,
                stratified=args.stratified,
                use_heatmap_mass=args.use_heatmap_mass,
            )

        pearson_r, pearson_p = stats.pearsonr(cosine_sims, scores)
        spearman_r, spearman_p = stats.spearmanr(cosine_sims, scores)
        
        all_results[method_name] = {
            'cosine_sims': cosine_sims,
            'mious': scores,
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
        }

        print(f"\n  {method_name}: N={len(cosine_sims)}")
        print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
        print(f"  Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")


    # ---- Summary table ----
    print(f"\n\n{'='*70}")
    print(f"  CORRELATION SUMMARY ({model_type})")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print(f"  {'-'*61}")
    for method_name, res in all_results.items():
        print(f"  {method_name:<15} {res['pearson_r']:>10.4f} {res['pearson_p']:>12.2e} {res['spearman_r']:>12.4f} {res['spearman_p']:>12.2e}")
    print(f"{'='*70}")

    # ---- Paired within-image analysis ----
    if args.paired and cos_sim_matrix is not None:
        print(f"\n{'='*70}")
        print(f"  PAIRED ANALYSIS (closest vs farthest negative per image)")
        print(f"{'='*70}")
        # Re-run with explicit closest+farthest per image for each CLIP method
        for method_name in [m for m in methods if m != 'DAAM']:
            needs_renorm = (method_name != 'LeGrad')
            metric_fn = compute_heatmap_mass if args.use_heatmap_mass else compute_per_image_miou
            delta_cos = []
            delta_score = []
            paired_rng = random.Random(args.seed)
            for idx in tqdm(range(limit), desc=f"  Paired {method_name}"):
                try:
                    img_ref = imgs_refs[idx, 0]
                    img_obj = np.array(f[img_ref])
                    img_np = img_obj.transpose(2, 1, 0)
                    base_img = Image.fromarray(img_np)
                    img_t = preprocess(base_img).unsqueeze(0).to(args.device)

                    gt_ref = gts_refs[idx, 0]
                    gt_wrapper = f[gt_ref]
                    if gt_wrapper.dtype == 'object':
                        real_gt_ref = gt_wrapper[0, 0]
                        real_gt = np.array(f[real_gt_ref])
                        gt_mask = real_gt.transpose(1, 0)
                    else:
                        gt_mask = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                    gt_pil = Image.fromarray(gt_mask.astype(np.uint8))
                    gt_pil = transforms.Resize(
                        (args.image_size, args.image_size),
                        interpolation=InterpolationMode.NEAREST,
                    )(gt_pil)
                    gt_mask = np.array(gt_pil).astype(np.int32)

                    wnid = wnids_in_seg[idx]
                    cls_idx = wnid_to_idx[wnid]
                    sims = cos_sim_matrix[cls_idx].copy()
                    sims[cls_idx] = -999  # exclude self

                    closest_idx = int(np.argmax(sims))
                    sims_for_far = cos_sim_matrix[cls_idx].copy()
                    sims_for_far[cls_idx] = 999
                    farthest_idx = int(np.argmin(sims_for_far))

                    # Closest negative
                    close_emb = all_text_embs[closest_idx:closest_idx + 1]
                    hm_close = compute_heatmap(method_name, model, img_t, close_emb, args.gradcam_layer)
                    sc_close = metric_fn(hm_close, gt_mask, threshold=args.threshold, needs_renorm=needs_renorm) \
                        if not args.use_heatmap_mass else metric_fn(hm_close, gt_mask, needs_renorm=needs_renorm)

                    # Farthest negative
                    far_emb = all_text_embs[farthest_idx:farthest_idx + 1]
                    hm_far = compute_heatmap(method_name, model, img_t, far_emb, args.gradcam_layer)
                    sc_far = metric_fn(hm_far, gt_mask, threshold=args.threshold, needs_renorm=needs_renorm) \
                        if not args.use_heatmap_mass else metric_fn(hm_far, gt_mask, needs_renorm=needs_renorm)

                    delta_cos.append(cos_sim_matrix[closest_idx, cls_idx] - cos_sim_matrix[farthest_idx, cls_idx])
                    delta_score.append(sc_close - sc_far)
                except Exception as e:
                    continue

            delta_cos = np.array(delta_cos)
            delta_score = np.array(delta_score)
            if len(delta_cos) > 2:
                pr, pp = stats.pearsonr(delta_cos, delta_score)
                sr, sp = stats.spearmanr(delta_cos, delta_score)
                print(f"  {method_name:<15} Δ Pearson r={pr:.4f} (p={pp:.2e}), Δ Spearman ρ={sr:.4f} (p={sp:.2e}), N={len(delta_cos)}")

    f.close()

    # ---- Visualization ----
    os.makedirs(args.output_dir, exist_ok=True)

    n_methods = len(methods)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    markers = ['o', 's', '^', 'D', 'P']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    global_min = min(res['cosine_sims'].min() for res in all_results.values())
    global_max = max(res['cosine_sims'].max() for res in all_results.values())
    x_range = np.linspace(global_min, global_max, 200)

    # --- Left panel: scatter + regression lines ---
    ax = axes[0]
    for i, method_name in enumerate(methods):
        res = all_results[method_name]
        color = colors[i % len(colors)]

        # Scatter (low alpha so overlapping points are visible)
        ax.scatter(res['cosine_sims'], res['mious'], alpha=0.08, s=8, c=color,
                   edgecolors='none', label=None)

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(res['cosine_sims'], res['mious'])
        ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=2.5,
                label=f'{method_name} (r={res["pearson_r"]:.3f}, ρ={res["spearman_r"]:.3f})')

    ax.set_xlabel('Cosine Similarity (Negative Prompt ↔ Actual Class)', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{model_type} — Cosine Similarity vs Hallucination ({metric_label})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(global_min - 0.02, global_max + 0.02)

    # --- Right panel: binned mean lines with error bands ---
    ax2 = axes[1]
    n_bins = 15
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    for i, method_name in enumerate(methods):
        res = all_results[method_name]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        bin_centers = []
        bin_means = []
        bin_stds = []

        for b in range(n_bins):
            if b == n_bins - 1:
                mask = (res['cosine_sims'] >= bin_edges[b]) & (res['cosine_sims'] <= bin_edges[b + 1])
            else:
                mask = (res['cosine_sims'] >= bin_edges[b]) & (res['cosine_sims'] < bin_edges[b + 1])
            if mask.sum() >= 3:  # require at least 3 samples per bin
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_means.append(res['mious'][mask].mean())
                bin_stds.append(res['mious'][mask].std() / np.sqrt(mask.sum()))

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        # Line + markers
        ax2.plot(bin_centers, bin_means, color=color, linewidth=2, marker=marker,
                 markersize=5, label=f'{method_name} (r={res["pearson_r"]:.3f})')
        # Shaded error band (±1 SE)
        ax2.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                         color=color, alpha=0.15)

    ax2.set_xlabel('Cosine Similarity (Negative Prompt ↔ Actual Class)', fontsize=12)
    ax2.set_ylabel(f'Mean {metric_label}', fontsize=12)
    ax2.set_title(f'Binned Mean {metric_label} (±1 SE) by Cosine Similarity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(global_min - 0.02, global_max + 0.02)

    fig.suptitle(
        f'{model_type} — {limit} images, {args.num_negatives} negatives each',
        fontsize=10, color='gray', y=0.98
    )

    plt.tight_layout()
    suffix = '_all_methods' if n_methods > 1 else ''
    output_path = os.path.join(args.output_dir, f'{args.output_name}{suffix}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # Save raw data
    save_dict = {'model_type': model_type}
    for method_name, res in all_results.items():
        prefix = method_name.lower().replace(' ', '_')
        save_dict[f'{prefix}_cosine_sims'] = res['cosine_sims']
        save_dict[f'{prefix}_mious'] = res['mious']
        save_dict[f'{prefix}_pearson_r'] = res['pearson_r']
        save_dict[f'{prefix}_pearson_p'] = res['pearson_p']
        save_dict[f'{prefix}_spearman_r'] = res['spearman_r']
        save_dict[f'{prefix}_spearman_p'] = res['spearman_p']

    data_path = os.path.join(args.output_dir, f'{args.output_name}{suffix}_data.npz')
    np.savez(data_path, **save_dict)
    print(f"Data saved to: {data_path}")


if __name__ == '__main__':
    main()
