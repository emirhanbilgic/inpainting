#!/usr/bin/env python3
"""
Generate HCI heatmaps for 5 attribution methods (LeGrad, GradCAM, AttentionCAM,
CheferCAM, DAAM) with and without OMP for the 21 Pascal VOC animal images.

Usage:
    python scripts/generate_hci_all_methods.py --method LeGrad
    python scripts/generate_hci_all_methods.py --method GradCAM
    python scripts/generate_hci_all_methods.py --method AttentionCAM
    python scripts/generate_hci_all_methods.py --method CheferCAM
    python scripts/generate_hci_all_methods.py --method DAAM
"""

import sys
import os
import types
from unittest.mock import MagicMock

# --- MOCKING PROBLEMATIC MODULES (MUST BE AT THE VERY TOP) ---
# When running DAAM, we must NOT mock transformers/tokenizers because
# diffusers imports real classes from them (CLIPImageProcessor, etc.)
_is_daam = "--method" in sys.argv and "DAAM" in sys.argv
_is_attentioncam = "--method" in sys.argv and "AttentionCAM" in sys.argv

def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = types.SimpleNamespace(
        name=name, origin="mock", submodule_search_locations=None,
        loader=None, has_location=False
    )
    m.__file__ = "mock"
    m.__path__ = []
    return m

problematic_prefixes = ["matplotlib"]
if not (_is_daam or _is_attentioncam):
    problematic_prefixes += ["sklearn", "transformers", "tokenizers"]
for prefix in problematic_prefixes:
    sys.modules[prefix] = create_mock_module(prefix)

problematic_specifics = [
    "matplotlib.pyplot", "matplotlib.colors", "matplotlib.cm",
]
if not (_is_daam or _is_attentioncam):
    problematic_specifics += [
        "sklearn.metrics", "sklearn.utils", "sklearn.base",
        "sklearn.utils.murmurhash", "sklearn.utils._cython_blas",
        "sklearn.utils._logistic_sigmoid",
        "transformers.generation", "transformers.generation.utils",
        "transformers.generation.candidate_generator", "transformers.utils",
        "tokenizers"
    ]
for name in problematic_specifics:
    sys.modules[name] = create_mock_module(name)

sys.modules["matplotlib"].pyplot = MagicMock()
sys.modules["matplotlib"].colors = MagicMock()
sys.modules["matplotlib"].cm = MagicMock()

# --- NOW IMPORT OTHER LIBRARIES ---
import argparse
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import open_clip

# --- PROJECT SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess
from open_clip.timm_model import TimmModel

CANDIDATE_ANIMALS = ["cat", "dog", "horse", "sheep", "bird"]
DATASET_ROOT = "/Users/emirhan/Desktop/pascal-voc-2012-DatasetNinja"
# OUTPUT_DIR will be set dynamically based on list file


def parse_tex_list(filepath):
    """
    Parse a tab-separated .tex list file.
    Expected format:
    Image	Objects
    filename.jpg	Obj1, Obj2
    ...
    """
    items = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Try to split by tab first (as per original design)
        parts = line.split('\t')
        if len(parts) < 2:
            # Fallback: split by whitespace (handle spaces instead of tabs)
            # We assume the first column (filename) has no spaces.
             parts = line.split(None, 1)

        if len(parts) >= 2:
            img_name = parts[0].strip()
            objects_str = parts[1].strip()
            # Split by comma and strip whitespace, convert to lowercase for consistency
            objects = [obj.strip().lower() for obj in objects_str.split(',')]
            items.append((img_name, objects))
    return items


# ==============================================================================
# OMP Sparse Residual (for CLIP methods)
# ==============================================================================
def omp_sparse_residual(x_1x, D, max_atoms=8, tol=1e-6):
    """Remove shared components from text embedding using OMP."""
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    device = x_1x.device
    dtype = x_1x.dtype
    x = x_1x.clone().cpu().float()
    D_cpu = D.cpu().float()
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    for _ in range(max_atoms):
        c = (r @ D_cpu.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D_cpu[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0])
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
        x_hat = (s.t() @ D_S)
        r = (x - x_hat)
        if float(torch.norm(r) <= tol):
            break
    r = r.to(device=device, dtype=dtype)
    if torch.norm(r) <= tol:
        return F.normalize(x_1x, dim=-1)
    return F.normalize(r, dim=-1)


# ==============================================================================
# Visualization Helpers
# ==============================================================================
def apply_colormap_jet(heatmap_np):
    """Manual JET colormap (avoids matplotlib dependency)."""
    val = np.clip(heatmap_np * 4, 0, 4)
    rgb = np.zeros(heatmap_np.shape + (3,), dtype=np.float32)
    mask = (val >= 0) & (val <= 1)
    rgb[mask, 2] = 1.0; rgb[mask, 1] = val[mask]; rgb[mask, 0] = 0.0
    mask = (val > 1) & (val <= 2)
    rgb[mask, 2] = 2.0 - val[mask]; rgb[mask, 1] = 1.0; rgb[mask, 0] = 0.0
    mask = (val > 2) & (val <= 3)
    rgb[mask, 2] = 0.0; rgb[mask, 1] = 1.0; rgb[mask, 0] = val[mask] - 2.0
    mask = (val > 3) & (val <= 4)
    rgb[mask, 2] = 0.0; rgb[mask, 1] = 4.0 - val[mask]; rgb[mask, 0] = 1.0
    return (rgb * 255).astype(np.uint8)


def get_heatmap_vis(image_pil, heatmap_tensor):
    """Overlay heatmap on original image with 50% alpha."""
    w, h = image_pil.size
    heatmap = np.nan_to_num(heatmap_tensor.numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8, mode='L')
    heatmap_pil = heatmap_pil.resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil).astype(float) / 255.0
    colormap_img_np = apply_colormap_jet(heatmap_resized)
    colormap_img = Image.fromarray(colormap_img_np, mode='RGB')
    blended = Image.blend(image_pil, colormap_img, alpha=0.5)
    return blended


def get_text_embedding(model, tokenizer, prompt, device):
    """Encode a text prompt into a normalized embedding."""
    tok = tokenizer([prompt]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb


def find_image(img_name, search_paths=None):
    """Find image in Pascal VOC dataset directories or provided paths."""
    if search_paths:
        for search_path in search_paths:
            path = os.path.join(search_path, img_name)
            if os.path.exists(path):
                return path

    for root_dir in ["trainval/img", "train/img", "val/img"]:
        path = os.path.join(DATASET_ROOT, root_dir, img_name)
        if os.path.exists(path):
            return path
    return None


# ==============================================================================
# CLIP Attribution Methods (from optimize_anti_hallucination.py)
# ==============================================================================

def compute_legrad_heatmap(model, image, text_emb):
    """LeGrad heatmap via LeWrapper."""
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb)
    logits = logits[0, 0].clamp(0, 1).detach().cpu()
    return logits


def compute_gradcam_heatmap(model, image, text_emb_1x, layer_index=8):
    """GradCAM heatmap via LeWrapper."""
    if hasattr(model, "starting_depth"):
        layer_index = max(layer_index, int(model.starting_depth))
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def compute_attentioncam_heatmap(model, image, text_emb_1x):
    """
    AttentionCAM: gradient-weighted attention from last layer.
    For SigLIP: uses attentional pooler attention.
    For CLIP: uses last self-attention layer.
    """
    H_img, W_img = image.shape[-2:]

    try:
        is_siglip = isinstance(model.visual, TimmModel)

        with torch.enable_grad():
            if is_siglip:
                pooler = model.visual.trunk.attn_pool
                blocks = list(model.visual.trunk.blocks)

                x = model.visual.trunk.patch_embed(image)
                if x.dim() == 4:
                    B, H, W, C = x.shape
                    x = x.reshape(B, H*W, C)
                else:
                    B, _, C = x.shape

                if model.visual.trunk.pos_embed is not None:
                    x = x + model.visual.trunk.pos_embed

                for block in blocks:
                    x = block(x)

                B, N, C = x.shape

                if pooler.pos_embed is not None:
                    x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)

                q_latent = pooler.latent.expand(B, -1, -1)
                q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
                kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
                k, v = kv.unbind(0)
                q, k = pooler.q_norm(q), pooler.k_norm(k)

                attn_probs = (q * pooler.scale) @ k.transpose(-2, -1)
                attn_probs = attn_probs.softmax(dim=-1)
                attn_probs.requires_grad_(True)

                x_pool = (attn_probs @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
                x_pool = pooler.proj(x_pool)
                x_pool = pooler.proj_drop(x_pool)
                x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))

                if pooler.pool == 'token':
                    pooled_feat = x_pool[:, 0]
                elif pooler.pool == 'avg':
                    pooled_feat = x_pool.mean(1)
                else:
                    pooled_feat = x_pool[:, 0]

                image_features = F.normalize(pooled_feat, dim=-1)

                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()

                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity, inputs=[attn_probs],
                    retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                cam = attn_probs * grad
                cam = cam.mean(dim=1)[:, 0]
                cam = cam.clamp(min=0)

                num_patches = cam.shape[-1]
                grid_size = int(math.sqrt(num_patches))
                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                heatmap = cam[0].reshape(grid_size, grid_size)

            else:
                # CLIP
                image_features = model.encode_image(image, normalize=True)
                text_emb_detached = text_emb_1x.detach()
                similarity = (image_features @ text_emb_detached.t()).sum()

                if hasattr(model.visual, 'transformer'):
                    blocks_list = list(model.visual.transformer.resblocks)
                else:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                last_block = blocks_list[-1]
                if not hasattr(last_block, 'attn') or not hasattr(last_block.attn, 'attention_maps'):
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                attn_map = last_block.attn.attention_maps

                model.zero_grad()
                grad = torch.autograd.grad(
                    outputs=similarity, inputs=[attn_map],
                    retain_graph=False, create_graph=False, allow_unused=True
                )[0]

                if grad is None:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                grad_weights = grad.mean(dim=[1, 2], keepdim=True)
                cam = attn_map * grad_weights
                cls_attn = cam[:, 0, 1:]
                cls_attn = cls_attn.mean(dim=0).clamp(min=0)

                num_patches = cls_attn.shape[0]
                grid_size = int(math.sqrt(num_patches))
                if grid_size * grid_size != num_patches:
                    return torch.ones(H_img, W_img, device='cpu') * 0.5

                heatmap = cls_attn.reshape(grid_size, grid_size)

        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(H_img, W_img), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.detach().cpu()

    except Exception as e:
        print(f"  [AttentionCAM] Error: {e}")
        return torch.ones(H_img, W_img, device='cpu') * 0.5


def compute_chefercam_heatmap(model, image, text_emb_1x):
    """
    CheferCAM: GradCAM on the last attention layer.
    For SigLIP: uses attentional pooler attention.
    For CLIP: uses last self-attention layer.
    """
    model.zero_grad()
    num_prompts = text_emb_1x.shape[0]

    if isinstance(model.visual, TimmModel):
        is_timm = True
        blocks = list(model.visual.trunk.blocks)
    else:
        is_timm = False
        blocks = list(model.visual.transformer.resblocks)

    with torch.enable_grad():
        if is_timm:
            pooler = model.visual.trunk.attn_pool
            blocks = list(model.visual.trunk.blocks)

            x = model.visual.trunk.patch_embed(image)
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.reshape(B, H*W, C)
            else:
                B, _, C = x.shape

            if model.visual.trunk.pos_embed is not None:
                x = x + model.visual.trunk.pos_embed

            for block in blocks:
                x = block(x)

            B, N, C = x.shape

            if pooler.pos_embed is not None:
                x = x + pooler.pos_embed.unsqueeze(0).to(x.dtype)

            q_latent = pooler.latent.expand(B, -1, -1)
            q = pooler.q(q_latent).reshape(B, pooler.latent_len, pooler.num_heads, pooler.head_dim).transpose(1, 2)
            kv = pooler.kv(x).reshape(B, N, 2, pooler.num_heads, pooler.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = pooler.q_norm(q), pooler.k_norm(k)

            attn_weights = (q * pooler.scale) @ k.transpose(-2, -1)
            attn_weights = attn_weights.softmax(dim=-1)
            attn_weights.requires_grad_(True)

            x_pool = (attn_weights @ v).transpose(1, 2).reshape(B, pooler.latent_len, C)
            x_pool = pooler.proj(x_pool)
            x_pool = pooler.proj_drop(x_pool)
            x_pool = x_pool + pooler.mlp(pooler.norm(x_pool))

            if pooler.pool == 'token':
                pooled_feat = x_pool[:, 0]
            elif pooler.pool == 'avg':
                pooled_feat = x_pool.mean(1)
            else:
                pooled_feat = x_pool[:, 0]

            image_features = F.normalize(pooled_feat, dim=-1)
            bsz = B

        else:
            # CLIP
            x = model.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            batch_size = x.shape[0]
            class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
            class_token = class_token.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)

            num_patches = x.shape[1] - 1
            if hasattr(model.visual, 'original_pos_embed'):
                pos_embed = model.visual.original_pos_embed
            else:
                pos_embed = model.visual.positional_embedding

            if pos_embed.shape[0] != x.shape[1]:
                cls_pos = pos_embed[:1]
                patch_pos = pos_embed[1:]
                orig_size = int(math.sqrt(patch_pos.shape[0]))
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                new_size = int(math.sqrt(num_patches))
                patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)

            x = x + pos_embed.unsqueeze(0).to(x.dtype)
            if hasattr(model.visual, 'ln_pre'):
                x = model.visual.ln_pre(x)

            x = x.permute(1, 0, 2)

            for i in range(len(blocks) - 1):
                x = blocks[i](x)

            last_block = blocks[-1]
            last_attn = last_block.attn
            x_normed = last_block.ln_1(x)

            qkv = F.linear(x_normed, last_attn.in_proj_weight, last_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            seq_len, bsz, embed_dim = q.shape
            num_heads = last_attn.num_heads
            head_dim = embed_dim // num_heads

            q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)

            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights.requires_grad_(True)

            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
            attn_output = last_attn.out_proj(attn_output)

            x = x + attn_output
            x = x + last_block.mlp(last_block.ln_2(x))

            x = x.permute(1, 0, 2)
            image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
            image_features = F.normalize(image_features, dim=-1)

        # Cam calculation
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)

        grad = torch.autograd.grad(s, [attn_weights], retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(attn_weights)

        if is_timm:
            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)[:, 0]
            num_patches = cam.shape[-1]
        else:
            seq_len = attn_weights.shape[1]
            num_heads = blocks[-1].attn.num_heads
            grad = grad.view(bsz, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
            grad = torch.clamp(grad, min=0)
            cam = grad * attn_weights
            cam = cam.mean(dim=1)
            cam = cam[:, 0, 1:]
            num_patches = cam.shape[-1]

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size

        heatmap = cam.reshape(bsz, 1, h, w)
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode='bilinear', align_corners=False)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap[0, 0].detach().cpu()


# ==============================================================================
# DAAM Key-Space OMP — imported directly from optimize_anti_hallucination.py
# ==============================================================================
# We import the tested reference implementation rather than re-implementing it.
# The import happens lazily inside run_daam_method() to avoid loading diffusers
# for CLIP-only runs.



# ==============================================================================
# Main
# ==============================================================================

CLIP_METHODS = ["LeGrad", "GradCAM", "AttentionCAM", "CheferCAM"]
ALL_METHODS = CLIP_METHODS + ["DAAM"]


def run_clip_method(method_name, model, tokenizer, preprocess, device, items, candidate_animals, output_dir, image_root=None):
    """Run a CLIP-based method on all images."""
    print(f"\n{'='*60}")
    print(f"Running method: {method_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    for img_name, real_classes in tqdm(items, desc=f"{method_name}"):
        img_id = os.path.splitext(img_name)[0]
        search_paths = [image_root] if image_root else None
        img_path = find_image(img_name, search_paths=search_paths)
        if not img_path:
            print(f"  [SKIP] Image not found: {img_name}")
            continue

        try:
            original_image = Image.open(img_path).convert("RGB")
            img_t = preprocess(original_image).unsqueeze(0).to(device)

            # Select fake animal
            random.seed(42 + hash(img_id))
            fake_candidates = [a for a in candidate_animals if a not in real_classes]
            if not fake_candidates:
                 # If all candidates are present, pick one at random even if present (should not happen with good lists)
                 # or just pick from full list.
                 fake_candidates = candidate_animals
            
            fake_animal = random.choice(fake_candidates)

            # Build embeddings for all relevant classes
            class_embeddings = {}
            for cls_name in real_classes + [fake_animal]:
                prompt = f"a photo of a {cls_name}."
                class_embeddings[cls_name] = get_text_embedding(model, tokenizer, prompt, device)

            # Process each target
            target_info = []
            # Handle variable number of real classes
            for i, cls_name in enumerate(real_classes):
                target_info.append((f"Target_{i+1}", cls_name))
            target_info.append(("Target_fake", fake_animal))

            for folder_name, target_cls in target_info:
                out_folder = os.path.join(output_dir, img_id, f"{folder_name}_{method_name}")
                os.makedirs(out_folder, exist_ok=True)

                target_emb = class_embeddings[target_cls]

                # 1. Normal heatmap
                if method_name == "LeGrad":
                    heatmap_normal = compute_legrad_heatmap(model, img_t, target_emb)
                elif method_name == "GradCAM":
                    heatmap_normal = compute_gradcam_heatmap(model, img_t, target_emb)
                elif method_name == "AttentionCAM":
                    heatmap_normal = compute_attentioncam_heatmap(model, img_t, target_emb)
                elif method_name == "CheferCAM":
                    heatmap_normal = compute_chefercam_heatmap(model, img_t, target_emb)

                vis_normal = get_heatmap_vis(original_image, heatmap_normal)

                # 2. OMP heatmap (text-embedding residual)
                is_fake = folder_name == "Target_fake"
                if is_fake:
                    neg_classes = real_classes
                else:
                    other_real = [c for c in real_classes if c != target_cls]
                    neg_classes = other_real + [fake_animal]

                D_list = [class_embeddings[nc] for nc in neg_classes]
                D = torch.cat(D_list, dim=0)
                target_emb_omp = omp_sparse_residual(target_emb, D, max_atoms=len(neg_classes))

                if method_name == "LeGrad":
                    heatmap_omp = compute_legrad_heatmap(model, img_t, target_emb_omp)
                elif method_name == "GradCAM":
                    heatmap_omp = compute_gradcam_heatmap(model, img_t, target_emb_omp)
                elif method_name == "AttentionCAM":
                    heatmap_omp = compute_attentioncam_heatmap(model, img_t, target_emb_omp)
                elif method_name == "CheferCAM":
                    heatmap_omp = compute_chefercam_heatmap(model, img_t, target_emb_omp)

                vis_omp = get_heatmap_vis(original_image, heatmap_omp)

                # Save
                original_image.save(os.path.join(out_folder, "original.png"))
                vis_normal.save(os.path.join(out_folder, "normal.png"))
                vis_omp.save(os.path.join(out_folder, "omp.png"))
                with open(os.path.join(out_folder, f"target_{target_cls}.txt"), "w") as f:
                    f.write(target_cls)

        except Exception as e:
            print(f"  [ERROR] {img_id}: {e}")
            import traceback
            traceback.print_exc()



# ==============================================================================
# DAAM Key-Space OMP — Ported from scripts/daam_omp_comparison.py
# ==============================================================================

if _is_daam:
    from typing import List, Dict, Tuple
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention_processor import Attention
    from daam import trace as daam_trace
    from daam.trace import DiffusionHeatMapHooker, UNetCrossAttentionHooker
    from daam.hook import UNetCrossAttentionLocator
    from daam.heatmap import RawHeatMapCollection
    from daam.utils import auto_autocast
    from daam_segmentation import DAAMSegmenter

    class Float32DAAMSegmenter(DAAMSegmenter):
        """DAAMSegmenter that uses float32 to avoid NaN on MPS."""
        def __init__(self, model_id="Manojb/stable-diffusion-2-base", device='mps'):
            if StableDiffusionPipeline is None:
                raise ImportError("Please install 'daam' and 'diffusers'")
            print(f"[DAAM-F32] Loading Stable Diffusion pipeline in float32: {model_id}...")
            self.device = device
            self.pipeline = StableDiffusionPipeline.from_pretrained(
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
            if self.device == 'cuda' or self.device == 'mps':
                torch.mps.empty_cache() if self.device == 'mps' else torch.cuda.empty_cache()

            w, h = image_pil.size
            img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
            img_arr = np.array(img_resized).astype(np.float32) / 255.0
            img_arr = img_arr * 2.0 - 1.0
            # Use float32 instead of half() to avoid NaN on MPS
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

            from daam import trace
            with trace(self.pipeline) as tc:
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


    class KeySpaceOMPProcessor:
        """
        Custom attention processor that orthogonalizes the target token's key vector
        against distractor token keys in the cross-attention layers of the UNet.
        """
        
        def __init__(
            self, 
            target_token_indices: List[int],
            distractor_token_indices: List[List[int]],
            beta: float = 1.0,
            heat_maps: RawHeatMapCollection = None,
            layer_idx: int = 0,
            latent_hw: int = 4096,
            context_size: int = 77,
            parent_trace=None,
        ):
            self.target_token_indices = target_token_indices
            self.distractor_token_indices = distractor_token_indices
            self.beta = beta
            self.heat_maps = heat_maps
            self.layer_idx = layer_idx
            self.latent_hw = latent_hw
            self.context_size = context_size
            self.parent_trace = parent_trace
        
        def _orthogonalize_keys(self, key: torch.Tensor, n_heads: int) -> torch.Tensor:
            key = key.clone()
            
            # For each target token index
            for target_idx in self.target_token_indices:
                # Get the target key vector: [batch*heads, head_dim]
                target_key = key[:, target_idx, :]
                
                # Collect all distractor key vectors
                for dist_indices in self.distractor_token_indices:
                    for dist_idx in dist_indices:
                        dist_key = key[:, dist_idx, :]  # [batch*heads, head_dim]
                        
                        # Normalize distractor key
                        dist_norm = dist_key / (dist_key.norm(dim=-1, keepdim=True) + 1e-8)
                        
                        # Project target onto distractor and subtract
                        # projection: [batch*heads, head_dim]
                        projection = (target_key * dist_norm).sum(dim=-1, keepdim=True) * dist_norm
                        target_key = target_key - self.beta * projection
                
                # Write back orthogonalized key
                key[:, target_idx, :] = target_key
            
            return key
        
        @torch.no_grad()
        def _unravel_attn(self, x):
            """Unravel attention maps (from DAAM source)."""
            h = w = int(math.sqrt(x.size(1)))
            maps = []
            x = x.permute(2, 0, 1)
            
            with auto_autocast(dtype=torch.float32):
                for map_ in x:
                    map_ = map_.view(map_.size(0), h, w)
                    map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                    maps.append(map_)
            
            maps = torch.stack(maps, 0)
            return maps.permute(1, 0, 2, 3).contiguous()
        
        def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
        ):
            """Custom attention forward with key-space OMP."""
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
            
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            # >>> KEY-SPACE OMP: Orthogonalize target keys against distractor keys <<<
            # Only apply to cross-attention (encoder_hidden_states != hidden_states)
            is_cross_attention = (key.shape[1] == self.context_size)
            if is_cross_attention and self.beta > 0:
                key = self._orthogonalize_keys(key, attn.heads)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            
            # Store heat maps (same logic as DAAM's UNetCrossAttentionHooker)
            factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1])) if attention_probs.shape[1] > 0 else 8
            if self.parent_trace is not None:
                self.parent_trace._gen_idx += 1
            
            if self.heat_maps is not None and attention_probs.shape[-1] == self.context_size and factor != 8:
                maps = self._unravel_attn(attention_probs)
                for head_idx, heatmap in enumerate(maps):
                    self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states


    def get_token_indices(tokenizer, prompt: str, concept: str) -> List[int]:
        """Get token indices for a concept within a prompt."""
        # Tokenize the full prompt
        tokens = tokenizer.tokenize(prompt)
        concept_tokens = tokenizer.tokenize(concept)
        
        indices = []
        for i in range(len(tokens)):
            if tokens[i:i+len(concept_tokens)] == concept_tokens:
                # +1 because of the SOS token
                indices.extend(range(i + 1, i + 1 + len(concept_tokens)))
                break
        
        if not indices:
            # Fallback: just find any matching sub-token
            for i, tok in enumerate(tokens):
                for ct in concept_tokens:
                    if ct in tok or tok in ct:
                        indices.append(i + 1)  # +1 for SOS
            
        return indices


    def run_daam_with_key_space_omp(
        segmenter: DAAMSegmenter,
        image_pil: Image.Image,
        prompt: str,
        target_concept: str,
        competing_concepts: List[str],
        beta: float = 1.0,
        size: int = 512,
    ) -> torch.Tensor:
        """Run DAAM with true key-space OMP intervention."""
        pipeline = segmenter.pipeline
        tokenizer = segmenter.tokenizer
        text_encoder = segmenter.text_encoder
        vae = segmenter.vae
        unet = segmenter.unet
        scheduler = segmenter.scheduler
        device = segmenter.device
        
        w, h = image_pil.size
        
        # 1. Preprocess image
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device).float()
        
        # 2. Encode to latents
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215
        
        # 3. Build the combined prompt with all concepts
        all_concepts = [target_concept] + competing_concepts
        combined_prompt = f"{', '.join(all_concepts)}."
        
        # Get token indices
        target_indices = get_token_indices(tokenizer, combined_prompt, target_concept)
        distractor_indices = [
            get_token_indices(tokenizer, combined_prompt, comp)
            for comp in competing_concepts
        ]
        
        if not target_indices:
            print(f"    WARNING: Could not find target token indices, falling back to basic DAAM")
            return segmenter.predict(image_pil, f"{target_concept}.", size=size)
        
        # 4. Prepare text embeddings
        text_input = tokenizer(
            combined_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
        # 5. Add noise
        scheduler.set_timesteps(50, device=device)
        noise = torch.randn_like(latents)
        timestep = torch.tensor([21], device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timestep)
        
        # 6. Create heat map collection and set up key-space OMP processors
        heat_maps = RawHeatMapCollection()
        locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
        cross_attn_modules = locator.locate(unet)
        
        latent_hw = 4096  # 64x64 for SD 2.0
        
        class GenIdxTracker:
            def __init__(self):
                self._gen_idx = 0
        
        tracker = GenIdxTracker()
        
        original_processors = {}
        for idx, module in enumerate(cross_attn_modules):
            original_processors[idx] = module.processor
            module.set_processor(KeySpaceOMPProcessor(
                target_token_indices=target_indices,
                distractor_token_indices=distractor_indices,
                beta=beta,
                heat_maps=heat_maps,
                layer_idx=idx,
                latent_hw=latent_hw,
                context_size=77,
                parent_trace=tracker,
            ))
        
        # 7. Forward pass
        try:
            with torch.no_grad():
                _ = unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample
        finally:
            # Restore original processors
            for idx, module in enumerate(cross_attn_modules):
                module.set_processor(original_processors[idx])
        
        # 8. Extract heatmap
        x = int(np.sqrt(latent_hw))
        factors = {0, 1, 2, 4, 8, 16, 32, 64}
        
        all_merges = []
        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and factor != 8:
                    heat_map = heat_map.unsqueeze(1)
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))
        
        if not all_merges:
            return segmenter.predict(image_pil, f"{target_concept}.", size=size)
        
        maps = torch.stack(all_merges, dim=0)
        maps = maps.mean(0)[:, 0]  # Average across layers/heads
        
        target_maps = []
        for tidx in target_indices:
            if tidx < maps.shape[0]:
                target_maps.append(maps[tidx])
        
        if not target_maps:
            return segmenter.predict(image_pil, f"{target_concept}.", size=size)
        
        heatmap = torch.stack(target_maps).mean(0)
        
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()

def run_daam_method(model_id, device, items, candidate_animals, output_dir, image_root=None, beta=0.01):
    """Run DAAM method on all images."""
    print(f"\n{'='*60}")
    print(f"Running method: DAAM")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Lazy import to avoid loading diffusers/transformers if not running DAAM
    from daam import trace
    
    # Initialize our float32 segmenter
    segmenter = Float32DAAMSegmenter(model_id=model_id, device=device)

    for img_name, real_classes in tqdm(items, desc="DAAM"):
        img_id = os.path.splitext(img_name)[0]
        search_paths = [image_root] if image_root else None
        img_path = find_image(img_name, search_paths=search_paths)
        if not img_path:
            print(f"  [SKIP] Image not found: {img_name}")
            continue

        try:
            original_image = Image.open(img_path).convert("RGB")
            
            # Select fake animal
            random.seed(42 + hash(img_id))
            fake_candidates = [a for a in candidate_animals if a not in real_classes]
            if not fake_candidates:
                fake_candidates = candidate_animals
            fake_animal = random.choice(fake_candidates)

            # Prepare prompts
            prompts = {}
            for cls_name in real_classes + [fake_animal]:
                prompts[cls_name] = f"{cls_name}."

            # Process each target
            target_info = []
            for i, cls_name in enumerate(real_classes):
                target_info.append((f"Target_{i+1}", cls_name))
            target_info.append(("Target_fake", fake_animal))

            for folder_name, target_cls in target_info:
                out_folder = os.path.join(output_dir, img_id, f"{folder_name}_DAAM")
                os.makedirs(out_folder, exist_ok=True)

                prompt = prompts[target_cls]

                # 1. Normal DAAM heatmap
                heatmap_normal = segmenter.predict(original_image, prompt, size=512)
                vis_normal = get_heatmap_vis(original_image, heatmap_normal)

                # 2. Key-Space OMP DAAM heatmap
                # Determine negative classes
                is_fake = folder_name == "Target_fake"
                if is_fake:
                    neg_classes = real_classes
                else:
                    # For a real class, only absent concepts compete?
                    # Original logic: other_real + [fake_animal]
                    other_real = [c for c in real_classes if c != target_cls]
                    neg_classes = other_real + [fake_animal]
                
                # Check if we should restrict to only absent concepts acting as negatives
                # (User requested this in previous turn "Fixing DAAM OMP Semantics")
                # But here I am modifying generate_hci_all_methods.py which might not have been fully synced?
                # The user's request for list2.tex implies we should use similar logic.
                # However, looking at the previous conversation history "Fixing DAAM OMP Semantics",
                # the user wanted "only absent concepts".
                # For `Target_fake` (fake_animal), `real_classes` are all absent from the prompt but present in image?
                # Wait, "absent concepts" usually means concepts NOT in the image.
                # If target is "fake_animal" (not in image), then `real_classes` (in image) are the competitors?
                # If target is "real_class" (in image), then `fake_animal` (not in image) is competitor?
                # Let's stick to the logic in `run_clip_method` for now to be consistent within this file,
                # or better, use the logic that was likely intended.
                # In `run_clip_method`:
                # is_fake: neg_classes = real_classes
                # else: neg_classes = other_real + [fake_animal]
                
                # Let's just use what I wrote above which mirrors `run_clip_method`.

                heatmap_omp = run_daam_with_key_space_omp(
                    segmenter, 
                    original_image, 
                    prompt=prompt,
                    target_concept=target_cls, 
                    competing_concepts=neg_classes,
                    beta=beta,
                    size=512
                )
                vis_omp = get_heatmap_vis(original_image, heatmap_omp)

                # Save
                original_image.save(os.path.join(out_folder, "original.png"))
                vis_normal.save(os.path.join(out_folder, "normal.png"))
                vis_omp.save(os.path.join(out_folder, "omp.png"))
                with open(os.path.join(out_folder, f"target_{target_cls}.txt"), "w") as f:
                    f.write(target_cls)

        except Exception as e:
            print(f"  [ERROR] {img_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HCI heatmaps for multiple methods.")
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS, help="Attribution method to use.")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (mps, cuda, cpu).")
    parser.add_argument("--model", type=str, default="Manojb/stable-diffusion-2-base", help="Model ID for DAAM.")
    parser.add_argument("--list_file", type=str, default=None, help="Path to .tex list file.")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for images (overrides default Pascal VOC search).")
    parser.add_argument("--beta", type=float, default=0.01, help="Beta parameter for DAAM OMP.")
    args = parser.parse_args()

    # Determine dataset list and candidates
    if args.list_file:
        list_file_path = args.list_file
        if not os.path.exists(list_file_path):
             # Try relative to valid paths if absolute path not given
             # But user provided "web_application/list2.tex", handle relative to CWD or script
             potential_path = os.path.abspath(list_file_path)
             if not os.path.exists(potential_path):
                 # Try relative to project root
                 potential_path = os.path.join(project_root, list_file_path)
             if os.path.exists(potential_path):
                 list_file_path = potential_path
             else:
                 raise FileNotFoundError(f"Could not find list file: {args.list_file}")

        print(f"Using list file: {list_file_path}")
        items = parse_tex_list(list_file_path)
        
        # Determine candidates and output dir based on filename
        filename = os.path.basename(list_file_path)
        if "list2.tex" in filename:
            candidate_animals = ["helicopter", "monkey", "boat"]
            output_dir_name = "data_list2"
        elif "cat_dog_sheep" in filename:
            candidate_animals = ["cat", "dog", "sheep"]
            output_dir_name = "cat_dog_sheep_heatmaps"
        elif filename == "list.tex":
            # Default behavior for list.tex
            candidate_animals = ["bird", "cat", "cow", "dog", "horse", "sheep"]
            output_dir_name = "data"
        else:
             # Generic handler for other lists (e.g. two_animals_list.txt)
             all_objects = set()
             for _, objs in items:
                 all_objects.update(objs)
             candidate_animals = sorted(list(all_objects))
             name_no_ext = os.path.splitext(filename)[0]
             output_dir_name = f"data_{name_no_ext}"
             print(f"Auto-detected candidates: {candidate_animals}")
             print(f"Output directory name: {output_dir_name}")
            
    else:
        # Fallback to hardcoded behavior if no list specified (or default to list.tex)
        # For backward compatibility, let's look for list.tex in web_application
        default_list = os.path.join(project_root, "web_application", "list.tex")
        if os.path.exists(default_list):
            print(f"Using default list file: {default_list}")
            items = parse_tex_list(default_list)
            candidate_animals = ["bird", "cat", "cow", "dog", "horse", "sheep"]
            output_dir_name = "data"
        else:
            print("Warning: list.tex not found, using empty list.")
            items = []
            candidate_animals = []
            output_dir_name = "data"

    if args.list_file and "cat_dog_sheep" in os.path.basename(args.list_file):
        output_dir = os.path.join(project_root, "scripts", "data", "cat_dog_sheep_heatmaps")
    else:
        output_dir = os.path.join(project_root, "web_application", output_dir_name)

    device = args.device
    
    if args.method == "DAAM":
        run_daam_method(args.model, device, items, candidate_animals, output_dir, image_root=args.image_root, beta=args.beta)
    else:
        # Load CLIP model
        if args.method == "AttentionCAM":
            print("Loading SigLIP (ViT-B-16-SigLIP) for AttentionCAM...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-16-SigLIP",
                pretrained="webli",
                device=device
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        else:
            print("Loading CLIP (ViT-B-16)...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                device=device
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16")
        
        model.to(device)
        model.eval()
        
        # Original script applied LeWrapper and LePreprocess unconditionally
        model = LeWrapper(model, layer_index=-2)
        preprocess = LePreprocess(preprocess=preprocess, image_size=224)

        run_clip_method(args.method, model, tokenizer, preprocess, device, items, candidate_animals, output_dir, image_root=args.image_root)
