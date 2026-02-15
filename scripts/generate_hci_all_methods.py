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
if not _is_daam:
    problematic_prefixes += ["sklearn", "transformers", "tokenizers"]
for prefix in problematic_prefixes:
    sys.modules[prefix] = create_mock_module(prefix)

problematic_specifics = [
    "matplotlib.pyplot", "matplotlib.colors", "matplotlib.cm",
]
if not _is_daam:
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

# --- DATA ---
ITEMS = [
    ("2007_000904.jpg", ["cow", "horse"]),
    ("2007_001594.jpg", ["dog", "sheep"]),
    ("2007_001763.jpg", ["cat", "dog"]),
    ("2007_001825.jpg", ["cat", "dog"]),
    ("2007_002268.jpg", ["cat", "dog"]),
    ("2007_002597.jpg", ["cat", "dog"]),
    ("2007_004537.jpg", ["cow", "horse"]),
    ("2007_005114.jpg", ["cow", "horse"]),
    ("2007_006944.jpg", ["horse", "sheep"]),
    ("2007_007417.jpg", ["cat", "dog"]),
    ("2007_009331.jpg", ["dog", "horse"]),
    ("2008_002536.jpg", ["dog", "sheep"]),
    ("2009_001885.jpg", ["dog", "horse"]),
    ("2010_002763.jpg", ["cat", "cow"]),
    ("2010_003670.jpg", ["dog", "sheep"]),
    ("2010_004760.jpg", ["cat", "dog"]),
    ("2010_005796.jpg", ["cat", "dog"]),
    ("2011_000219.jpg", ["bird", "cow"]),
    ("2011_000548.jpg", ["bird", "cow"]),
    ("2011_000834.jpg", ["bird", "sheep"]),
    ("2011_002464.jpg", ["bird", "sheep"]),
]

CANDIDATE_ANIMALS = ["bird", "cat", "cow", "dog", "horse", "sheep"]

DATASET_ROOT = "/Users/emirhan/Desktop/pascal-voc-2012-DatasetNinja"
OUTPUT_DIR = os.path.join(project_root, "web_application", "data")


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


def find_image(img_name):
    """Find image in Pascal VOC dataset directories."""
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


def run_clip_method(method_name, model, tokenizer, preprocess, device):
    """Run a CLIP-based method on all images."""
    print(f"\n{'='*60}")
    print(f"Running method: {method_name}")
    print(f"{'='*60}")

    for img_name, real_classes in tqdm(ITEMS, desc=f"{method_name}"):
        img_id = os.path.splitext(img_name)[0]
        img_path = find_image(img_name)
        if not img_path:
            print(f"  [SKIP] Image not found: {img_name}")
            continue

        try:
            original_image = Image.open(img_path).convert("RGB")
            img_t = preprocess(original_image).unsqueeze(0).to(device)

            # Select fake animal
            random.seed(42 + hash(img_id))
            fake_candidates = [a for a in CANDIDATE_ANIMALS if a not in real_classes]
            fake_animal = random.choice(fake_candidates)

            # Build embeddings for all relevant classes
            class_embeddings = {}
            for cls_name in real_classes + [fake_animal]:
                prompt = f"a photo of a {cls_name}."
                class_embeddings[cls_name] = get_text_embedding(model, tokenizer, prompt, device)

            # Process each target
            target_info = [
                ("Target_1", real_classes[0]),
                ("Target_2", real_classes[1]),
                ("Target_fake", fake_animal),
            ]

            for folder_name, target_cls in target_info:
                out_folder = os.path.join(OUTPUT_DIR, img_id, f"{folder_name}_{method_name}")
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


def run_daam_method(device):
    """Run DAAM method on all images."""
    from daam_segmentation import DAAMSegmenter
    from optimize_anti_hallucination import run_daam_with_key_space_omp

    print(f"\n{'='*60}")
    print(f"Running method: DAAM")
    print(f"{'='*60}")

    print("Loading DAAM segmenter...")
    segmenter = DAAMSegmenter(model_id="Manojb/stable-diffusion-2-base", device=device)

    # DAAMSegmenter loads pipeline in float16 which causes NaN on MPS/CPU.
    # Convert to float32 for non-CUDA devices.
    if device != "cuda":
        segmenter.pipeline = segmenter.pipeline.to(torch.float32)
        segmenter.vae = segmenter.pipeline.vae
        segmenter.unet = segmenter.pipeline.unet
        segmenter.text_encoder = segmenter.pipeline.text_encoder
        print(f"  [DAAM] Converted pipeline to float32 for {device}")

    for img_name, real_classes in tqdm(ITEMS, desc="DAAM"):
        img_id = os.path.splitext(img_name)[0]
        img_path = find_image(img_name)
        if not img_path:
            print(f"  [SKIP] Image not found: {img_name}")
            continue

        try:
            original_image = Image.open(img_path).convert("RGB")

            # Select fake animal (same seed as CLIP methods)
            random.seed(42 + hash(img_id))
            fake_candidates = [a for a in CANDIDATE_ANIMALS if a not in real_classes]
            fake_animal = random.choice(fake_candidates)

            target_info = [
                ("Target_1", real_classes[0]),
                ("Target_2", real_classes[1]),
                ("Target_fake", fake_animal),
            ]

            for folder_name, target_cls in target_info:
                out_folder = os.path.join(OUTPUT_DIR, img_id, f"{folder_name}_DAAM")
                os.makedirs(out_folder, exist_ok=True)

                prompt = f"a photo of a {target_cls}."

                # 1. Normal DAAM heatmap
                heatmap_normal = segmenter.predict(original_image, prompt, size=512)
                vis_normal = get_heatmap_vis(original_image, heatmap_normal)

                # 2. DAAM with key-space OMP
                is_fake = folder_name == "Target_fake"
                if is_fake:
                    competing = real_classes
                else:
                    other_real = [c for c in real_classes if c != target_cls]
                    competing = other_real + [fake_animal]

                heatmap_omp = run_daam_with_key_space_omp(
                    segmenter, original_image, target_cls, competing, beta=1.0, size=512
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


def main():
    parser = argparse.ArgumentParser(description="Generate HCI heatmaps for multiple methods")
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS,
                        help="Attribution method to run")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.method in CLIP_METHODS:
        print("Loading CLIP (ViT-B-16)...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        model.eval()
        model = LeWrapper(model, layer_index=-2)
        preprocess = LePreprocess(preprocess=preprocess, image_size=224)

        run_clip_method(args.method, model, tokenizer, preprocess, device)

    elif args.method == "DAAM":
        run_daam_method(device)

    print(f"\nDone! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
