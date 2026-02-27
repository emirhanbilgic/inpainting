#!/usr/bin/env python3
"""
Visualize CLIP text embeddings as point clouds before and after OMP.

Generates multiple prompt variations per concept to create a cloud of
embeddings, then applies OMP and shows how each cloud moves/reshapes.

Usage:
    python scripts/visualize_clip_embeddings_omp.py
    python scripts/visualize_clip_embeddings_omp.py --concepts cat sheep dog horse bird
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from sklearn.decomposition import PCA

# --- PROJECT SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import open_clip


# ==============================================================================
# Custom crescent moon marker
# ==============================================================================
def make_crescent_marker():
    """Create a crescent moon marker as a matplotlib Path."""
    # Outer circle (full)
    theta = np.linspace(0, 2 * np.pi, 60)
    outer_x = np.cos(theta)
    outer_y = np.sin(theta)

    # Inner circle (shifted right to carve out crescent)
    inner_x = 0.55 + 0.75 * np.cos(theta[::-1])
    inner_y = 0.75 * np.sin(theta[::-1])

    verts = (
        list(zip(outer_x, outer_y))
        + [(outer_x[0], outer_y[0])]  # close outer
        + list(zip(inner_x, inner_y))
        + [(inner_x[0], inner_y[0])]  # close inner
    )
    codes = (
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(outer_x) - 1) + [MplPath.CLOSEPOLY]
        + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner_x) - 1) + [MplPath.CLOSEPOLY]
    )
    return MplPath(verts, codes)


CRESCENT_MARKER = make_crescent_marker()

# Default markers per concept name — extendable
CONCEPT_MARKERS = {
    'cat': '*',            # star
    'sheep': CRESCENT_MARKER,  # crescent
    'dog': 'D',            # diamond
    'horse': 'p',          # pentagon
    'bird': '^',           # triangle up
}


# ==============================================================================
# Prompt templates — variations to create a point cloud per concept
# ==============================================================================
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a {} in the wild.",
    "a picture of a {}.",
    "a close-up photo of a {}.",
    "a {} standing in a field.",
    "a {} sitting on the ground.",
    "a small {}.",
    "a large {}.",
    "a {} looking at the camera.",
    "a painting of a {}.",
    "a sketch of a {}.",
    "a {} outdoors.",
    "a {} in nature.",
    "a cute {}.",
    "a fluffy {}.",
    "a {} next to a tree.",
    "an image of a {}.",
    "a {} walking.",
    "a {} resting.",
    "a {} on grass.",
]


# ==============================================================================
# OMP Sparse Residual (from generate_hci_all_methods.py)
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
# Helpers
# ==============================================================================
def get_text_embedding(model, tokenizer, prompt, device):
    """Encode a text prompt into a normalized embedding."""
    tok = tokenizer([prompt]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb


def compute_cloud_embeddings(model, tokenizer, concepts, device, templates):
    """
    Compute original and OMP-residual embeddings for each concept,
    using multiple prompt templates to form a point cloud.
    """
    # 1. Canonical embedding per concept (for the OMP dictionary)
    canonical = {}
    for c in concepts:
        canonical[c] = get_text_embedding(model, tokenizer, f"a photo of a {c}.", device)

    # 2. All variations per concept
    originals = {c: [] for c in concepts}
    for c in concepts:
        for tmpl in templates:
            prompt = tmpl.format(c)
            emb = get_text_embedding(model, tokenizer, prompt, device)
            originals[c].append(emb)

    # 3. For each concept, OMP each variation against all other concepts' canonical embeddings
    omp_results = {c: [] for c in concepts}
    for target in concepts:
        distractors = [c for c in concepts if c != target]
        D = torch.cat([canonical[d] for d in distractors], dim=0)
        for emb in originals[target]:
            omp_emb = omp_sparse_residual(emb, D, max_atoms=len(distractors))
            omp_results[target].append(omp_emb)

    return originals, omp_results, canonical


# ==============================================================================
# Visualization
# ==============================================================================
def visualize_point_clouds(originals, omp_results, concepts, output_path):
    """
    Produce two separate, clean paper-ready figures:
      1. before_omp.{png,pdf}  — original embeddings
      2. after_omp.{png,pdf}   — OMP residual embeddings
    White background, Times New Roman, no title, no legend.
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    # Gather original vectors
    orig_vecs, orig_labels = [], []
    for c in concepts:
        for emb in originals[c]:
            orig_vecs.append(emb.detach().cpu().numpy().flatten())
            orig_labels.append(c)
    orig_vecs = np.stack(orig_vecs)

    # Gather OMP vectors
    omp_vecs, omp_labels = [], []
    for c in concepts:
        for emb in omp_results[c]:
            omp_vecs.append(emb.detach().cpu().numpy().flatten())
            omp_labels.append(c)
    omp_vecs = np.stack(omp_vecs)

    # Separate PCAs
    pca_orig = PCA(n_components=2)
    orig_2d = pca_orig.fit_transform(orig_vecs)

    pca_omp = PCA(n_components=2)
    omp_2d = pca_omp.fit_transform(omp_vecs)

    # Organize by concept
    orig_points = {c: [] for c in concepts}
    omp_points = {c: [] for c in concepts}
    for c, pt in zip(orig_labels, orig_2d):
        orig_points[c].append(pt)
    for c, pt in zip(omp_labels, omp_2d):
        omp_points[c].append(pt)
    for c in concepts:
        orig_points[c] = np.array(orig_points[c])
        omp_points[c] = np.array(omp_points[c])

    # --- Color palette (paper-friendly, strong contrast on white) ---
    palette = {
        0: '#C0392B',  # red (cat)
        1: '#2471A3',  # blue (sheep)
        2: '#1E8449',  # green (dog)
        3: '#B7950B',  # gold
        4: '#884EA0',  # purple
        5: '#D35400',  # orange
    }

    POINT_SIZE = 280
    margin = 0.15
    FONT_SIZE_LABEL = 18
    FONT_SIZE_AXIS = 14
    FONT_SIZE_TICK = 12

    # Helper to draw one figure
    def _draw_figure(data_points, all_pts_2d, save_stem):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Axis limits
        px_min, px_max = all_pts_2d[:, 0].min(), all_pts_2d[:, 0].max()
        py_min, py_max = all_pts_2d[:, 1].min(), all_pts_2d[:, 1].max()
        px_range = max(px_max - px_min, 1e-6)
        py_range = max(py_max - py_min, 1e-6)

        for i, c in enumerate(concepts):
            ci = i % len(palette)
            color = palette[ci]
            pts = data_points[c]
            marker = CONCEPT_MARKERS.get(c, 'o')

            # Point cloud
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=POINT_SIZE, c=color,
                edgecolors='black', linewidths=0.5,
                alpha=0.75, zorder=4,
                marker=marker,
            )

            # Center of mass
            cx, cy = pts.mean(axis=0)
            # Outer ring
            ax.scatter(cx, cy, s=700, c='none', edgecolors='black',
                       linewidths=2.5, zorder=6, marker='o')
            # Inner fill
            ax.scatter(cx, cy, s=400, c=color, edgecolors='black',
                       linewidths=1.5, zorder=7, marker='o')
            # Label
            ax.annotate(
                c, (cx, cy),
                textcoords="offset points", xytext=(18, 14),
                fontsize=FONT_SIZE_LABEL, fontweight='bold', fontstyle='italic',
                color=color,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                zorder=8,
            )

        ax.set_xlim(px_min - margin * px_range, px_max + margin * px_range)
        ax.set_ylim(py_min - margin * py_range, py_max + margin * py_range)
        ax.set_xlabel('PCA Component 1', fontsize=FONT_SIZE_AXIS, labelpad=8)
        ax.set_ylabel('PCA Component 2', fontsize=FONT_SIZE_AXIS, labelpad=8)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        # Clean spines
        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(0.8)

        plt.tight_layout()

        png_path = f"{save_stem}.png"
        pdf_path = f"{save_stem}.pdf"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  → {png_path}")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"  → {pdf_path}")
        plt.close()

    out_dir = os.path.dirname(output_path)
    stem = os.path.join(out_dir, "_".join(concepts))

    print("Figure 1: Before OMP")
    _draw_figure(orig_points, orig_2d, f"{stem}_before_omp")

    print("Figure 2: After OMP")
    _draw_figure(omp_points, omp_2d, f"{stem}_after_omp")

    return pca_orig


def print_cosine_similarities(canonical_orig, canonical_omp, concepts):
    """Print cosine sims between canonical embeddings before/after OMP."""
    print("\n" + "=" * 70)
    print("COSINE SIMILARITIES (canonical embeddings)")
    print("=" * 70)
    print(f"\n{'Pair':<25} {'Before OMP':<15} {'After OMP':<15} {'Delta':<12}")
    print("-" * 67)
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i + 1:]:
            orig_sim = float(F.cosine_similarity(canonical_orig[c1], canonical_orig[c2]))
            omp_sim = float(F.cosine_similarity(canonical_omp[c1], canonical_omp[c2]))
            delta = omp_sim - orig_sim
            print(f"  {c1} ↔ {c2:<18} {orig_sim:<15.4f} {omp_sim:<15.4f} {delta:+.4f}")
    print()


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize CLIP embedding point clouds before/after OMP."
    )
    parser.add_argument(
        "--concepts", nargs='+', default=["cat", "sheep", "dog"],
        help="Concepts to visualize (default: cat sheep dog)"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the figure"
    )
    args = parser.parse_args()

    device = args.device
    concepts = args.concepts

    if args.output is None:
        concepts_str = "_".join(concepts)
        args.output = os.path.join(
            project_root, "scripts", "outputs",
            f"clip_cloud_{concepts_str}.png"
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load CLIP model
    print("Loading CLIP (ViT-B-16)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Compute point cloud embeddings
    print(f"Computing {len(PROMPT_TEMPLATES)} prompt variations × {len(concepts)} concepts...")
    originals, omp_results, canonical = compute_cloud_embeddings(
        model, tokenizer, concepts, device, PROMPT_TEMPLATES
    )

    # Canonical OMP for similarity table
    canonical_omp = {}
    for target in concepts:
        distractors = [c for c in concepts if c != target]
        D = torch.cat([canonical[d] for d in distractors], dim=0)
        canonical_omp[target] = omp_sparse_residual(canonical[target], D, max_atoms=len(distractors))

    print_cosine_similarities(canonical, canonical_omp, concepts)

    # Visualize
    print("Generating point cloud visualizations...")
    pca = visualize_point_clouds(originals, omp_results, concepts, args.output)

    print("\nDone!")
