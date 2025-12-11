#!/usr/bin/env python
"""
Generate comparison heatmaps for LeGrad and Grad-CAM methods with and without OMP sparse encoding.
For each image, creates two comparison figures:
1. comparison_legrad_[imagename].png: LeGrad vs Sparse LeGrad
2. comparison_gradcam_[imagename].png: Grad-CAM vs Sparse Grad-CAM

When computing sparse encodings with OMP, the dictionary contains all class embeddings EXCEPT the target class.
"""
import os
import sys

# Ensure we import from the local legrad module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import open_clip
from legrad import LeWrapper, LePreprocess


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], dictionary atoms (rows), L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()  # residual starts as x
    
    for _ in range(max_atoms):
        # Correlations with residual
        c = (r @ D.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        
        # Mask already selected atoms
        if len(selected) > 0:
            c_abs[selected] = -1.0
        
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        
        selected.append(idx)
        
        # Solve least squares on selected atoms
        D_S = D[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        
        # Regularize G for stability
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t, 1]
        x_hat = (s.t() @ D_S).to(x.dtype)  # [1, d]
        r = (x - x_hat)
        
        # Early stop if residual very small
        if float(torch.norm(r) <= tol):
            break
    
    # Return normalized residual
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


def compute_heatmap_legrad(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor) -> torch.Tensor:
    """
    Compute LeGrad heatmap.
    text_emb_1x: [1, d] normalized text embedding
    Returns: [H, W] heatmap in [0, 1]
    """
    with torch.enable_grad():
        heatmap = model.compute_legrad(image=image, text_embedding=text_emb_1x)  # [1, 1, H, W]
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def compute_heatmap_gradcam(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor, layer_index: int = 11) -> torch.Tensor:
    """
    Compute Grad-CAM heatmap.
    text_emb_1x: [1, d] normalized text embedding
    layer_index: Default 11 (last layer) works best with mean pooling approach.
    Returns: [H, W] heatmap in [0, 1]
    """
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)  # [1, 1, H, W]
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def overlay_heatmap(ax, base_img: Image.Image, heatmap: torch.Tensor, title: str, alpha: float = 0.6):
    """
    Overlay heatmap on base image.
    heatmap: [H, W] float in [0, 1]
    """
    H, W = heatmap.shape
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    ax.imshow(base_resized)
    ax.imshow(heatmap.numpy(), cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10, pad=5)
    ax.axis('off')


def main():
    # Configuration
    data_dir = '/Users/emirhan/Desktop/LeGrad-1/scripts/data'
    output_dir = '/Users/emirhan/Desktop/LeGrad-1/comparison_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    image_size = 448
    model_name = 'ViT-B-16'
    pretrained = 'laion2b_s34b_b88k'
    max_atoms = 8  # For OMP
    layer_index = 11  # For Grad-CAM (last layer works best with mean pooling)
    
    # Classes and prompts
    classes = ['car', 'plane', 'bird', 'cat', 'dog']
    prompts = [f'a photo of a {cls}' for cls in classes]
    
    print(f"Setting up model: {model_name} ({pretrained})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model.eval()
    
    # Wrap with LeGrad (include all layers for LeGrad)
    model = LeWrapper(model, layer_index=0)
    preprocess = LePreprocess(preprocess=preprocess, image_size=image_size)
    
    # Verify methods exist
    if not hasattr(model, 'compute_gradcam'):
        print("ERROR: compute_gradcam method not found on LeWrapper!")
        print(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
        sys.exit(1)
    
    # Encode all text prompts
    print("Encoding text prompts...")
    tok = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tok, normalize=True)  # [5, d]
    
    # Get all images
    image_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    
    print(f"Found {len(image_files)} images")
    print(f"Classes: {classes}")
    print(f"Output directory: {output_dir}\n")
    
    # Process each image
    for img_idx, img_file in enumerate(image_files):
        img_path = os.path.join(data_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        print(f"[{img_idx + 1}/{len(image_files)}] Processing: {img_file}")
        
        # Load and preprocess image
        base_img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(base_img).unsqueeze(0).to(device)
        
        # --- Process LeGrad ---
        print("  Computing LeGrad heatmaps...")
        legrad_heatmaps = []  # Row 1: Regular LeGrad
        sparse_legrad_heatmaps = []  # Row 2: Sparse LeGrad
        
        for prompt_idx, prompt in enumerate(prompts):
            # Original text embedding
            text_emb_original = text_embs[prompt_idx:prompt_idx+1]  # [1, d]
            
            # Compute regular LeGrad heatmap
            heatmap_regular = compute_heatmap_legrad(model, img_tensor, text_emb_original)
            legrad_heatmaps.append((prompt, heatmap_regular))
            
            # Build dictionary: all classes EXCEPT current one
            dict_indices = [i for i in range(len(classes)) if i != prompt_idx]
            D = text_embs[dict_indices]  # [4, d]
            
            # Apply OMP to get sparse residual
            text_emb_sparse = omp_sparse_residual(text_emb_original, D, max_atoms=max_atoms)
            
            # Compute sparse LeGrad heatmap
            heatmap_sparse = compute_heatmap_legrad(model, img_tensor, text_emb_sparse)
            sparse_legrad_heatmaps.append((prompt, heatmap_sparse))
        
        # Create LeGrad comparison figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Row 1: Regular LeGrad
        for col, (prompt, heatmap) in enumerate(legrad_heatmaps):
            overlay_heatmap(axes[0, col], base_img, heatmap, f'LeGrad\n{prompt}')
        
        # Row 2: Sparse LeGrad
        for col, (prompt, heatmap) in enumerate(sparse_legrad_heatmaps):
            overlay_heatmap(axes[1, col], base_img, heatmap, f'Sparse LeGrad\n{prompt}')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'comparison_legrad_{img_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: comparison_legrad_{img_name}.png")
        
        # --- Process Grad-CAM ---
        print("  Computing Grad-CAM heatmaps...")
        gradcam_heatmaps = []  # Row 1: Regular Grad-CAM
        sparse_gradcam_heatmaps = []  # Row 2: Sparse Grad-CAM
        
        for prompt_idx, prompt in enumerate(prompts):
            # Original text embedding
            text_emb_original = text_embs[prompt_idx:prompt_idx+1]  # [1, d]
            
            # Compute regular Grad-CAM heatmap
            heatmap_regular = compute_heatmap_gradcam(model, img_tensor, text_emb_original, layer_index)
            gradcam_heatmaps.append((prompt, heatmap_regular))
            
            # Build dictionary: all classes EXCEPT current one
            dict_indices = [i for i in range(len(classes)) if i != prompt_idx]
            D = text_embs[dict_indices]  # [4, d]
            
            # Apply OMP to get sparse residual
            text_emb_sparse = omp_sparse_residual(text_emb_original, D, max_atoms=max_atoms)
            
            # Compute sparse Grad-CAM heatmap
            heatmap_sparse = compute_heatmap_gradcam(model, img_tensor, text_emb_sparse, layer_index)
            sparse_gradcam_heatmaps.append((prompt, heatmap_sparse))
        
        # Create Grad-CAM comparison figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Row 1: Regular Grad-CAM
        for col, (prompt, heatmap) in enumerate(gradcam_heatmaps):
            overlay_heatmap(axes[0, col], base_img, heatmap, f'Grad-CAM\n{prompt}')
        
        # Row 2: Sparse Grad-CAM
        for col, (prompt, heatmap) in enumerate(sparse_gradcam_heatmaps):
            overlay_heatmap(axes[1, col], base_img, heatmap, f'Sparse Grad-CAM\n{prompt}')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'comparison_gradcam_{img_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: comparison_gradcam_{img_name}.png")
        print()
    
    print(f"\nAll done! Generated comparison images for {len(image_files)} images.")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

