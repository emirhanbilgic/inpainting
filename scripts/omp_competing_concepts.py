#!/usr/bin/env python
import argparse
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from scripts.daam_segmentation import DAAMSegmenter
except ImportError:
    try:
        from daam_segmentation import DAAMSegmenter
    except ImportError:
        print("Error: Could not import DAAMSegmenter. Make sure you are in the project root or scripts directory.")
        sys.exit(1)

def overlay_heatmap(ax, image, heatmap, title):
    """Overlay heatmap on image."""
    # Ensure heatmap is a numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()
        
    H, W = heatmap.shape
    # Resize original image to match heatmap dimensions for correct overlay
    base_resized = image.resize((W, H), Image.BICUBIC).convert("RGB")
    
    ax.imshow(base_resized)
    ax.imshow(heatmap, cmap='jet', alpha=0.6, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def main():
    parser = argparse.ArgumentParser(description='DAAM Competing Concepts (Sparse Encoding)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--concepts', type=str, nargs='+', 
                        default=['dog', 'cat', 'automobile', 'plane', 'bird'],
                        help='List of competing concepts')
    parser.add_argument('--model_id', type=str, default='Manojb/stable-diffusion-2-base')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--atoms', type=int, default=8, help='Max atoms for OMP')
    parser.add_argument('--output_dir', type=str, default='outputs/omp_competing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DAAM Segmenter
    print(f"Loading DAAM Segmenter ({args.model_id})...")
    seg = DAAMSegmenter(model_id=args.model_id, device=args.device)
    
    # Load Image
    try:
        raw_image = Image.open(args.image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    prompts = [f"a photo of a {c}." for c in args.concepts]
    print(f"Concepts: {args.concepts}")
    
    # Setup plot: Rows = Methods [DAAM, Sparse DAAM], Cols = Concepts
    num_concepts = len(args.concepts)
    fig, axes = plt.subplots(2, num_concepts, figsize=(4 * num_concepts, 8))
    
    # Handle edge case where there is only 1 concept (axes would be [2,])
    if num_concepts == 1:
        axes = axes.reshape(2, 1)
    
    for i, concept in enumerate(args.concepts):
        prompt = prompts[i]
        print(f"Processing concept: {concept}...")
        
        # 1. Standard DAAM (Row 0, Col i)
        heatmap_daam = seg.predict(raw_image, prompt, size=args.image_size)
        overlay_heatmap(axes[0, i], raw_image, heatmap_daam, f"DAAM\n{prompt}")
        
        # 2. Key-Space OMP (Row 1, Col i)
        competing_base = [c for j, c in enumerate(args.concepts) if j != i]
        
        print(f"  Running Key-Space OMP (Internal) for: {competing_base}")
        
        heatmap_omp = seg.predict_key_space_omp(
            raw_image, 
            prompt, 
            target_concept=concept,
            competing_concepts=competing_base,
            omp_beta=1.0, # Full orthogonalization
            size=args.image_size
        )
        overlay_heatmap(axes[1, i], raw_image, heatmap_omp, f"Key-Space OMP\n{prompt}")

    plt.tight_layout()
    output_filename = os.path.join(args.output_dir, f"daam_key_space_omp_{os.path.basename(args.image_path)}")
    plt.savefig(output_filename, dpi=150)
    print(f"Saved result to {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()
