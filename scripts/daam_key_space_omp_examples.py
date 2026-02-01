"""
DAAM Key-Space OMP Examples

Generates visualization examples showing DAAM heatmaps with and without 
Key-Space OMP orthogonalization for each concept separately.
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from daam_segmentation import DAAMSegmenter

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../docs/daam_key_space_omp")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def visualize_multi_concept(image_pil, concepts, base_heatmaps, omp_heatmaps, save_path):
    """
    Create visualization showing DAAM vs OMP for each concept.
    
    Layout:
    Row 1: Original | Concept1 DAAM | Concept1 OMP | Concept2 DAAM | Concept2 OMP | ...
    """
    n_concepts = len(concepts)
    n_cols = 1 + 2 * n_concepts  # Original + (DAAM, OMP) per concept
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Original Image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # For each concept: DAAM and OMP side by side
    for i, concept in enumerate(concepts):
        col_base = 1 + 2 * i
        
        # DAAM heatmap
        axes[col_base].imshow(image_pil)
        axes[col_base].imshow(base_heatmaps[concept].numpy(), cmap='jet', alpha=0.6)
        axes[col_base].set_title(f"DAAM: '{concept}'", fontsize=10)
        axes[col_base].axis('off')
        
        # OMP heatmap
        axes[col_base + 1].imshow(image_pil)
        axes[col_base + 1].imshow(omp_heatmaps[concept].numpy(), cmap='jet', alpha=0.6)
        axes[col_base + 1].set_title(f"OMP: '{concept}'", fontsize=10)
        axes[col_base + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {save_path}")


def run_examples():
    """Run DAAM Key-Space OMP examples on sample images."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize segmenter
    print("=" * 60)
    print("Initializing DAAMSegmenter...")
    print("=" * 60)
    segmenter = DAAMSegmenter(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define examples with all concepts to visualize
    examples = [
        # Example 1: Cat and Dog
        {
            "image": os.path.join(DATA_DIR, "cat_and_dog.jpeg"),
            "concepts": ["dog", "cat", "bird"],
            "name": "cat_and_dog"
        },
        # Example 2: Shepherd Dog
        {
            "image": os.path.join(DATA_DIR, "shepherd_dog.png"),
            "concepts": ["dog", "cat", "grass", "sky"],
            "name": "shepherd_dog"
        },
        # Example 3: Dog and Car
        {
            "image": os.path.join(DATA_DIR, "test_dog_and_car.jpg"),
            "concepts": ["dog", "cat", "bird","car"],
            "name": "dog_and_car"
        },
        # Example 4: Bird
        {
            "image": os.path.join(DATA_DIR, "bird.png"),
            "concepts": ["bird", "tree", "sky"],
            "name": "bird"
        },
        # Example 5: Cat single
        {
            "image": os.path.join(DATA_DIR, "cat.jpg"),
            "concepts": ["cat", "dog", "person"],
            "name": "cat_single"
        },
    ]
    
    for i, ex in enumerate(examples):
        print(f"\n{'=' * 60}")
        print(f"Example {i+1}/{len(examples)}: {ex['name']}")
        print(f"{'=' * 60}")
        
        # Load image
        if not os.path.exists(ex["image"]):
            print(f"  [SKIP] Image not found: {ex['image']}")
            continue
            
        image_pil = Image.open(ex["image"]).convert("RGB")
        concepts = ex["concepts"]
        
        print(f"  Concepts: {concepts}")
        
        base_heatmaps = {}
        omp_heatmaps = {}
        
        # For each concept, compute DAAM and OMP heatmaps
        for concept in concepts:
            prompt = f"a photo of a {concept}."
            # Competing concepts = all other concepts
            competing = [c for c in concepts if c != concept]
            
            print(f"  Processing '{concept}' (competing: {competing})...")
            
            # Base DAAM heatmap
            base_heatmaps[concept] = segmenter.predict(image_pil, prompt)
            
            # Key-Space OMP heatmap
            omp_heatmaps[concept] = segmenter.predict_key_space_omp(
                image_pil,
                prompt=prompt,
                target_concept=concept,
                competing_concepts=competing,
                omp_beta=1.0
            )
        
        # Visualize and save
        save_path = os.path.join(OUTPUT_DIR, f"{ex['name']}_comparison.png")
        visualize_multi_concept(image_pil, concepts, base_heatmaps, omp_heatmaps, save_path)
        
    print(f"\n{'=' * 60}")
    print(f"All examples saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_examples()
