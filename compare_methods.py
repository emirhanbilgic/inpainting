"""
Comparison script for LeGrad vs GradCAM
This script visualizes how the two methods create different heatmaps for the same image and text prompt.
"""

import os
from PIL import Image
import open_clip
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from legrad import LeWrapper, LePreprocess
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def unnormalize_image(tensor_image):
    """Convert normalized tensor back to displayable image"""
    if tensor_image.ndim > 3:
        tensor_image = tensor_image.squeeze(0)
    
    image_unormed = (tensor_image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                    + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]
    image_pil = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    return image_pil

def create_heatmap_overlay(image_pil, heatmap_tensor, alpha=0.6):
    """Create a heatmap overlay on the image"""
    W, H = heatmap_tensor.shape[-2:]
    image_resized = image_pil.resize((W, H))
    
    # Convert heatmap to numpy
    if heatmap_tensor.ndim > 2:
        heatmap_tensor = heatmap_tensor.squeeze()
    heatmap = heatmap_tensor.detach().cpu().numpy()
    
    # Convert to colormap
    heatmap_uint8 = (heatmap * 255).astype('uint8')
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    img_array = np.array(image_resized)
    overlay = (1 - alpha) * img_array + alpha * heatmap_colored
    overlay = overlay.astype('uint8')
    
    return overlay, heatmap

def compare_methods(image_path, text_prompt, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', 
                   save_path='comparison_output.png', layer_index=8):
    """
    Compare LeGrad and GradCAM on a given image and text prompt
    
    Args:
        image_path: Path to the input image
        text_prompt: Text prompt for explanation (e.g., 'a photo of a cat')
        model_name: Vision model architecture
        pretrained: Pretrained weights to use
        save_path: Where to save the comparison figure
        layer_index: Which layer to use for GradCAM (default: 8 for middle layer of ViT-B-16)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------- Initialize model -------
    print(f"Loading model: {model_name} with {pretrained}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, 
        pretrained=pretrained, 
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model.eval()
    
    # ------- Equip the model with LeGrad -------
    print("Wrapping model with LeGrad...")
    # Use layer_index=6 to enable gradients from layer 6 onwards (so layer 8 is accessible)
    model = LeWrapper(model, layer_index=6)
    preprocess = LePreprocess(preprocess=preprocess, image_size=448)
    
    # ------- Load and preprocess image -------
    print(f"Loading image: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # ------- Tokenize text -------
    text_tokens = tokenizer([text_prompt]).to(device)
    text_embedding = model.encode_text(text_tokens, normalize=True)
    
    print(f"Image shape: {image_tensor.shape}")
    print(f"Text prompt: '{text_prompt}'")
    
    # ------- Compute LeGrad -------
    print("\nComputing LeGrad (multi-layer attention gradients)...")
    with torch.no_grad():
        pass  # Clear any previous computations
    
    legrad_map = model.compute_legrad(
        text_embedding=text_embedding, 
        image=image_tensor
    )
    
    # ------- Compute GradCAM -------
    print(f"Computing GradCAM (single-layer features at layer {layer_index})...")
    gradcam_map = model.compute_gradcam(
        text_embedding=text_embedding, 
        image=image_tensor,
        layer_index=layer_index
    )
    
    # ------- Create visualizations -------
    print("Creating visualizations...")
    
    # Unnormalize image for display
    image_pil = unnormalize_image(image_tensor)
    
    # Create overlays
    legrad_overlay, legrad_heatmap = create_heatmap_overlay(image_pil, legrad_map[0], alpha=0.6)
    gradcam_overlay, gradcam_heatmap = create_heatmap_overlay(image_pil, gradcam_map[0], alpha=0.6)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: LeGrad
    axes[0, 0].imshow(image_pil)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(legrad_heatmap, cmap='jet')
    axes[0, 1].set_title('LeGrad Heatmap\n(Multi-layer attention gradients)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(legrad_overlay)
    axes[0, 2].set_title('LeGrad Overlay', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: GradCAM
    axes[1, 0].imshow(image_pil)
    axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gradcam_heatmap, cmap='jet')
    axes[1, 1].set_title(f'GradCAM Heatmap\n(Layer {layer_index} feature gradients)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(gradcam_overlay)
    axes[1, 2].set_title('GradCAM Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add overall title
    fig.suptitle(f'LeGrad vs GradCAM Comparison\nPrompt: "{text_prompt}"', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {save_path}")
    
    # Show figure
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    print(f"LeGrad - Min: {legrad_map.min():.4f}, Max: {legrad_map.max():.4f}, Mean: {legrad_map.mean():.4f}")
    print(f"GradCAM - Min: {gradcam_map.min():.4f}, Max: {gradcam_map.max():.4f}, Mean: {gradcam_map.mean():.4f}")
    print("="*60)
    
    return legrad_map, gradcam_map, image_pil


def compare_multiple_images(data_dir, prompts_dict, output_dir='comparison_outputs'):
    """
    Compare LeGrad and GradCAM on multiple images
    
    Args:
        data_dir: Directory containing images
        prompts_dict: Dictionary mapping image names to text prompts
                      e.g., {'cat.jpg': 'a photo of a cat', 'dog.jpeg': 'a photo of a dog'}
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name, prompt in prompts_dict.items():
        image_path = os.path.join(data_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue
        
        output_name = os.path.splitext(image_name)[0] + '_comparison.png'
        output_path = os.path.join(output_dir, output_name)
        
        print("\n" + "="*80)
        print(f"Processing: {image_name}")
        print("="*80)
        
        try:
            compare_methods(
                image_path=image_path,
                text_prompt=prompt,
                save_path=output_path
            )
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue


if __name__ == "__main__":
    # Example 1: Single image comparison
    print("="*80)
    print("EXAMPLE 1: Single Image Comparison")
    print("="*80)
    
    data_dir = "/Users/emirhan/Desktop/LeGrad-1/scripts/data"
    
    # Compare on a cat image
    compare_methods(
        image_path=os.path.join(data_dir, "cat.jpg"),
        text_prompt="a photo of a cat",
        save_path="legrad_vs_gradcam_cat.png",
        layer_index=8
    )
    
    # Example 2: Multiple images
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Multiple Images Comparison")
    print("="*80)
    
    prompts = {
        'cat.jpg': 'a photo of a cat',
        'dog.jpeg': 'a photo of a dog',
        'car.jpg': 'a photo of a car',
        'plane.jpg': 'a photo of an airplane',
        'cat_and_dog.jpeg': 'a photo of a cat',
    }
    
    compare_multiple_images(
        data_dir=data_dir,
        prompts_dict=prompts,
        output_dir='comparison_outputs'
    )
    
    print("\n" + "="*80)
    print("ALL COMPARISONS COMPLETE!")
    print("="*80)

