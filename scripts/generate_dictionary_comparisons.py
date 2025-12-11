#!/usr/bin/env python
"""
Generate 7 different comparison sets with different dictionary compositions for OMP sparse encoding.

Dictionary variations:
1. Only prompts (other classes)
2. Only hypernyms (from WordNet)
3. Only co-hyponyms (siblings from WordNet)
4. Hypernyms + prompts
5. Prompts + co-hyponyms
6. Hypernyms + co-hyponyms
7. All (prompts + hypernyms + co-hyponyms)
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
from typing import List, Dict


def get_wordnet_relations(keyword: str, limit_per_relation: int = 10) -> Dict[str, List[str]]:
    """
    Get WordNet relations: hypernyms and co-hyponyms (siblings).
    Returns dict with keys 'hypernyms' and 'cohyponyms'.
    """
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            print(f"Downloading WordNet data...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn
    except Exception as e:
        print(f"[WordNet] Warning: Failed to load NLTK/WordNet: {e}")
        return {'hypernyms': [], 'cohyponyms': []}
    
    hypernyms = []
    cohyponyms = []
    seen_hyper = set()
    seen_cohy = set()
    key_low = keyword.lower()
    
    synsets = wn.synsets(keyword, pos=wn.NOUN)
    
    for s in synsets[:limit_per_relation]:
        # Get hypernyms
        for h in s.hypernyms()[:limit_per_relation]:
            for l in h.lemmas()[:limit_per_relation]:
                name = l.name().replace('_', ' ').lower()
                if name != key_low and name not in seen_hyper:
                    hypernyms.append(name)
                    seen_hyper.add(name)
        
        # Get co-hyponyms (siblings)
        for h in s.hypernyms()[:limit_per_relation]:
            for sib in h.hyponyms()[:limit_per_relation]:
                if sib != s:  # Exclude self
                    for l in sib.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen_cohy:
                            cohyponyms.append(name)
                            seen_cohy.add(name)
    
    return {
        'hypernyms': hypernyms[:limit_per_relation * 2],
        'cohyponyms': cohyponyms[:limit_per_relation * 2]
    }


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], dictionary atoms (rows), L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    
    x = x_1x.clone()
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)
        c_abs = c.abs()
        
        if len(selected) > 0:
            c_abs[selected] = -1.0
        
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        
        selected.append(idx)
        
        D_S = D[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)
        x_hat = (s.t() @ D_S).to(x.dtype)
        r = (x - x_hat)
        
        if float(torch.norm(r) <= tol):
            break
    
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


def compute_heatmap_legrad(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor) -> torch.Tensor:
    """Compute LeGrad heatmap."""
    with torch.enable_grad():
        heatmap = model.compute_legrad(image=image, text_embedding=text_emb_1x)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def compute_heatmap_gradcam(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor, layer_index: int = 11) -> torch.Tensor:
    """Compute Grad-CAM heatmap. Default layer 11 (last layer) works best with mean pooling."""
    with torch.enable_grad():
        heatmap = model.compute_gradcam(image=image, text_embedding=text_emb_1x, layer_index=layer_index)
    heatmap = heatmap[0, 0].clamp(0, 1).detach().cpu()
    return heatmap


def overlay_heatmap(ax, base_img: Image.Image, heatmap: torch.Tensor, title: str, alpha: float = 0.6):
    """Overlay heatmap on base image."""
    H, W = heatmap.shape
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    ax.imshow(base_resized)
    ax.imshow(heatmap.numpy(), cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9, pad=5)
    ax.axis('off')


def encode_words(tokenizer, model, words: List[str], device: torch.device) -> torch.Tensor:
    """Encode a list of words/phrases into text embeddings."""
    if not words:
        return None
    # Use "a photo of a {word}" format for consistency
    prompts = [f"a photo of a {w}" for w in words]
    tok = tokenizer(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb


def main():
    # Configuration
    data_dir = '/Users/emirhan/Desktop/LeGrad-1/scripts/data'
    base_output_dir = '/Users/emirhan/Desktop/LeGrad-1/comparison_outputs'
    
    image_size = 448
    model_name = 'ViT-B-16'
    pretrained = 'laion2b_s34b_b88k'
    max_atoms = 8
    layer_index = 11  # Last layer works best with mean pooling GradCAM
    max_similarity = 0.9  # Filter out dictionary atoms with similarity >= this value
    
    # Classes and prompts
    classes = ['automobile', 'plane', 'bird', 'cat', 'dog']
    prompts = [f'a photo of a {cls}' for cls in classes]
    
    # Define 7 dictionary configurations
    configs = [
        ('1_only_prompts', {'prompts': True, 'hypernyms': False, 'cohyponyms': False}),
        ('2_only_hypernyms', {'prompts': False, 'hypernyms': True, 'cohyponyms': False}),
        ('3_only_cohyponyms', {'prompts': False, 'hypernyms': False, 'cohyponyms': True}),
        ('4_hypernyms_prompts', {'prompts': True, 'hypernyms': True, 'cohyponyms': False}),
        ('5_prompts_cohyponyms', {'prompts': True, 'hypernyms': False, 'cohyponyms': True}),
        ('6_hypernyms_cohyponyms', {'prompts': False, 'hypernyms': True, 'cohyponyms': True}),
        ('7_all', {'prompts': True, 'hypernyms': True, 'cohyponyms': True}),
    ]
    
    # Create output directories
    for config_name, _ in configs:
        os.makedirs(os.path.join(base_output_dir, config_name), exist_ok=True)
    
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
    
    # Wrap with LeGrad
    model = LeWrapper(model, layer_index=0)
    preprocess = LePreprocess(preprocess=preprocess, image_size=image_size)
    
    # Encode all text prompts
    print("Encoding text prompts...")
    tok = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(tok, normalize=True)
    
    # Get WordNet relations for all classes
    print("\nGathering WordNet relations...")
    wordnet_data = {}
    for cls in classes:
        relations = get_wordnet_relations(cls, limit_per_relation=10)
        wordnet_data[cls] = relations
        print(f"  {cls}: {len(relations['hypernyms'])} hypernyms, {len(relations['cohyponyms'])} co-hyponyms")
        print(f"    Hypernyms: {relations['hypernyms'][:5]}{'...' if len(relations['hypernyms']) > 5 else ''}")
        print(f"    Co-hyponyms: {relations['cohyponyms'][:5]}{'...' if len(relations['cohyponyms']) > 5 else ''}")
    
    # Pre-encode WordNet embeddings for all classes
    print("\nEncoding WordNet embeddings...")
    wordnet_embeddings = {}
    for cls in classes:
        hypernym_words = wordnet_data[cls]['hypernyms']
        cohyponym_words = wordnet_data[cls]['cohyponyms']
        
        wordnet_embeddings[cls] = {
            'hypernyms': encode_words(tokenizer, model, hypernym_words, device) if hypernym_words else None,
            'cohyponyms': encode_words(tokenizer, model, cohyponym_words, device) if cohyponym_words else None,
        }
    
    # Get all images
    image_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])
    
    print(f"\nFound {len(image_files)} images")
    print(f"Classes: {classes}")
    print(f"Processing {len(configs)} dictionary configurations...\n")
    
    # Process each configuration
    for config_name, config_flags in configs:
        output_dir = os.path.join(base_output_dir, config_name)
        print(f"\n{'='*80}")
        print(f"Configuration: {config_name}")
        print(f"  Prompts: {config_flags['prompts']}")
        print(f"  Hypernyms: {config_flags['hypernyms']}")
        print(f"  Co-hyponyms: {config_flags['cohyponyms']}")
        print(f"{'='*80}\n")
        
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
            legrad_heatmaps = []
            sparse_legrad_heatmaps = []
            
            for prompt_idx, prompt in enumerate(prompts):
                cls = classes[prompt_idx]
                text_emb_original = text_embs[prompt_idx:prompt_idx+1]
                
                # Compute regular LeGrad
                heatmap_regular = compute_heatmap_legrad(model, img_tensor, text_emb_original)
                legrad_heatmaps.append((prompt, heatmap_regular))
                
                # Build dictionary based on config
                dict_parts = []
                dict_labels = []
                
                if config_flags['prompts']:
                    # Add other class prompts
                    dict_indices = [i for i in range(len(classes)) if i != prompt_idx]
                    if dict_indices:
                        dict_parts.append(text_embs[dict_indices])
                        dict_labels.extend([classes[i] for i in dict_indices])
                
                if config_flags['hypernyms']:
                    # Add hypernyms
                    hyper_emb = wordnet_embeddings[cls]['hypernyms']
                    if hyper_emb is not None:
                        dict_parts.append(hyper_emb)
                        dict_labels.extend([f"{w}(H)" for w in wordnet_data[cls]['hypernyms']])
                
                if config_flags['cohyponyms']:
                    # Add co-hyponyms
                    cohy_emb = wordnet_embeddings[cls]['cohyponyms']
                    if cohy_emb is not None:
                        dict_parts.append(cohy_emb)
                        dict_labels.extend([f"{w}(C)" for w in wordnet_data[cls]['cohyponyms']])
                
                # Create dictionary
                if dict_parts:
                    D = torch.cat(dict_parts, dim=0)
                    D = F.normalize(D, dim=-1)
                    
                    # Filter out atoms with similarity >= max_similarity to avoid super similar prompts
                    sim = (D @ text_emb_original.t()).squeeze(-1).abs()
                    keep_mask = sim < max_similarity
                    
                    if keep_mask.sum() > 0:
                        D = D[keep_mask]
                        # Also filter dict_labels for debugging
                        dict_labels = [label for i, label in enumerate(dict_labels) if keep_mask[i]]
                        if sim.numel() > 0:
                            dropped = (~keep_mask).sum().item()
                            if dropped > 0:
                                print(f"      [{cls}] Filtered {dropped}/{len(sim)} atoms with sim >= {max_similarity}")
                    else:
                        D = text_emb_original.new_zeros((0, text_emb_original.shape[-1]))
                else:
                    D = text_emb_original.new_zeros((0, text_emb_original.shape[-1]))
                
                # Apply OMP
                if D.numel() > 0:
                    text_emb_sparse = omp_sparse_residual(text_emb_original, D, max_atoms=max_atoms)
                else:
                    text_emb_sparse = text_emb_original
                
                # Compute sparse LeGrad
                heatmap_sparse = compute_heatmap_legrad(model, img_tensor, text_emb_sparse)
                sparse_legrad_heatmaps.append((prompt, heatmap_sparse))
            
            # Create LeGrad comparison figure
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            for col, (prompt, heatmap) in enumerate(legrad_heatmaps):
                overlay_heatmap(axes[0, col], base_img, heatmap, f'LeGrad\n{prompt}')
            for col, (prompt, heatmap) in enumerate(sparse_legrad_heatmaps):
                overlay_heatmap(axes[1, col], base_img, heatmap, f'Sparse LeGrad\n{prompt}')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'comparison_legrad_{img_name}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: comparison_legrad_{img_name}.png")
            
            # --- Process Grad-CAM ---
            print("  Computing Grad-CAM heatmaps...")
            gradcam_heatmaps = []
            sparse_gradcam_heatmaps = []
            
            for prompt_idx, prompt in enumerate(prompts):
                cls = classes[prompt_idx]
                text_emb_original = text_embs[prompt_idx:prompt_idx+1]
                
                # Compute regular Grad-CAM
                heatmap_regular = compute_heatmap_gradcam(model, img_tensor, text_emb_original, layer_index)
                gradcam_heatmaps.append((prompt, heatmap_regular))
                
                # Build dictionary (same as LeGrad)
                dict_parts = []
                dict_labels = []
                
                if config_flags['prompts']:
                    dict_indices = [i for i in range(len(classes)) if i != prompt_idx]
                    if dict_indices:
                        dict_parts.append(text_embs[dict_indices])
                        dict_labels.extend([classes[i] for i in dict_indices])
                
                if config_flags['hypernyms']:
                    hyper_emb = wordnet_embeddings[cls]['hypernyms']
                    if hyper_emb is not None:
                        dict_parts.append(hyper_emb)
                        dict_labels.extend([f"{w}(H)" for w in wordnet_data[cls]['hypernyms']])
                
                if config_flags['cohyponyms']:
                    cohy_emb = wordnet_embeddings[cls]['cohyponyms']
                    if cohy_emb is not None:
                        dict_parts.append(cohy_emb)
                        dict_labels.extend([f"{w}(C)" for w in wordnet_data[cls]['cohyponyms']])
                
                if dict_parts:
                    D = torch.cat(dict_parts, dim=0)
                    D = F.normalize(D, dim=-1)
                    
                    # Filter out atoms with similarity >= max_similarity to avoid super similar prompts
                    sim = (D @ text_emb_original.t()).squeeze(-1).abs()
                    keep_mask = sim < max_similarity
                    
                    if keep_mask.sum() > 0:
                        D = D[keep_mask]
                        dict_labels = [label for i, label in enumerate(dict_labels) if keep_mask[i]]
                    else:
                        D = text_emb_original.new_zeros((0, text_emb_original.shape[-1]))
                else:
                    D = text_emb_original.new_zeros((0, text_emb_original.shape[-1]))
                
                # Apply OMP
                if D.numel() > 0:
                    text_emb_sparse = omp_sparse_residual(text_emb_original, D, max_atoms=max_atoms)
                else:
                    text_emb_sparse = text_emb_original
                
                # Compute sparse Grad-CAM
                heatmap_sparse = compute_heatmap_gradcam(model, img_tensor, text_emb_sparse, layer_index)
                sparse_gradcam_heatmaps.append((prompt, heatmap_sparse))
            
            # Create Grad-CAM comparison figure
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            for col, (prompt, heatmap) in enumerate(gradcam_heatmaps):
                overlay_heatmap(axes[0, col], base_img, heatmap, f'Grad-CAM\n{prompt}')
            for col, (prompt, heatmap) in enumerate(sparse_gradcam_heatmaps):
                overlay_heatmap(axes[1, col], base_img, heatmap, f'Sparse Grad-CAM\n{prompt}')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'comparison_gradcam_{img_name}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: comparison_gradcam_{img_name}.png")
    
    print(f"\n{'='*80}")
    print(f"All done! Generated comparisons for {len(image_files)} images across {len(configs)} configurations.")
    print(f"Total: {len(image_files) * len(configs) * 2} comparison images")
    print(f"Base output directory: {base_output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

