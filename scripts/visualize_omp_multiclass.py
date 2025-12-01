
import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual without training.
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()  # residual starts as x
    
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
    return F.normalize(r, dim=-1)

def get_embeddings_for_class(model, tokenizer, class_name):
    templates = [
        "a photo of a {}", "a small {}", "a large {}", "a picture of the {}", 
        "the {}", "a drawing of a {}", "a sketch of a {}", "a real {}",
        "a {} looking at camera", "a {} in the wild", "closeup of a {}", "image of {}"
    ]
    prompts = [t.format(class_name) for t in templates]
    with torch.no_grad():
        tok = tokenizer(prompts)
        emb = model.encode_text(tok, normalize=True)
    return emb, prompts

def main():
    device = torch.device('cpu')
    model_name = 'ViT-B-16'
    pretrained = 'laion2b_s34b_b88k'
    
    print(f"Loading {model_name}...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name=model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Classes to visualize
    classes = ["raven", "cowbird", "bronzed cowbird", "shiny cowbird"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    
    print(f"Classes: {classes}")
    
    # 1. Collect all embeddings
    class_embeddings = {}
    all_embeddings_list = []
    
    for cls in classes:
        emb, prompts = get_embeddings_for_class(model, tokenizer, cls)
        class_embeddings[cls] = emb
        all_embeddings_list.append(emb)
        
    # 2. Compute Residuals for EACH class against OTHERS
    class_residuals = {}
    all_residuals_list = []
    
    print("Computing residuals...")
    for i, target_cls in enumerate(classes):
        target_emb = class_embeddings[target_cls]
        
        # Build dictionary from ALL OTHER classes
        dict_parts = []
        for other_cls in classes:
            if other_cls == target_cls:
                continue
            dict_parts.append(class_embeddings[other_cls])
        
        full_dict = torch.cat(dict_parts, dim=0)
        
        # Compute residual for each prompt in target class
        res_list = []
        for j in range(target_emb.shape[0]):
            # Using fewer atoms since we just want to remove the main shared "concepts"
            # e.g. remove "animal" from cat (shared with dog)
            r = omp_sparse_residual(target_emb[j:j+1], full_dict, max_atoms=8)
            res_list.append(r)
        
        res_tensor = torch.cat(res_list, dim=0)
        class_residuals[target_cls] = res_tensor
        all_residuals_list.append(res_tensor)

    # 3. PCA on Everything (Original + Residuals)
    # We want a shared space to see how they move relative to each other
    
    all_orig = torch.cat(all_embeddings_list, dim=0)
    all_res = torch.cat(all_residuals_list, dim=0)
    combined = torch.cat([all_orig, all_res], dim=0).numpy()
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    
    # 4. Plot
    plt.figure(figsize=(14, 12))
    
    total_samples = all_orig.shape[0]
    samples_per_class = total_samples // len(classes)
    
    # Plot loop
    for i, cls in enumerate(classes):
        start_idx = i * samples_per_class
        end_idx = start_idx + samples_per_class
        
        # Original Coordinates
        orig_x = coords[start_idx:end_idx, 0]
        orig_y = coords[start_idx:end_idx, 1]
        
        # Residual Coordinates (offset by total_samples in the combined array)
        res_start = total_samples + start_idx
        res_end = total_samples + end_idx
        res_x = coords[res_start:res_end, 0]
        res_y = coords[res_start:res_end, 1]
        
        c = colors[i % len(colors)]
        
        # Scatter Original
        plt.scatter(orig_x, orig_y, c=c, label=f'{cls} (Original)', 
                   alpha=0.6, s=60, marker='o', edgecolors='white')
        
        # Scatter Residual
        plt.scatter(res_x, res_y, c=c, label=f'{cls} (Residual)', 
                   alpha=0.9, s=80, marker='^', edgecolors='black')
        
        # Draw Arrows
        for k in range(len(orig_x)):
            plt.arrow(orig_x[k], orig_y[k], 
                     res_x[k] - orig_x[k], res_y[k] - orig_y[k],
                     fc=c, ec=c, alpha=0.3, width=0.002, head_width=0.025, length_includes_head=True)
            
        # Draw a centroid path for clarity (thicker arrow)
        cent_orig_x, cent_orig_y = orig_x.mean(), orig_y.mean()
        cent_res_x, cent_res_y = res_x.mean(), res_y.mean()
        plt.annotate("", xy=(cent_res_x, cent_res_y), xytext=(cent_orig_x, cent_orig_y),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))

    plt.title("Multi-Class OMP Disentanglement\nEach class projects away from all others")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    out_path = 'outputs/omp_multiclass_vis.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()

