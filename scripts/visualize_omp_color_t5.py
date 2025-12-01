
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from transformers import T5Tokenizer, T5EncoderModel

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

def main():
    device = torch.device('cpu')
    model_name = 't5-base'
    
    print(f"Loading {model_name} (Text-to-Text Transfer Transformer)...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Use T5EncoderModel to get the encoder output only
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    def get_t5_embedding(text_list):
        # T5 encoding: averaging the last hidden state across tokens
        with torch.no_grad():
            inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            # outputs.last_hidden_state is [Batch, SeqLen, Dim]
            # Mean pooling over non-padding tokens
            mask = inputs.attention_mask.unsqueeze(-1) # [B, L, 1]
            masked_emb = outputs.last_hidden_state * mask
            sum_emb = masked_emb.sum(dim=1)
            count = mask.sum(dim=1)
            avg_emb = sum_emb / count.clamp(min=1e-9)
            # L2 normalize
            return F.normalize(avg_emb, dim=-1)

    # Define Phrases
    target_text = "a photo of a red car"
    
    # Dictionary: specific subset
    dict_texts = [
        "a photo of a grey car",
        "a photo of a fast car"
    ]
    
    concept_text = "a photo of a red"
    base_text = "a photo of a car"
    bag_text = "a photo of a red bag"

    all_texts = [target_text] + dict_texts + [concept_text, base_text, bag_text]
    
    print(f"Target: {target_text}")
    print(f"Dictionary ({len(dict_texts)} items): {dict_texts}")
    print(f"Concept: {concept_text}")

    # Encode
    print("Encoding with T5...")
    all_emb = get_t5_embedding(all_texts)
    
    n_dict = len(dict_texts)
    target_emb = all_emb[0:1]                 # [1, d]
    dict_emb = all_emb[1:1+n_dict]            # [N, d]
    concept_emb = all_emb[1+n_dict:2+n_dict]  # [1, d]
    base_emb = all_emb[2+n_dict:3+n_dict]     # [1, d]
    bag_emb = all_emb[3+n_dict:4+n_dict]      # [1, d]
    
    # Run OMP
    # Use fewer atoms to be less aggressive (e.g., 2 instead of 5)
    max_atoms = 2
    print(f"Running OMP with {max_atoms} atoms...")
    residual_emb = omp_sparse_residual(target_emb, dict_emb, max_atoms=max_atoms)

    # Prepare for PCA
    plot_embs = torch.cat([target_emb, residual_emb, dict_emb, concept_emb, base_emb, bag_emb], dim=0)
    
    plot_labels = ["Red Car (Orig)", "Red Car (Res)"] + \
                  [t.replace("a photo of a ", "").title() for t in dict_texts] + \
                  ["Red (Concept)", "Car (Base)", "Red Bag"]
    
    plot_data = plot_embs.numpy()
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(plot_data)
    
    sim_orig_concept = F.cosine_similarity(target_emb, concept_emb).item()
    sim_res_concept = F.cosine_similarity(residual_emb, concept_emb).item()
    
    sim_orig_base = F.cosine_similarity(target_emb, base_emb).item()
    sim_res_base = F.cosine_similarity(residual_emb, base_emb).item()
    
    print(f"\nSimilarity Analysis (T5-Base):")
    print(f"  'Red Car' vs 'Red': {sim_orig_concept:.4f} -> Residual vs 'Red': {sim_res_concept:.4f}")
    print(f"  'Red Car' vs 'Car': {sim_orig_base:.4f} -> Residual vs 'Car': {sim_res_base:.4f}")

    # Plotting
    plt.figure(figsize=(14, 12))
    
    idx_dict_start = 2
    idx_dict_end = 2 + n_dict
    idx_concept = 2 + n_dict
    idx_base = 2 + n_dict + 1
    idx_bag = 2 + n_dict + 2
    
    # 1. Plot Dictionary
    plt.scatter(coords[idx_dict_start:idx_dict_end, 0], coords[idx_dict_start:idx_dict_end, 1], 
                c='lightgray', s=80, alpha=0.6, label='Dictionary (Other Car Types)')
    for i in range(n_dict):
        plt.text(coords[idx_dict_start+i, 0]+0.005, coords[idx_dict_start+i, 1]+0.005, 
                 plot_labels[2+i], fontsize=8, alpha=0.6, color='gray')
        
    # 2. Plot Base Concept (Car)
    plt.scatter(coords[idx_base, 0], coords[idx_base, 1], c='black', s=150, marker='s', label="'Car' Concept")
    plt.text(coords[idx_base, 0]+0.01, coords[idx_base, 1]+0.01, plot_labels[idx_base], fontsize=10, weight='bold')

    # 3. Plot Target Concept (Red)
    plt.scatter(coords[idx_concept, 0], coords[idx_concept, 1], c='red', s=200, marker='*', label="'Red' Concept")
    plt.text(coords[idx_concept, 0]+0.01, coords[idx_concept, 1]+0.01, plot_labels[idx_concept], fontsize=12, weight='bold', color='darkred')

    # Plot Red Bag
    plt.scatter(coords[idx_bag, 0], coords[idx_bag, 1], c='purple', s=150, marker='D', label="'Red Bag'")
    plt.text(coords[idx_bag, 0]+0.01, coords[idx_bag, 1]+0.01, plot_labels[idx_bag], fontsize=10, weight='bold', color='purple')

    # 4. Plot Target Original
    plt.scatter(coords[0, 0], coords[0, 1], c='blue', s=150, label='Target (Original)')
    plt.text(coords[0, 0]+0.01, coords[0, 1]+0.01, plot_labels[0], fontsize=10, color='blue')

    # 5. Plot Target Residual
    plt.scatter(coords[1, 0], coords[1, 1], c='green', s=150, marker='^', label='Target (Residual)')
    plt.text(coords[1, 0]+0.01, coords[1, 1]+0.01, plot_labels[1], fontsize=10, color='green')

    # Arrow from Original to Residual
    plt.arrow(coords[0, 0], coords[0, 1], 
              coords[1, 0] - coords[0, 0], coords[1, 1] - coords[0, 1],
              fc='gray', ec='gray', alpha=0.5, width=0.001, head_width=0.02, length_includes_head=True)
    
    plt.plot([coords[1, 0], coords[idx_concept, 0]], [coords[1, 1], coords[idx_concept, 1]], 
             'r--', alpha=0.3, linewidth=1)

    plt.title(f"OMP Effect (T5-Base): 'Red Car' projected away from {n_dict} contexts (Max Atoms: {max_atoms})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = 'outputs/omp_color_vis_t5.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()

