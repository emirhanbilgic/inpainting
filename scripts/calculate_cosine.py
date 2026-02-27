import os
import sys
import torch
import numpy as np
import open_clip

# Add project root and scripts dir to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external', 'clip_text_span'))

from utils.imagenet_classes import imagenet_classes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using device: {device}")
    
    # Load model
    model_name = 'ViT-B-32'
    pretrained = 'openai'
    print(f"Loading {model_name} ({pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    
    # Compute embeddings
    print("Computing text embeddings for 1000 ImageNet classes...")
    prompts = [f"a photo of a {c}." for c in imagenet_classes]
    
    with torch.no_grad():
        # Batch size of 250 to avoid memory issues
        all_embeddings = []
        batch_size = 250
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            text_tokens = tokenizer(batch_prompts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu())
            
        embeddings = torch.cat(all_embeddings, dim=0)
    
    # Calculate pairwise cosine similarity
    print("Calculating pairwise cosine similarities...")
    # matrix multiplication since vectors are normalized
    similarity_matrix = (embeddings @ embeddings.T).numpy()
    
    # Analyze the similarities
    n_classes = len(imagenet_classes)
    
    # Get all off-diagonal values
    # We can use triu_indices to get upper triangle without diagonal to avoid duplicates
    i, j = np.triu_indices(n_classes, k=1)
    off_diagonal_sims = similarity_matrix[i, j]
    
    min_sim = float(np.min(off_diagonal_sims))
    max_sim = float(np.max(off_diagonal_sims))
    mean_sim = float(np.mean(off_diagonal_sims))
    median_sim = float(np.median(off_diagonal_sims))
    
    print(f"Total pairs: {len(off_diagonal_sims)}")
    print(f"Minimum similarity: {min_sim:.4f}")
    print(f"Maximum similarity: {max_sim:.4f}")
    print(f"Mean similarity: {mean_sim:.4f}")
    print(f"Median similarity: {median_sim:.4f}")
    
    # Check if any is exactly 0
    exactly_zero = np.sum(off_diagonal_sims == 0.0)
    print(f"Pairs with exactly 0.0 similarity: {exactly_zero}")
    
    # Check if any is very close to 0
    close_to_zero = np.sum(np.abs(off_diagonal_sims) < 1e-4)
    print(f"Pairs with similarity < 1e-4: {close_to_zero}")
    
    print("\nTheory check: Is theta ever empty/never 0 (perfectly orthogonal)?")
    if exactly_zero == 0:
        print("YES! No two classes are perfectly orthogonal. There is always SOME semantic leakage.")
    else:
        print("NO. There are completely orthogonal pairs.")

if __name__ == '__main__':
    main()
