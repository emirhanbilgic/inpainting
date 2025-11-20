#!/usr/bin/env python
"""
Zero-shot CUB-200-2011 evaluation for CLIP with and without sparse text encodings.

This script:
- Loads CUB-200-2011 dataset (train/test split).
- Builds prompts for all 200 CUB classes (e.g. "a photo of a Black footed Albatross.")
- Forms a dictionary D of all class text embeddings and/or WordNet neighbors.
- Computes sparse residual encodings using OMP.
- Evaluates zero-shot top-1 and top-5 accuracy.

Usage:
    python scripts/cub_zero_shot_sparse.py --cub_root /path/to/CUB_200_2011 --split test
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import open_clip


# -----------------------------------------------------------------------------
# CUB Dataset
# -----------------------------------------------------------------------------

class CUBDataset(Dataset):
    def __init__(self, root: str, split: str = 'test', transform=None):
        """
        root: Path to CUB_200_2011 directory containing 'images/', 'images.txt', etc.
        split: 'train' or 'test'
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Metadata files
        images_txt = os.path.join(root, 'images.txt')
        train_test_split_txt = os.path.join(root, 'train_test_split.txt')
        class_labels_txt = os.path.join(root, 'image_class_labels.txt')
        
        if not os.path.exists(images_txt):
            raise RuntimeError(f"CUB metadata not found at {root}. Expected images.txt etc.")

        # 1. Load image paths: ID -> Path
        self.img_id_to_path = {}
        with open(images_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id, path = parts[0], parts[1]
                    self.img_id_to_path[img_id] = path

        # 2. Load split: ID -> is_train (1=train, 0=test)
        self.img_id_to_is_train = {}
        with open(train_test_split_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id, is_train = parts[0], int(parts[1])
                    self.img_id_to_is_train[img_id] = is_train

        # 3. Load labels: ID -> Class ID (1..200)
        self.img_id_to_class = {}
        with open(class_labels_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_id, class_id = parts[0], int(parts[1])
                    self.img_id_to_class[img_id] = class_id

        # 4. Filter images based on split
        target_split = 1 if split == 'train' else 0
        self.samples = []
        
        for img_id, is_train in self.img_id_to_is_train.items():
            if is_train == target_split:
                if img_id in self.img_id_to_path and img_id in self.img_id_to_class:
                    path = os.path.join(root, 'images', self.img_id_to_path[img_id])
                    # Convert 1-based class ID to 0-based index
                    label = self.img_id_to_class[img_id] - 1
                    self.samples.append((path, label))
        
        print(f"[CUB] Loaded {len(self.samples)} images for split '{split}' from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_cub_classes(root: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load CUB classes from classes.txt.
    Returns:
        classes: List of raw class names ordered by ID (0-based index corresponding to file ID-1).
        id_to_label: Map from raw class name to cleaned human readable label.
    """
    classes_txt = os.path.join(root, 'classes.txt')
    raw_classes = {} # id -> raw_name
    
    with open(classes_txt, 'r') as f:
        for line in f:
            # Format: 1 001.Black_footed_Albatross
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id = int(parts[0])
                raw_name = parts[1]
                raw_classes[class_id] = raw_name
    
    # Ensure we have a list 0..199
    sorted_ids = sorted(raw_classes.keys())
    assert len(sorted_ids) == 200, f"Expected 200 classes, found {len(sorted_ids)}"
    
    classes_list = [raw_classes[i] for i in sorted_ids]
    
    # Build clean labels mapping
    raw_to_clean = {}
    for raw in classes_list:
        # "001.Black_footed_Albatross" -> "Black footed Albatross"
        clean = raw.split('.', 1)[-1].replace('_', ' ')
        raw_to_clean[raw] = clean
        
    return classes_list, raw_to_clean

# -----------------------------------------------------------------------------
# WordNet / Sparse Helpers (Copied/Adapted from imagenet_zero_shot_sparse.py)
# -----------------------------------------------------------------------------

def wordnet_neighbors_configured(
    keyword: str,
    use_synonyms: bool,
    use_hypernyms: bool,
    use_hyponyms: bool,
    use_siblings: bool,
    use_fallback: bool = True,
    limit_per_relation: int = 8
) -> List[str]:
    """
    Configurable WordNet neighbors.
    """
    try:
        import nltk
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn
    except Exception as e:
        print(f"[WordNet] Warning: Failed to load NLTK/WordNet: {e}")
        return []

    def get_neighbors_for_term(term: str) -> List[str]:
        out = []
        seen = set()
        key_low = term.lower()
        # Only look for NOUNs
        synsets = wn.synsets(term, pos=wn.NOUN)
        
        for s in synsets[:limit_per_relation]:
            if use_synonyms:
                for l in s.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name); seen.add(name)
            if use_hypernyms:
                for h in s.hypernyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name); seen.add(name)
            if use_hyponyms:
                for h in s.hyponyms()[:limit_per_relation]:
                    for l in h.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name); seen.add(name)
            if use_siblings:
                for h in s.hypernyms()[:limit_per_relation]:
                    for sib in h.hyponyms()[:limit_per_relation]:
                        for l in sib.lemmas()[:limit_per_relation]:
                            name = l.name().replace('_', ' ').lower()
                            if name != key_low and name not in seen:
                                out.append(name); seen.add(name)
        return out

    # 1. Try full keyword
    results = get_neighbors_for_term(keyword)
    if results:
        return results[: max(1, limit_per_relation * 3)]

    if not use_fallback:
        return []

    # 2. Fallback: iterate sub-phrases from right to left (suffixes)
    # e.g. "great white shark" -> "white shark" -> "shark"
    words = keyword.split()
    if len(words) > 1:
        # Try progressively shorter suffixes
        for i in range(1, len(words)):
            sub_term = " ".join(words[i:])
            results = get_neighbors_for_term(sub_term)
            if results:
                # print(f"[WordNet] Fallback: '{keyword}' -> '{sub_term}' found {len(results)} neighbors.")
                return results[: max(1, limit_per_relation * 3)]
    
    return []


def inject_context(prompt: str, key: str, neighbor: str) -> str:
    # Case insensitive replace of the LAST occurrence of key to preserve prompt structure
    lprompt = prompt.lower()
    lkey = key.lower()
    idx = lprompt.rfind(lkey)
    if idx == -1: 
        return neighbor # fallback if key not found
    return prompt[:idx] + neighbor + prompt[idx+len(key):]


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    if D is None or D.numel() == 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(max(1, min(max_atoms, K)))
    selected: List[int] = []
    r = x.clone()
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t, 1]
        x_hat = (s.t() @ D_S).to(x.dtype)        # [1, d]
        r = (x - x_hat)
        if float(torch.norm(r)) <= tol:
            break
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


@torch.no_grad()
def compute_text_embeddings(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    batch_size = 256
    all_embs: List[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tokens = tokenizer(chunk).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)


def precompute_sparse_text_embeddings(
    model,
    tokenizer,
    class_prompts: List[str],
    text_embs: torch.Tensor,
    raw_classes: List[str],
    raw_to_clean: Dict[str, str],
    atoms: int,
    max_cos_sim: float | None,
    wn_config: Dict[str, bool],
    dict_include_prompts: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute sparse residual encodings.
    """
    C, d = text_embs.shape
    sparse_embs = torch.empty_like(text_embs)
    print(f"[sparse] Precomputing sparse encodings for {C} classes with {atoms} atoms...")
    print(f"[sparse] Config: WordNet={wn_config}, include_prompts={dict_include_prompts}, max_sim={max_cos_sim}")

    use_wn = any(wn_config.values())

    for c in tqdm(range(C), desc="OMP (classes)"):
        x = text_embs[c : c + 1]  # [1, d]
        raw_name = raw_classes[c]
        clean_label = raw_to_clean.get(raw_name, raw_name)
        
        parts = []
        d_labels = []

        # 1. Other class prompts
        if dict_include_prompts and C > 1:
            # Exclude self (index c)
            if c > 0:
                parts.append(text_embs[:c])
                d_labels.extend([f"class_{i}" for i in range(c)])
            if c + 1 < C:
                parts.append(text_embs[c + 1 :])
                d_labels.extend([f"class_{i}" for i in range(c + 1, C)])
        
        # 2. WordNet neighbors
        if use_wn:
            raw_neighbors = wordnet_neighbors_configured(
                clean_label,
                use_synonyms=wn_config.get('synonyms', False),
                use_hypernyms=wn_config.get('hypernyms', False),
                use_hyponyms=wn_config.get('hyponyms', False),
                use_siblings=wn_config.get('siblings', False),
                use_fallback=wn_config.get('fallback_search', True),
                limit_per_relation=8
            )
            if raw_neighbors:
                prompt = class_prompts[c]
                neighbor_prompts = [inject_context(prompt, clean_label, w) for w in raw_neighbors]
                
                if neighbor_prompts:
                    with torch.no_grad():
                        n_tok = tokenizer(neighbor_prompts).to(device)
                        n_emb = model.encode_text(n_tok)
                        n_emb = F.normalize(n_emb, dim=-1)
                        parts.append(n_emb)
                        d_labels.extend(neighbor_prompts)
        
        D = torch.cat(parts, dim=0) if len(parts) > 0 else text_embs.new_zeros((0, d))
        
        verbose = (c < 3)
        if verbose:
            print(f"\n[class {c}] label='{clean_label}' prompt='{class_prompts[c]}'")
            print(f"  Dictionary size raw: {D.shape[0]}")

        if D.numel() > 0:
            D = F.normalize(D, dim=-1)
            
            # Filter by max similarity
            sim = (D @ x.t()).squeeze(-1).abs()
            
            # Debug print for small dicts
            if verbose and D.shape[0] < 100:
                 sorted_idxs = torch.argsort(sim, descending=True)
                 print("  Dictionary atoms (sim to target):")
                 for idx in sorted_idxs[:10]:
                     idx_val = idx.item()
                     w_lbl = d_labels[idx_val] if idx_val < len(d_labels) else "unknown"
                     print(f"    - '{w_lbl}': {sim[idx_val]:.4f}")

            if max_cos_sim is not None and float(max_cos_sim) < 1.0:
                keep = sim < float(max_cos_sim)
                if verbose:
                    dropped = (~keep).sum().item()
                    if dropped > 0:
                        print(f"  Dropped {dropped} atoms with sim >= {max_cos_sim}")
                D = D[keep]
        
        if D.numel() == 0:
            if verbose:
                print("  Dictionary empty after filtering. Using original.")
            sparse_embs[c] = x
            continue
            
        sparse = omp_sparse_residual(x, D, max_atoms=atoms)
        sparse_embs[c] = sparse.to(device)
        
    return sparse_embs


@torch.no_grad()
def evaluate_dataset(
    model,
    dataloader: DataLoader,
    text_embs: torch.Tensor,
    sparse_text_embs: torch.Tensor,
    atoms: int,
    device: torch.device,
) -> None:
    model.eval()
    text_embs = text_embs.to(device)
    sparse_text_embs = sparse_text_embs.to(device)

    correct_top1_std = 0
    correct_top5_std = 0
    correct_top1_sparse = 0
    correct_top5_sparse = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Encode images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Standard logits
        logits_std = image_features @ text_embs.t()
        # Sparse-encoded logits
        logits_sparse = image_features @ sparse_text_embs.t()

        if hasattr(model, "logit_scale"):
            logit_scale = model.logit_scale.exp()
            logits_std = logit_scale * logits_std
            logits_sparse = logit_scale * logits_sparse

        # Top-1 / Top-5
        _, pred_std_top5 = logits_std.topk(5, dim=-1)
        correct_std_top1_batch = (pred_std_top5[:, 0] == targets).sum().item()
        correct_std_top5_batch = (pred_std_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        _, pred_sparse_top5 = logits_sparse.topk(5, dim=-1)
        correct_sparse_top1_batch = (pred_sparse_top5[:, 0] == targets).sum().item()
        correct_sparse_top5_batch = (pred_sparse_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)
        correct_top1_std += correct_std_top1_batch
        correct_top5_std += correct_std_top5_batch
        correct_top1_sparse += correct_sparse_top1_batch
        correct_top5_sparse += correct_sparse_top5_batch

    top1_std = 100.0 * correct_top1_std / total
    top5_std = 100.0 * correct_top5_std / total
    top1_sparse = 100.0 * correct_top1_sparse / total
    top5_sparse = 100.0 * correct_top5_sparse / total

    print("\n=== Zero-shot Results ===")
    print(f"Total images: {total}")
    print(f"[Standard CLIP]  Top-1: {top1_std:.2f}%   Top-5: {top5_std:.2f}%")
    print(f"[Sparse (atoms={atoms})] Top-1: {top1_sparse:.2f}%   Top-5: {top5_sparse:.2f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP on CUB-200-2011 with sparse encodings.")
    parser.add_argument("--cub_root", type=str, required=True, help="Path to CUB dataset root (containing images.txt etc).")
    parser.add_argument("--split", type=str, default="test", help="test or train split")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}.", help="Prompt template")
    parser.add_argument("--model_name", type=str, default="ViT-B-16", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="OpenCLIP pretrained tag")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--max_dict_cos_sim", type=float, default=0.9)
    
    # Dictionary construction flags
    parser.add_argument("--dict_include_prompts", type=int, default=1, help="Include other class prompts in dictionary (0/1)")
    
    # WordNet config
    parser.add_argument('--wn_use_synonyms', type=int, default=0)
    parser.add_argument('--wn_use_hypernyms', type=int, default=0)
    parser.add_argument('--wn_use_hyponyms', type=int, default=0)
    parser.add_argument('--wn_use_siblings', type=int, default=1)
    parser.add_argument("--wn_fallback_search", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cuda")
    print(f"[device] Using device: {device}")
    
    # 1. Load Model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # 2. Load Data
    print(f"[data] Loading CUB data from {args.cub_root}...")
    dataset = CUBDataset(root=args.cub_root, split=args.split, transform=preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 3. Build Prompts
    raw_classes, raw_to_clean = load_cub_classes(args.cub_root)
    prompts = []
    for raw in raw_classes:
        label = raw_to_clean[raw]
        prompts.append(args.prompt_template.format(label))
        
    print(f"[prompts] Built {len(prompts)} prompts. Example:")
    for i in range(min(3, len(prompts))):
        print(f"  Class {i} ({raw_classes[i]}): {prompts[i]}")

    # 4. Compute Standard Embeddings
    text_embs = compute_text_embeddings(model, tokenizer, prompts, device)

    # 5. Compute Sparse Embeddings
    wn_config = {
        'synonyms': bool(args.wn_use_synonyms),
        'hypernyms': bool(args.wn_use_hypernyms),
        'hyponyms': bool(args.wn_use_hyponyms),
        'siblings': bool(args.wn_use_siblings),
        'fallback_search': bool(args.wn_fallback_search),
    }
    
    sparse_text_embs = precompute_sparse_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        class_prompts=prompts,
        text_embs=text_embs,
        raw_classes=raw_classes,
        raw_to_clean=raw_to_clean,
        atoms=args.atoms,
        max_cos_sim=args.max_dict_cos_sim,
        wn_config=wn_config,
        dict_include_prompts=bool(args.dict_include_prompts),
        device=device
    )
    
    # 6. Evaluate
    evaluate_dataset(
        model=model,
        dataloader=dataloader,
        text_embs=text_embs,
        sparse_text_embs=sparse_text_embs,
        atoms=args.atoms,
        device=device
    )

if __name__ == "__main__":
    main()

