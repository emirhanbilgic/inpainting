#!/usr/bin/env python
"""
OMP-based disentangling of the attribute "red" from the object "car"
using text embeddings only (no images).

We:
- Use "a photo of a red car" as the target embedding x
- Use ["a photo of a blue car", "a photo of a green car", "a photo of a black car"]
  as a dictionary that spans the "car" subspace
- Run Orthogonal Matching Pursuit (OMP) to obtain the residual r = x - P_D(x)
- Measure cosine similarities:
    Sim(x, car)          vs Sim(r, car)
    Sim(x, red)          vs Sim(r, red)

We repeat this for three text encoders:
- CLIP     (OpenCLIP ViT-B-16, laion2b_s34b_b88k)
- SigLIP   (OpenCLIP ViT-B-16-SigLIP, webli)
- T5       (Hugging Face "t5-base", mean-pooled last hidden state)

Run:
    python scripts/omp_red_car_disentangle.py
"""

import os
import sys

import torch
import torch.nn.functional as F

import open_clip


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual.

    Args:
        x_1x: [1, d] target vector (assumed L2-normalized)
        D:   [K, d] dictionary atoms (rows), L2-normalized
        max_atoms: maximum number of atoms to select
        tol: early stopping threshold on residual norm / correlation

    Returns:
        Residual r (L2-normalized): [1, d]
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


def get_openclip_text_encoder(model_name: str, pretrained: str, device: torch.device):
    """
    Load an OpenCLIP model and return (encode_fn, model_name_pretty).

    encode_fn(prompts: List[str]) -> Tensor [N, d], L2-normalized.
    """
    print(f"\n[OpenCLIP] Loading {model_name} ({pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model.eval()

    def encode(prompts):
        tok = tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tok)
        emb = F.normalize(emb, dim=-1)
        return emb

    pretty_name = f"{model_name} ({pretrained})"
    return encode, pretty_name


def get_t5_text_encoder(device: torch.device):
    """
    Build a T5 text encoder using Hugging Face Transformers.

    Requires:
        pip install transformers

    Returns:
        encode(prompts: List[str]) -> Tensor [N, d], L2-normalized
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required for T5 embeddings.\n"
            "Install it with:\n"
            "    pip install transformers\n"
        )

    model_name = "t5-base"
    print(f"\n[T5] Loading {model_name} from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(prompts):
        # T5 is an encoder-decoder; we only use the encoder output.
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            hidden = out.last_hidden_state  # [B, L, d]

        # Mean-pool over non-padding tokens
        mask = batch["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
        hidden = hidden * mask
        summed = hidden.sum(dim=1)  # [B, d]
        counts = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        emb = summed / counts
        emb = F.normalize(emb, dim=-1)
        return emb

    pretty_name = "T5-base (mean-pooled encoder)"
    return encode, pretty_name


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity for two 1D or [1, d] tensors (assumed already normalized).
    Returns a Python float.
    """
    if a.ndim == 2:
        a = a.squeeze(0)
    if b.ndim == 2:
        b = b.squeeze(0)
    return float(torch.sum(a * b).item())


def run_experiment(encode, encoder_name: str, device: torch.device, max_atoms: int = 3):
    """
    Run the red/car OMP disentangling experiment for a given text encoder.
    """
    print(f"\n{'=' * 80}")
    print(f"Encoder: {encoder_name}")
    print(f"{'=' * 80}")

    prompts = {
        "red_car": "a photo of a red car",
        "blue_car": "a photo of a blue car",
        "green_car": "a photo of a green car",
        "black_car": "a photo of a black car",
        "car": "a photo of a car",
        "red": "a photo of a red",
    }

    order = ["red_car", "blue_car", "green_car", "black_car", "car", "red"]
    texts = [prompts[k] for k in order]

    print("Prompts:")
    for k in order:
        print(f"  {k:9s}: {prompts[k]}")

    emb_all = encode(texts)  # [6, d]

    # Unpack embeddings (all L2-normalized)
    e_red_car, e_blue_car, e_green_car, e_black_car, e_car, e_red = emb_all

    # Dictionary of car variations for OMP
    D_car = torch.stack([e_blue_car, e_green_car, e_black_car], dim=0)  # [3, d]
    D_car = F.normalize(D_car, dim=-1)

    # Target: red car
    x = e_red_car.unsqueeze(0)  # [1, d]

    # OMP residual
    r = omp_sparse_residual(x, D_car, max_atoms=max_atoms)  # [1, d]

    # Cosine similarities (before / after)
    sim_x_car = cosine(e_red_car, e_car)
    sim_r_car = cosine(r, e_car)

    sim_x_red = cosine(e_red_car, e_red)
    sim_r_red = cosine(r, e_red)

    print("\nCosine similarities (all embeddings L2-normalized):")
    print(f"  Sim(red car, car)           = {sim_x_car: .4f}")
    print(f"  Sim(residual, car)          = {sim_r_car: .4f}  (from {sim_x_car: .4f})")
    print()
    print(f"  Sim(red car, red)           = {sim_x_red: .4f}")
    print(f"  Sim(residual, red)          = {sim_r_red: .4f}  (from {sim_x_red: .4f})")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CLIP (OpenCLIP ViT-B-16) ---
    clip_encode, clip_name = get_openclip_text_encoder(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device,
    )
    run_experiment(clip_encode, f"CLIP / OpenCLIP {clip_name}", device=device, max_atoms=3)

    # --- SigLIP (OpenCLIP ViT-B-16-SigLIP) ---
    siglip_encode, siglip_name = get_openclip_text_encoder(
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        device=device,
    )
    run_experiment(siglip_encode, f"SigLIP / OpenCLIP {siglip_name}", device=device, max_atoms=3)

    # --- T5 ---
    t5_encode, t5_name = get_t5_text_encoder(device=device)
    run_experiment(t5_encode, t5_name, device=device, max_atoms=3)

    print("\nDone. You should now see cosine similarities for CLIP, SigLIP, and T5.")


if __name__ == "__main__":
    main()



