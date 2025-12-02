#!/usr/bin/env python

import argparse
import os
import re
from io import BytesIO
from typing import List

import requests
from PIL import Image

import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import open_clip

from legrad import LeWrapper


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return name or "prompt"


def load_image(image_url: str = None, image_path: str = None) -> Image.Image:
    if image_url:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if image_path:
        return Image.open(image_path).convert("RGB")
    raise ValueError("Provide either --image_url or --image_path")


def parse_prompts(prompt: str, prompts: List[str]) -> List[str]:
    if prompts and len(prompts) > 0:
        return prompts
    if prompt:
        return [p.strip() for p in prompt.split(",") if p.strip()]
    raise ValueError("Provide --prompt or --prompts")


def pil_to_tensor_no_numpy(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    byte_data = img.tobytes()
    t = torch.tensor(list(byte_data), dtype=torch.uint8)
    t = t.view(h, w, 3).permute(2, 0, 1)  # C,H,W
    return t


def safe_preprocess(img: Image.Image, image_size: int = 448) -> torch.Tensor:
    t = pil_to_tensor_no_numpy(img)  # uint8, CxHxW
    # Resize + center crop using tensor ops
    t = TF.resize(t, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    t = TF.center_crop(t, [image_size, image_size])
    # Normalize like CLIP
    t = t.float() / 255.0
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t


def tensor_rgb_to_pil_bytes(t: torch.ByteTensor) -> bytes:
    # t: uint8, [3,H,W]
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    t = t.clamp(0, 255)
    t_hw3 = t.permute(1, 2, 0).contiguous().view(-1)  # [H*W*3]
    return bytes(t_hw3.tolist())


def save_overlay_pil_no_numpy(base_img: Image.Image, heat_01: torch.Tensor, out_path: str, alpha: float = 0.6):
    # base_img: PIL RGB
    # heat_01: torch.float [H, W] in [0,1]
    H, W = heat_01.shape
    # Resize original to heatmap size (via PIL to avoid numpy)
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    base_t = pil_to_tensor_no_numpy(base_resized).to(torch.float32)  # [3,H,W]

    # Create red mask from heat
    red = (heat_01 * 255.0).clamp(0, 255).to(torch.uint8)
    red3 = torch.stack([red, torch.zeros_like(red), torch.zeros_like(red)], dim=0)  # [3,H,W]

    # Blend: (1-alpha)*base + alpha*red
    blended = ((1.0 - alpha) * base_t + alpha * red3.to(torch.float32)).clamp(0, 255).to(torch.uint8)

    raw = tensor_rgb_to_pil_bytes(blended)
    out_img = Image.frombytes("RGB", (W, H), raw)
    out_img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare GradCAM (ViT) with LeGrad on the same image and prompts.")
    parser.add_argument("--image_url", type=str, default=None, help="URL of the image to process")
    parser.add_argument("--image_path", type=str, default=None, help="Local image path to process")
    parser.add_argument("--prompt", type=str, default="cat", help="Single or comma-separated prompts")
    parser.add_argument("--prompts", type=str, nargs="*", default=None, help="Multiple prompts list")
    parser.add_argument("--model_name", type=str, default="ViT-B-16", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="Pretrained tag")
    parser.add_argument("--image_size", type=int, default=448, help="Square image size for preprocessing")
    parser.add_argument("--target_layer", type=int, default=8, help="Transformer block index for GradCAM (0-based)")
    parser.add_argument("--output_dir", type=str, default="outputs_compare", help="Directory to save heatmaps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (skip OpenCLIP PIL->tensor to avoid numpy path)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()

    # Equip with LeGrad / hooks
    model = LeWrapper(model)

    # Inputs
    pil_image = load_image(image_url=args.image_url, image_path=args.image_path)
    image_t = safe_preprocess(pil_image, image_size=args.image_size).unsqueeze(0).to(device)

    prompts = parse_prompts(args.prompt, args.prompts)
    tokenized = tokenizer([f"a photo of a {p}." for p in prompts]).to(device)
    text_emb = model.encode_text(tokenized, normalize=True)

    # Compute LeGrad and GradCAM heatmaps
    with torch.no_grad():
        logits_legrad = model.compute_legrad(image=image_t, text_embedding=text_emb)  # [1, P, H, W]

    # GradCAM needs gradients, so we can't wrap this in no_grad
    logits_gradcam = model.compute_gradcam_vit(text_embedding=text_emb, image=image_t, target_layer=args.target_layer)

    # Save overlays for each prompt
    legrad_maps = logits_legrad.squeeze(0).cpu()      # [P, H, W]
    gradcam_maps = logits_gradcam.squeeze(0).cpu()    # [P, H, W]

    for idx, p in enumerate(prompts):
        heat_legrad = legrad_maps[idx]
        heat_gradcam = gradcam_maps[idx]

        # Normalize each map to [0,1] independently
        for method_name, heat in [("legrad", heat_legrad), ("gradcam", heat_gradcam)]:
            hmin = float(heat.min())
            hmax = float(heat.max())
            heat_01 = (heat - hmin) / (hmax - hmin + 1e-6)

            fname = f"{method_name}_{sanitize_filename(p)}.png"
            out_path = os.path.join(args.output_dir, fname)
            save_overlay_pil_no_numpy(pil_image, heat_01, out_path, alpha=0.6)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


