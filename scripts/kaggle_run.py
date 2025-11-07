#!/usr/bin/env python

import argparse
import os
import re
import sys
from typing import List

import requests
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip

from legrad import LeWrapper, LePreprocess


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return name or "prompt"


def load_image(image_url: str = None, image_path: str = None) -> Image.Image:
    if image_url:
        resp = requests.get(image_url, stream=True)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    if image_path:
        return Image.open(image_path).convert("RGB")
    raise ValueError("Provide either --image_url or --image_path")


def parse_prompts(prompt: str, prompts: List[str]) -> List[str]:
    if prompts and len(prompts) > 0:
        return prompts
    if prompt:
        return [p.strip() for p in prompt.split(",") if p.strip()]
    raise ValueError("Provide --prompt or --prompts")


def main():
    parser = argparse.ArgumentParser(description="Run LeGrad on Kaggle and save heatmaps.")
    parser.add_argument("--image_url", type=str, default=None, help="URL of the image to process")
    parser.add_argument("--image_path", type=str, default=None, help="Local image path to process")
    parser.add_argument("--prompt", type=str, default="cat", help="Single or comma-separated prompts")
    parser.add_argument("--prompts", type=str, nargs="*", default=None, help="Multiple prompts list")
    parser.add_argument("--model_name", type=str, default="ViT-B-16", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="Pretrained tag")
    parser.add_argument("--image_size", type=int, default=448, help="Square image size for preprocessing")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save heatmaps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()

    model = LeWrapper(model)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    # Inputs
    pil_image = load_image(image_url=args.image_url, image_path=args.image_path)
    image_t = preprocess(pil_image).unsqueeze(0).to(device)

    prompts = parse_prompts(args.prompt, args.prompts)
    tokenized = tokenizer([f"a photo of a {p}." for p in prompts]).to(device)
    text_emb = model.encode_text(tokenized, normalize=True)

    # Compute LeGrad
    logits = model.compute_legrad(image=image_t, text_embedding=text_emb)
    # logits shape is [1, num_prompts, H, W] for OpenCLIP path

    with torch.no_grad():
        # Iterate prompts and save heatmaps
        # Clamp and scale to [0,255]
        logits_np = logits.squeeze(0).cpu()  # [num_prompts, H, W]
        for idx, p in enumerate(prompts):
            heat = logits_np[idx]
            heat_01 = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat_255 = (heat_01 * 255.0).byte().numpy()

            # Colorize via cv2 if available, otherwise save grayscale
            try:
                import cv2
                color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
                # Resize original for overlay
                img_bgr = cv2.cvtColor(
                    (pil_image.resize((heat_255.shape[1], heat_255.shape[0]))).__array__(),
                    cv2.COLOR_RGB2BGR,
                )
                overlay = (0.4 * img_bgr + 0.6 * color).astype("uint8")
                out = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                out_image = Image.fromarray(out)
            except Exception:
                out_image = Image.fromarray(heat_255)

            fname = f"legrad_{sanitize_filename(p)}.png"
            out_path = os.path.join(args.output_dir, fname)
            out_image.save(out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


