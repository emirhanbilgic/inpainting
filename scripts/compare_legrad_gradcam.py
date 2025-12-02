import os
import sys
import argparse
import re

from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt

# Ensure we import the local legrad package from the repo, not any installed version
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from legrad import LeWrapper, LePreprocess


def sanitize(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def list_images(folder: str, limit: int, seed: int = 42):
    import random

    entries = []
    if not os.path.isdir(folder):
        return entries
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            ext = name.lower().rsplit(".", 1)[-1]
            if ext in {"jpg", "jpeg", "png", "bmp", "webp"}:
                entries.append(path)
    random.Random(seed).shuffle(entries)
    return entries[:limit]


def collect_dataset_paths(dataset_root: str, num_per_class: int, max_images: int = 0):
    paths = []
    cat_dir = os.path.join(dataset_root, "Cat")
    dog_dir = os.path.join(dataset_root, "Dog")
    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        paths += list_images(cat_dir, limit=num_per_class)
        paths += list_images(dog_dir, limit=num_per_class)
    else:
        flat_limit = max_images if max_images > 0 else max(1, num_per_class * 2)
        paths += list_images(dataset_root, limit=flat_limit)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Compare LeGrad and Grad-CAM heatmaps for CLIP over a dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help='Root folder with images (optionally containing "Cat" and "Dog" subfolders).',
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Text prompts, e.g. --prompts 'a photo of a dog.' 'a photo of a cat.'",
    )
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--num_per_class", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0, help="Cap total images processed; 0 means auto.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/compare_legrad_gradcam",
        help="Directory where per-image comparison figures will be saved.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- init CLIP model -------
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()

    # ------- Equip the model with LeGrad wrapper (now also providing Grad-CAM) -------
    model = LeWrapper(model)
    preprocess = LePreprocess(preprocess=preprocess, image_size=args.image_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image paths
    paths = collect_dataset_paths(args.dataset_root, num_per_class=args.num_per_class, max_images=args.max_images)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under {args.dataset_root}")

    # Prepare text embeddings once
    text_tokens = tokenizer(args.prompts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)  # [P, d]

    for pth in paths:
        try:
            raw_image = Image.open(pth).convert("RGB")
        except Exception:
            continue

        img_tensor = preprocess(raw_image).unsqueeze(0).to(device)  # [1, 3, H, W]

        # ------- LeGrad heatmaps -------
        # NOTE: do NOT wrap in torch.no_grad(); LeGrad needs gradients internally.
        legrad_maps = model.compute_legrad_clip(text_embedding=text_emb, image=img_tensor)  # [1, P, H, W]

        # ------- Grad-CAM heatmaps -------
        gradcam_maps = model.compute_gradcam_clip(
            text_embedding=text_emb, image=img_tensor, target_layer=-1
        )  # [1, P, H, W]

        # Optional: print simple stats to verify Grad-CAM is non-trivial
        try:
            g_min = float(gradcam_maps.min().item())
            g_max = float(gradcam_maps.max().item())
            print(f"[Grad-CAM] {os.path.basename(pth)} min={g_min:.4f}, max={g_max:.4f}")
        except Exception:
            pass

        # ------- Visual comparison -------
        legrad_np = legrad_maps.squeeze(0).detach().cpu().numpy()  # [P, H, W]
        gradcam_np = gradcam_maps.squeeze(0).detach().cpu().numpy()  # [P, H, W]

        P = legrad_np.shape[0]
        fig, axes = plt.subplots(nrows=P, ncols=3, figsize=(9, 3 * P))

        if P == 1:
            axes = [axes]

        resized = raw_image.resize((args.image_size, args.image_size))
        for i in range(P):
            # Original image
            axes[i][0].imshow(resized)
            axes[i][0].set_title(f"Image\nPrompt: {args.prompts[i]}")
            axes[i][0].axis("off")

            # LeGrad heatmap
            axes[i][1].imshow(resized)
            axes[i][1].imshow(legrad_np[i], cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
            axes[i][1].set_title("LeGrad")
            axes[i][1].axis("off")

            # Grad-CAM heatmap
            axes[i][2].imshow(resized)
            axes[i][2].imshow(gradcam_np[i], cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
            axes[i][2].set_title("Grad-CAM")
            axes[i][2].axis("off")

        plt.tight_layout()
        base = os.path.splitext(os.path.basename(pth))[0]
        out_name = f"{sanitize(base)}_compare_legrad_gradcam.png"
        out_path = os.path.join(args.output_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

