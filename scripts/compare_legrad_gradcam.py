import os
import argparse

import requests
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt

from legrad import LeWrapper, LePreprocess


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        img = Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
    else:
        if not os.path.isfile(path_or_url):
            raise FileNotFoundError(f"Image file not found: {path_or_url}")
        img = Image.open(path_or_url).convert("RGB")
    return img


def main():
    parser = argparse.ArgumentParser(description="Compare LeGrad and Grad-CAM heatmaps for CLIP.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path or URL to an input image.")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="Text prompts, e.g. --prompts 'a photo of a dog' 'a photo of a cat'")
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--output", type=str, default="outputs/compare_legrad_gradcam.png")
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

    # ------- Prepare inputs -------
    raw_image = load_image(args.image)
    img_tensor = preprocess(raw_image).unsqueeze(0).to(device)  # [1, 3, H, W]

    text_tokens = tokenizer(args.prompts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)  # [P, d]

    # ------- LeGrad heatmaps -------
    with torch.no_grad():
        legrad_maps = model.compute_legrad_clip(text_embedding=text_emb, image=img_tensor)  # [1, P, H, W]

    # ------- Grad-CAM heatmaps -------
    # Uses last transformer block by default (target_layer=-1)
    gradcam_maps = model.compute_gradcam_clip(text_embedding=text_emb, image=img_tensor, target_layer=-1)

    # ------- Visual comparison -------
    # Convert to CPU numpy
    legrad_np = legrad_maps.squeeze(0).detach().cpu().numpy()    # [P, H, W]
    gradcam_np = gradcam_maps.squeeze(0).detach().cpu().numpy()  # [P, H, W]

    P = legrad_np.shape[0]
    fig, axes = plt.subplots(nrows=P, ncols=3, figsize=(9, 3 * P))

    if P == 1:
        axes = [axes]

    for i in range(P):
        # Original image
        axes[i][0].imshow(raw_image.resize((args.image_size, args.image_size)))
        axes[i][0].set_title(f"Image\nPrompt: {args.prompts[i]}")
        axes[i][0].axis("off")

        # LeGrad heatmap
        axes[i][1].imshow(raw_image.resize((args.image_size, args.image_size)))
        axes[i][1].imshow(legrad_np[i], cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        axes[i][1].set_title("LeGrad")
        axes[i][1].axis("off")

        # Grad-CAM heatmap
        axes[i][2].imshow(raw_image.resize((args.image_size, args.image_size)))
        axes[i][2].imshow(gradcam_np[i], cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        axes[i][2].set_title("Grad-CAM")
        axes[i][2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()


