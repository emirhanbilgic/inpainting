import sys
import os
import types
from unittest.mock import MagicMock

# --- MOCKING PROBLEMATIC MODULES (MUST BE AT THE VERY TOP) ---
def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = types.SimpleNamespace(name=name, origin="mock", submodule_search_locations=None, loader=None, has_location=False)
    m.__file__ = "mock"
    m.__path__ = []
    return m

problematic_prefixes = ["matplotlib", "sklearn", "transformers", "tokenizers"]
for prefix in problematic_prefixes:
    sys.modules[prefix] = create_mock_module(prefix)

problematic_specifics = [
    "matplotlib.pyplot", "matplotlib.colors", "matplotlib.cm",
    "sklearn.metrics", "sklearn.utils", "sklearn.base",
    "sklearn.utils.murmurhash", "sklearn.utils._cython_blas", "sklearn.utils._logistic_sigmoid",
    "transformers.generation", "transformers.generation.utils",
    "transformers.generation.candidate_generator", "transformers.utils",
    "tokenizers"
]
for name in problematic_specifics:
    sys.modules[name] = create_mock_module(name)

sys.modules["matplotlib"].pyplot = MagicMock()
sys.modules["matplotlib"].colors = MagicMock()
sys.modules["matplotlib"].cm = MagicMock()

# --- NOW IMPORT OTHER LIBRARIES ---
import json
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import open_clip

# --- PROJECT SETUP ---
project_root = "/Users/emirhan/Desktop/LeGrad-1"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import legrad after mocking
from legrad import LeWrapper, LePreprocess
from open_clip.timm_model import TimmModel

# --- DATA ---
IMAGE_PATH = "/Users/emirhan/Desktop/LeGrad-1/scripts/data/bird.png"
OUTPUT_DIR = os.path.join(project_root, "outputs", "bird_parts_all_methods")

CONCEPTS = [
    "crown", "nape", "back", "feathers", "wing", "rump",
    "tail feathers", "eye", "beak", "throat", "breast",
    "belly", "foot", "claws"
]

METHODS = [
    "5 Ascending"
]

VIS_CONCEPTS = [
    "tail feathers", "eye", "beak", "breast", "belly", "foot", "claws"
]

# --- UTILS ---
def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    device = x_1x.device
    dtype = x_1x.dtype
    x = x_1x.clone().cpu().float()
    D_cpu = D.cpu().float()
    
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        c = (r @ D_cpu.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D_cpu[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0])
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
        x_hat = (s.t() @ D_S)
        r = (x - x_hat)
        if float(torch.norm(r) <= tol):
            break
    r = r.to(device=device, dtype=dtype)
    if torch.norm(r) <= tol:
        return F.normalize(x_1x, dim=-1)
    return F.normalize(r, dim=-1)

def get_text_embedding(model, tokenizer, prompt, device):
    tok = tokenizer([prompt]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb

def apply_colormap_jet(heatmap_np):
    val = np.clip(heatmap_np * 4, 0, 4)
    rgb = np.zeros(heatmap_np.shape + (3,), dtype=np.float32)
    mask = (val >= 0) & (val <= 1)
    rgb[mask, 2] = 1.0; rgb[mask, 1] = val[mask]; rgb[mask, 0] = 0.0
    mask = (val > 1) & (val <= 2)
    rgb[mask, 2] = 2.0 - val[mask]; rgb[mask, 1] = 1.0; rgb[mask, 0] = 0.0
    mask = (val > 2) & (val <= 3)
    rgb[mask, 2] = 0.0; rgb[mask, 1] = 1.0; rgb[mask, 0] = val[mask] - 2.0
    mask = (val > 3) & (val <= 4)
    rgb[mask, 2] = 0.0; rgb[mask, 1] = 4.0 - val[mask]; rgb[mask, 0] = 1.0
    return (rgb * 255).astype(np.uint8)

def get_heatmap_vis(image_pil, heatmap_tensor):
    w, h = image_pil.size
    heatmap = heatmap_tensor.numpy()
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8, mode='L')
    heatmap_pil = heatmap_pil.resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil).astype(float) / 255.0
    colormap_img_np = apply_colormap_jet(heatmap_resized)
    colormap_img = Image.fromarray(colormap_img_np, mode='RGB')
    blended = Image.blend(image_pil, colormap_img, alpha=0.5)
    return blended

# --- METHOD FUNCTIONS ---

def compute_legrad_heatmap(model, image, text_emb):
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb)
    logits = logits[0, 0].clamp(0, 1).detach().cpu()
    return logits

# --- RENDERING COMBINED GRID ---

def create_combined_grid_image(vis_dict_all_methods, original_image, concepts, methods):
    cell_w, cell_h = 224, 224
    padding_x = 20
    padding_y = 20
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 24)
        header_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 28)
    except:
        font = ImageFont.load_default()
        header_font = font
        
    num_rows = len(concepts)
    num_cols = 2 + len(methods)
    
    label_w = 140
    header_h = 50 # Fixed smaller header height
    
    grid_w = label_w + (num_cols - 1) * cell_w + (num_cols - 2) * padding_x
    grid_h = padding_y + header_h + padding_y + num_rows * cell_h + (num_rows - 1) * padding_y + padding_y
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Draw headers
    header_y = padding_y + 10 # Place text near the top
    draw.text((10, header_y), "Concept", fill="black", font=font)
    
    # LeGrad header (Centered over cell_w)
    legrad_text = "LeGrad"
    legrad_bbox = draw.textbbox((0, 0), legrad_text, font=header_font)
    legrad_w = legrad_bbox[2] - legrad_bbox[0]
    legrad_x = label_w + (cell_w - legrad_w) // 2
    draw.text((legrad_x, header_y), legrad_text, fill="black", font=header_font)
    
    for m_idx, method in enumerate(methods):
        x_base = label_w + (m_idx + 1) * (cell_w + padding_x)
        # Main method header (Centered over cell_w)
        method_name = "LeGrad-OSP"
        method_bbox = draw.textbbox((0, 0), method_name, font=header_font)
        method_w = method_bbox[2] - method_bbox[0]
        method_x = x_base + (cell_w - method_w) // 2
        draw.text((method_x, header_y), method_name, fill="black", font=header_font)
    
    # Draw content
    content_start_y = padding_y + header_h + padding_y
    for r_idx, concept in enumerate(concepts):
        y_offset = content_start_y + r_idx * (cell_h + padding_y)
        draw.text((10, y_offset + cell_h//2 - 12), concept, fill="black", font=font)
        
        # Draw LeGrad (Before) 
        vis_before, _ = vis_dict_all_methods[methods[0]][concept]
        vis_before_resized = vis_before.resize((cell_w, cell_h), Image.BILINEAR)
        grid_img.paste(vis_before_resized, (label_w, y_offset))
        
        # Draw After images for each method
        for m_idx, method in enumerate(methods):
            x_base = label_w + (m_idx + 1) * (cell_w + padding_x)
            
            _, vis_after = vis_dict_all_methods[method][concept]
            vis_after_resized = vis_after.resize((cell_w, cell_h), Image.BILINEAR)
            
            grid_img.paste(vis_after_resized, (x_base, y_offset))
            
            _, vis_after = vis_dict_all_methods[method][concept]
            vis_after_resized = vis_after.resize((cell_w, cell_h), Image.BILINEAR)
            
            grid_img.paste(vis_after_resized, (x_base, y_offset))
            
    return grid_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=224)
    
    print(f"Loading image {IMAGE_PATH}")
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    img_t = preprocess(original_image).unsqueeze(0).to(device)
    
    # Pre-compute embeddings for all concepts dictionary
    print("Computing embeddings...")
    embeddings_dict = {}
    for concept in CONCEPTS:
        prompt = f"a photo of a {concept} of a bird."
        embeddings_dict[concept] = get_text_embedding(model, tokenizer, prompt, device)
        
    print("Processing concepts...")
    vis_dict_all_methods = {m: {} for m in METHODS}
    
    for method in METHODS:
        print(f"\n--- Method: {method} ---")
        parts = method.split(" ")
        num_concepts = int(parts[0])
        sort_order = parts[1].lower()

        for target_concept in tqdm(VIS_CONCEPTS):
            target_emb = embeddings_dict[target_concept]
            
            # --- 1. Normal (Before Ortho) ---
            heatmap_before = compute_legrad_heatmap(model, img_t, target_emb)
            vis_before = get_heatmap_vis(original_image, heatmap_before)
            
            # --- 2. OMP (After Ortho) ---
            other_concepts = [c for c in CONCEPTS if c != target_concept]
            
            # Compute cosine similarities
            sims = []
            for c in other_concepts:
                c_emb = embeddings_dict[c]
                sim = F.cosine_similarity(target_emb, c_emb).item()
                sims.append((sim, c))
            
            # Sort based on sort_order
            sims.sort(key=lambda x: x[0], reverse=(sort_order == "descending"))
            
            # Select top num_concepts
            selected_other_concepts = [x[1] for x in sims[:num_concepts]]
            
            D_list = [embeddings_dict[c] for c in selected_other_concepts]
            D = torch.cat(D_list, dim=0)
            
            target_emb_omp = omp_sparse_residual(target_emb, D, max_atoms=num_concepts)
            
            heatmap_after = compute_legrad_heatmap(model, img_t, target_emb_omp)
            vis_after = get_heatmap_vis(original_image, heatmap_after)
            
            vis_dict_all_methods[method][target_concept] = (vis_before, vis_after)
            
    print("\nCreating combined grid image across all methods...")
    grid_img = create_combined_grid_image(vis_dict_all_methods, original_image, VIS_CONCEPTS, METHODS)
    grid_path_png = os.path.join(OUTPUT_DIR, "combined_grid_all_methods.png")
    grid_path_pdf = os.path.join(OUTPUT_DIR, "combined_grid_all_methods.pdf")
    grid_img.save(grid_path_png)
    grid_img.save(grid_path_pdf, "PDF", resolution=100.0)
    print(f"Finished. Visualizations saved to {grid_path_png} and {grid_path_pdf}")

if __name__ == "__main__":
    main()
