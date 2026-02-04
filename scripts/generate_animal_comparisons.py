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
    # Add some base attributes to avoid attribute errors during import
    return m

# Aggressively mock everything that might be imported
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

# Set up MagicMocks for the mocked modules
sys.modules["matplotlib"].pyplot = MagicMock()
sys.modules["matplotlib"].colors = MagicMock()
sys.modules["matplotlib"].cm = MagicMock()

# --- NOW IMPORT OTHER LIBRARIES ---
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm
import open_clip

# --- PROJECT SETUP ---
project_root = "/Users/emirhan/Desktop/LeGrad-1"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import legrad after mocking
from legrad import LeWrapper, LePreprocess

# --- DATA ---
ITEMS = [
    ("2007_000904.jpg", ["cow", "horse"]),
    ("2007_001594.jpg", ["dog", "sheep"]),
    ("2007_001763.jpg", ["cat", "dog"]),
    ("2007_001825.jpg", ["cat", "dog"]),
    ("2007_002268.jpg", ["cat", "dog"]),
    ("2007_002597.jpg", ["cat", "dog"]),
    ("2007_004537.jpg", ["cow", "horse"]),
    ("2007_005114.jpg", ["cow", "horse"]),
    ("2007_006944.jpg", ["horse", "sheep"]),
    ("2007_007417.jpg", ["cat", "dog"]),
    ("2007_009331.jpg", ["dog", "horse"]),
    ("2008_002536.jpg", ["dog", "sheep"]),
    ("2009_001885.jpg", ["dog", "horse"]),
    ("2010_002763.jpg", ["cat", "cow"]),
    ("2010_003670.jpg", ["dog", "sheep"]),
    ("2010_004760.jpg", ["cat", "dog"]),
    ("2010_005796.jpg", ["cat", "dog"]),
    ("2011_000219.jpg", ["bird", "cow"]),
    ("2011_000548.jpg", ["bird", "cow"]),
    ("2011_000834.jpg", ["bird", "sheep"]),
    ("2011_002464.jpg", ["bird", "sheep"]),
]

CANDIDATE_ANIMALS = ["bird", "cat", "cow", "dog", "horse", "sheep"]

DATASET_ROOT = "/Users/emirhan/Desktop/pascal-voc-2012-DatasetNinja"
OUTPUT_DIR = os.path.join(project_root, "HCI_animals_restructured")

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

def compute_heatmap(model, image, text_emb):
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb)
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits

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
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_u8, mode='L')
    heatmap_pil = heatmap_pil.resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil).astype(float) / 255.0
    colormap_img_np = apply_colormap_jet(heatmap_resized)
    colormap_img = Image.fromarray(colormap_img_np, mode='RGB')
    blended = Image.blend(image_pil, colormap_img, alpha=0.5)
    return blended

def find_image(img_name):
    for root_dir in ["trainval/img", "train/img", "val/img"]:
        path = os.path.join(DATASET_ROOT, root_dir, img_name)
        if os.path.exists(path):
            return path
    return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading LeGrad model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=224)
    
    for img_name, real_classes in tqdm(ITEMS):
        img_id = os.path.splitext(img_name)[0]
        img_path = find_image(img_name)
        
        if not img_path:
            continue
            
        try:
            original_image = Image.open(img_path).convert("RGB")
            img_t = preprocess(original_image).unsqueeze(0).to(device)
            
            # Select a fake animal
            fake_candidates = [a for a in CANDIDATE_ANIMALS if a not in real_classes]
            fake_animal = random.choice(fake_candidates)
            
            # Image folder
            img_folder = os.path.join(OUTPUT_DIR, img_id)
            os.makedirs(img_folder, exist_ok=True)
            
            # Map targets
            # Target_1, Target_2 (real ones), Target_fake
            target_info = [
                ("Target_1", real_classes[0]),
                ("Target_2", real_classes[1]),
                ("Target_fake", fake_animal)
            ]
            
            for folder_name, clean_target in target_info:
                target_path = os.path.join(img_folder, folder_name)
                os.makedirs(target_path, exist_ok=True)
                
                # 1. LeGrad
                prompt_target = f"a photo of a {clean_target}."
                target_emb = get_text_embedding(model, tokenizer, prompt_target, device)
                heatmap_legrad = compute_heatmap(model, img_t, target_emb)
                vis_legrad = get_heatmap_vis(original_image, heatmap_legrad)
                
                # 2. LeGrad OMP
                is_fake = folder_name == "Target_fake"
                neg_classes = []
                if is_fake:
                    neg_classes = real_classes
                else:
                    other_real = [c for c in real_classes if c != clean_target]
                    neg_classes = other_real + [fake_animal]
                
                D_list = []
                for nc in neg_classes:
                    prompt_nc = f"a photo of a {nc}."
                    D_list.append(get_text_embedding(model, tokenizer, prompt_nc, device))
                
                D = torch.cat(D_list, dim=0)
                target_emb_omp = omp_sparse_residual(target_emb, D, max_atoms=len(neg_classes))
                heatmap_omp = compute_heatmap(model, img_t, target_emb_omp)
                vis_omp = get_heatmap_vis(original_image, heatmap_omp)
                
                # Save individual images
                original_image.save(os.path.join(target_path, "original.png"))
                vis_legrad.save(os.path.join(target_path, "legrad.png"))
                vis_omp.save(os.path.join(target_path, "legrad_omp.png"))
                
                # Save target name for reference (invisible but useful)
                with open(os.path.join(target_path, f"target_{clean_target}.txt"), "w") as f:
                    f.write(clean_target)
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")

    print(f"Finished. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
