import sys
import os
import types
from unittest.mock import MagicMock

# --- MOCKING PROBLEMATIC MODULES ---
# We mock these BEFORE importing anything else to prevent numpy incompatibility crashes
# or backend issues (matplotlib).
# We use types.ModuleType and set __spec__ so that importlib and torch._dynamo are happy.

def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = types.SimpleNamespace(name=name, origin="mock", submodule_search_locations=None, loader=None, has_location=False)
    m.__file__ = "mock"
    m.__path__ = []
    # Add any submodules or attrs if needed
    return m

# 1. Matplotlib
m_mpl = create_mock_module("matplotlib")
m_mpl.pyplot = MagicMock()
m_mpl.colors = MagicMock()
m_mpl.cm = MagicMock()
sys.modules["matplotlib"] = m_mpl
sys.modules["matplotlib.pyplot"] = m_mpl.pyplot
sys.modules["matplotlib.colors"] = m_mpl.colors
sys.modules["matplotlib.cm"] = m_mpl.cm

# 2. Sklearn
m_sklearn = create_mock_module("sklearn")
m_sklearn.metrics = create_mock_module("sklearn.metrics")
sys.modules["sklearn"] = m_sklearn
sys.modules["sklearn.metrics"] = m_sklearn.metrics

# 3. Transformers
m_trans = create_mock_module("transformers")
sys.modules["transformers"] = m_trans

# Add 'tokenizers' just in case
m_tok = create_mock_module("tokenizers")
sys.modules["tokenizers"] = m_tok


import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
import open_clip

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Now import legrad (which imports utils -> matplotlib, but now it gets the mock)
from legrad import LeWrapper, LePreprocess

# Import OMP function from sparse_encoding.py
try:
    from sparse_encoding import omp_sparse_residual
except ImportError:
    # Fallback if import fails: Copy-paste the function
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

DATASET_ROOT = "/Users/emirhan/Desktop/pascal-voc-2012-DatasetNinja"
OUTPUT_DIR = os.path.join(project_root, "HCI")

import shutil
from PIL import Image, ImageOps, ImageDraw, ImageFont

# ... (Previous imports remain, but ensure PIL components are there)

# ... (Previous code) ...

def load_dataset_samples(num_samples=10):
    val_ann_dir = os.path.join(DATASET_ROOT, "val", "ann")
    val_img_dir = os.path.join(DATASET_ROOT, "val", "img")
    
    if not os.path.exists(val_ann_dir):
        print(f"Error: Annotation directory not found at {val_ann_dir}")
        return []

    # Find multi-class images that exist
    multi_class_files = []
    
    ann_files = [f for f in os.listdir(val_ann_dir) if f.endswith(".json")]
    print(f"Scanning {len(ann_files)} annotation files for multi-class images...")
    
    for ann_file in tqdm(ann_files):
        # ann_file is likes "2007_000033.jpg.json"
        img_filename = os.path.splitext(ann_file)[0]
        candidate_path = os.path.join(val_img_dir, img_filename)
        
        if os.path.exists(candidate_path):
            valid_img_path = candidate_path
            img_id = os.path.splitext(img_filename)[0] 
        else:
            continue
            
        with open(os.path.join(val_ann_dir, ann_file), "r") as f:
            data = json.load(f)
            classes = set()
            for obj in data.get("objects", []):
                class_title = obj.get("classTitle")
                if class_title and class_title != "neutral": 
                   classes.add(class_title)
            
            if len(classes) >= 3:
                # Store full info
                multi_class_files.append({
                    "id": img_id,
                    "ann_path": os.path.join(val_ann_dir, ann_file),
                    "img_path": valid_img_path,
                    "classes": list(classes),
                    "objects": data.get("objects", [])
                })
    
    print(f"Found {len(multi_class_files)} valid images with >= 3 unique classes.")
    
    if num_samples is None:
        return multi_class_files

    if len(multi_class_files) < num_samples:
        print(f"Warning: Only found {len(multi_class_files)} images, returning all.")
        return multi_class_files
        
    random.seed(42)
    sampled = random.sample(multi_class_files, num_samples)
    return sampled

    random.seed(42)
    sampled = random.sample(multi_class_files, num_samples)
    return sampled

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
    """
    Manual implementation of Jet colormap to avoid matplotlib dependency.
    Input: heatmap_np (H, W) in range [0, 1]
    Output: (H, W, 3) uint8 image
    """
    # 4 regions: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
    # Map to 0-4 values for easier lerp
    val = np.clip(heatmap_np * 4, 0, 4)
    
    # Initialize RGB with zeros
    rgb = np.zeros(heatmap_np.shape + (3,), dtype=np.float32)
    
    # 0 -> 1: Blue (0,0,1) to Cyan (0,1,1)
    mask = (val >= 0) & (val <= 1)
    # B=1, G=x, R=0
    rgb[mask, 2] = 1.0
    rgb[mask, 1] = val[mask]
    rgb[mask, 0] = 0.0
    
    # 1 -> 2: Cyan (0,1,1) to Green (0,1,0)
    mask = (val > 1) & (val <= 2)
    # B=2-x, G=1, R=0
    rgb[mask, 2] = 2.0 - val[mask]
    rgb[mask, 1] = 1.0
    rgb[mask, 0] = 0.0
    
    # 2 -> 3: Green (0,1,0) to Yellow (1,1,0)
    mask = (val > 2) & (val <= 3)
    # B=0, G=1, R=x-2
    rgb[mask, 2] = 0.0
    rgb[mask, 1] = 1.0
    rgb[mask, 0] = val[mask] - 2.0
    
    # 3 -> 4: Yellow (1,1,0) to Red (1,0,0)
    mask = (val > 3) & (val <= 4)
    # B=0, G=4-x, R=1
    rgb[mask, 2] = 0.0
    rgb[mask, 1] = 4.0 - val[mask]
    rgb[mask, 0] = 1.0
    
    return (rgb * 255).astype(np.uint8)

def get_heatmap_vis(image_pil, heatmap_tensor):
    """
    Return blend of image and heatmap using Jet colormap.
    """
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

def create_comparison_grid(img_id, original, results):
    """
    results: list of (class_name, normal_img, omp_img)
    """
    w, h = original.size
    rows = len(results)
    cols = 3 # Original+Text, Normal, OMP
    
    grid_w = w * cols
    grid_h = h * rows
    
    grid = Image.new('RGB', (grid_w, grid_h), "white")
    
    # Try to load a reasonable font
    font = None
    try:
        # Mac standard font location
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
             font = ImageFont.truetype("arial.ttf", 40)
        except:
             pass # Default font is tiny but better than crash
        
    for i, (cls_name, normal, omp) in enumerate(results):
        y = i * h
        
        # 1. Original with Text
        orig_with_text = original.copy()
        d = ImageDraw.Draw(orig_with_text)
        text = cls_name.capitalize()
        pos = (20, 20)
        # Shadow
        d.text((pos[0]+2, pos[1]+2), text, fill="black", font=font)
        d.text(pos, text, fill="white", font=font)
        
        grid.paste(orig_with_text, (0, y))
        grid.paste(normal, (w, y))
        grid.paste(omp, (2*w, y))
        
    grid.save(os.path.join(OUTPUT_DIR, f"{img_id}_comparison.png"))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing output directory {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    print("Loading LeGrad model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=448)
    
    # 2. Process ALL samples
    samples = load_dataset_samples(num_samples=None)
    
    print(f"Processing {len(samples)} images...")
    
    # 3. Run Experiments
    for sample in tqdm(samples):
        img_id = sample['id']
        img_path = sample['img_path']
        classes = sample['classes']
        
        try:
            original_image = Image.open(img_path).convert("RGB")
            img_t = preprocess(original_image).unsqueeze(0).to(device)
            
            # Precompute embeddings
            class_embeddings = {}
            for cls_name in classes:
                prompt = f"a photo of a {cls_name}."
                emb = get_text_embedding(model, tokenizer, prompt, device)
                class_embeddings[cls_name] = emb
            
            results = []
            
            for target_cls in classes:
                # 1. Normal LeGrad
                target_emb = class_embeddings[target_cls]
                heatmap_normal = compute_heatmap(model, img_t, target_emb)
                vis_normal = get_heatmap_vis(original_image, heatmap_normal)
                
                # 2. OMP LeGrad
                other_classes = [c for c in classes if c != target_cls]
                if not other_classes: # Should not happen with >=2 classes filtering
                    vis_omp = vis_normal
                else:
                    D_list = [class_embeddings[c] for c in other_classes]
                    D = torch.cat(D_list, dim=0) # [K, d]
                    target_emb_omp = omp_sparse_residual(target_emb, D, max_atoms=len(other_classes))
                    heatmap_omp = compute_heatmap(model, img_t, target_emb_omp)
                    vis_omp = get_heatmap_vis(original_image, heatmap_omp)

                results.append((target_cls, vis_normal, vis_omp))
            
            # Save combined grid
            create_comparison_grid(img_id, original_image, results)
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Experiments completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
