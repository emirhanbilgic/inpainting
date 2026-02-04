import sys
import os
import types
from unittest.mock import MagicMock

# --- MOCKING PROBLEMATIC MODULES ---
def create_mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = types.SimpleNamespace(name=name, origin="mock", submodule_search_locations=None, loader=None, has_location=False)
    m.__file__ = "mock"
    m.__path__ = []
    return m

m_mpl = create_mock_module("matplotlib")
m_mpl.pyplot = MagicMock()
m_mpl.colors = MagicMock()
m_mpl.cm = MagicMock()
sys.modules["matplotlib"] = m_mpl
sys.modules["matplotlib.pyplot"] = m_mpl.pyplot
sys.modules["matplotlib.colors"] = m_mpl.colors
sys.modules["matplotlib.cm"] = m_mpl.cm

m_sklearn = create_mock_module("sklearn")
m_sklearn.metrics = create_mock_module("sklearn.metrics")
sys.modules["sklearn"] = m_sklearn
sys.modules["sklearn.metrics"] = m_sklearn.metrics

m_trans = create_mock_module("transformers")
sys.modules["transformers"] = m_trans
m_tok = create_mock_module("tokenizers")
sys.modules["tokenizers"] = m_tok


import argparse
import json
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm
import open_clip

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from legrad import LeWrapper, LePreprocess

import math
from open_clip.timm_model import TimmModel

def compute_transformer_attribution(model, image, text_emb_1x, start_layer=1):
    """
    Computes Transformer Attribution (full LRP-based method from Chefer et al.).
    """
    model.zero_grad()
    
    blocks = list(model.visual.transformer.resblocks)
    num_layers = len(blocks)
    
    # Ensure start_layer is valid
    if start_layer < 0:
        start_layer = num_layers + start_layer
    start_layer = max(0, min(start_layer, num_layers - 1))
    
    num_prompts = text_emb_1x.shape[0]
    
    # Forward pass with gradients enabled to capture attention maps
    with torch.enable_grad():
        # Forward through visual encoder
        x = model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        batch_size = x.shape[0]
        class_token = model.visual.class_embedding.unsqueeze(0).unsqueeze(0)
        class_token = class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Add positional embedding
        num_patches = x.shape[1] - 1
        if hasattr(model.visual, 'original_pos_embed'):
            pos_embed = model.visual.original_pos_embed
        else:
            pos_embed = model.visual.positional_embedding
        
        if pos_embed.shape[0] != x.shape[1]:
            cls_pos = pos_embed[:1]
            patch_pos = pos_embed[1:]
            orig_size = int(math.sqrt(patch_pos.shape[0]))
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            new_size = int(math.sqrt(num_patches))
            patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bilinear', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_embed.shape[1])
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        
        x = x + pos_embed.unsqueeze(0).to(x.dtype)
        
        if hasattr(model.visual, 'ln_pre'):
            x = model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        
        # Forward through blocks and collect attention maps from start_layer onwards
        all_attn_weights = []
        for i, block in enumerate(blocks):
            if i < start_layer:
                # Forward normally for layers before start_layer
                x = block(x)
            else:
                # For layers from start_layer, capture attention weights
                attn_module = block.attn
                x_normed = block.ln_1(x)
                
                qkv = F.linear(x_normed, attn_module.in_proj_weight, attn_module.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                
                seq_len, bsz, embed_dim = q.shape
                num_heads = attn_module.num_heads
                head_dim = embed_dim // num_heads
                
                q = q.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                k = k.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                v = v.contiguous().view(seq_len, bsz * num_heads, head_dim).transpose(0, 1)
                
                scale = float(head_dim) ** -0.5
                attn_weights = torch.bmm(q * scale, k.transpose(-2, -1))
                attn_weights = F.softmax(attn_weights, dim=-1)  # [bsz*heads, N, N]
                all_attn_weights.append(attn_weights)
                
                attn_output = torch.bmm(attn_weights, v)
                attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
                attn_output = attn_module.out_proj(attn_output)
                
                x = x + attn_output
                x = x + block.mlp(block.ln_2(x))
        
        # Get final image features
        x = x.permute(1, 0, 2)
        image_features = model.visual.ln_post(x[:, 0, :]) @ model.visual.proj
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        sim = text_emb_1x @ image_features.transpose(-1, -2)
        one_hot = F.one_hot(torch.arange(0, num_prompts)).float().requires_grad_(True).to(text_emb_1x.device)
        s = torch.sum(one_hot * sim)
        
        # Compute gradients for all attention layers
        grads = torch.autograd.grad(s, all_attn_weights, retain_graph=False, create_graph=False)
        
        # Process each layer's gradient-weighted attention
        layer_contributions = []
        for i, (grad, attn_weights) in enumerate(zip(grads, all_attn_weights)):
            num_heads = blocks[start_layer + i].attn.num_heads
            
            # Reshape: [bsz*heads, N, N] -> [bsz, heads, N, N]
            grad = grad.view(bsz, num_heads, grad.shape[1], grad.shape[2])
            attn_weights = attn_weights.view(bsz, num_heads, attn_weights.shape[1], attn_weights.shape[2])
            
            # Apply ReLU to gradients
            grad = torch.clamp(grad, min=0)
            
            # Weight attention by gradients
            weighted_attn = grad * attn_weights  # [batch, heads, N, N]
            
            # Average over heads
            weighted_attn = weighted_attn.mean(dim=1)  # [batch, N, N]
            
            # Extract CLS to patches
            cls_to_patches = weighted_attn[:, 0, 1:]  # [batch, num_patches]
            
            layer_contributions.append(cls_to_patches)
        
        # Aggregate across layers (simple sum, as in reference)
        # Note: Full LRP would use more sophisticated propagation rules
        aggregated = sum(layer_contributions)  # [batch, num_patches]
        
        # Reshape to spatial grid
        num_patches = aggregated.shape[-1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            w = h = int(math.sqrt(num_patches))
            if w * h != num_patches:
                raise RuntimeError(f"Cannot reshape {num_patches} patches to square grid")
        else:
            w = h = grid_size
        
        heatmap = aggregated.reshape(bsz, 1, h, w)
        
        # Upsample to image size
        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap[0, 0].detach().cpu()

# Import OMP function
try:
    from sparse_encoding import omp_sparse_residual
except ImportError:
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
OUTPUT_DIR = os.path.join(project_root, "chefercam")

def load_dataset_samples(num_samples=None):
    val_ann_dir = os.path.join(DATASET_ROOT, "val", "ann")
    val_img_dir = os.path.join(DATASET_ROOT, "val", "img")
    
    if not os.path.exists(val_ann_dir):
        print(f"Error: Annotation directory not found at {val_ann_dir}")
        return []

    multi_class_files = []
    ann_files = [f for f in os.listdir(val_ann_dir) if f.endswith(".json")]
    
    for ann_file in tqdm(ann_files, desc="Scanning dataset"):
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
        return multi_class_files
        
    random.seed(42)
    sampled = random.sample(multi_class_files, num_samples)
    return sampled

def get_text_embedding(model, tokenizer, prompt, device):
    tok = tokenizer([prompt]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)
    return emb

def apply_colormap_jet(heatmap_np):
    val = np.clip(heatmap_np * 4, 0, 4)
    rgb = np.zeros(heatmap_np.shape + (3,), dtype=np.float32)
    
    mask = (val >= 0) & (val <= 1)
    rgb[mask, 2] = 1.0
    rgb[mask, 1] = val[mask]
    rgb[mask, 0] = 0.0
    
    mask = (val > 1) & (val <= 2)
    rgb[mask, 2] = 2.0 - val[mask]
    rgb[mask, 1] = 1.0
    rgb[mask, 0] = 0.0
    
    mask = (val > 2) & (val <= 3)
    rgb[mask, 2] = 0.0
    rgb[mask, 1] = 1.0
    rgb[mask, 0] = val[mask] - 2.0
    
    mask = (val > 3) & (val <= 4)
    rgb[mask, 2] = 0.0
    rgb[mask, 1] = 4.0 - val[mask]
    rgb[mask, 0] = 1.0
    
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

def create_comparison_grid(img_id, original, results):
    w, h = original.size
    rows = len(results)
    cols = 3 # Original+Text, Normal Chefer, OMP Chefer
    
    grid_w = w * cols
    grid_h = h * rows
    
    grid = Image.new('RGB', (grid_w, grid_h), "white")
    
    font = None
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
             font = ImageFont.truetype("arial.ttf", 40)
        except:
             pass 
        
    for i, (cls_name, normal, omp) in enumerate(results):
        y = i * h
        
        orig_with_text = original.copy()
        d = ImageDraw.Draw(orig_with_text)
        text = cls_name.capitalize()
        pos = (20, 20)
        d.text((pos[0]+2, pos[1]+2), text, fill="black", font=font)
        d.text(pos, text, fill="white", font=font)
        
        grid.paste(orig_with_text, (0, y))
        grid.paste(normal, (w, y))
        grid.paste(omp, (2*w, y))
        
    grid.save(os.path.join(OUTPUT_DIR, f"{img_id}_comparison.png"))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing output directory {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading CLIP (ViT-B-16)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()
    model = LeWrapper(model, layer_index=-2)
    preprocess = LePreprocess(preprocess=preprocess, image_size=448)
    
    samples = load_dataset_samples(num_samples=None)
    print(f"Processing {len(samples)} images with CheferCAM...")
    
    for sample in tqdm(samples):
        img_id = sample['id']
        img_path = sample['img_path']
        classes = sample['classes']
        
        try:
            original_image = Image.open(img_path).convert("RGB")
            img_t = preprocess(original_image).unsqueeze(0).to(device)
            
            class_embeddings = {}
            for cls_name in classes:
                prompt = f"a photo of a {cls_name}."
                emb = get_text_embedding(model, tokenizer, prompt, device)
                class_embeddings[cls_name] = emb
            
            results = []
            
            for target_cls in classes:
                # 1. Normal CheferCAM (Transformer Attribution)
                target_emb = class_embeddings[target_cls]
                
                # NOTE: CheferCAM takes model (LeWrapper) and text_emb
                heatmap_normal = compute_transformer_attribution(model, img_t, target_emb, start_layer=1)
                vis_normal = get_heatmap_vis(original_image, heatmap_normal)
                
                # 2. OMP CheferCAM
                other_classes = [c for c in classes if c != target_cls]
                if not other_classes:
                    vis_omp = vis_normal
                else:
                    D_list = [class_embeddings[c] for c in other_classes]
                    D = torch.cat(D_list, dim=0) 
                    target_emb_omp = omp_sparse_residual(target_emb, D, max_atoms=len(other_classes))
                    
                    # Compute CheferCAM with modified embedding
                    heatmap_omp = compute_transformer_attribution(model, img_t, target_emb_omp, start_layer=1)
                    vis_omp = get_heatmap_vis(original_image, heatmap_omp)

                results.append((target_cls, vis_normal, vis_omp))
            
            create_comparison_grid(img_id, original_image, results)
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Experiments completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
