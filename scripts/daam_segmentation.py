"""
DAAM Segmentation Module

This module provides DAAM (Diffusion Attentive Attribution Maps) based segmentation
for real images using Stable Diffusion's cross-attention mechanism.

Key methods:
- predict(): Basic DAAM heatmap generation for a target concept
- predict_key_space_omp(): Post-hoc orthogonalization of target heat map against 
  competing concept heat maps. This removes attention components that overlap with
  competitors, improving class discrimination.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
from typing import List

# Suppress the specific FutureWarning from daam/utils.py regarding torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

try:
    from diffusers import StableDiffusionPipeline
    from daam import trace
except ImportError as e:
    print(f"Warning: diffusers or daam not installed. DAAMSegmenter will fail if initialized. Error: {e}")
    StableDiffusionPipeline = None
    trace = None


class DAAMSegmenter:
    def __init__(self, model_id="Manojb/stable-diffusion-2-base", device='cuda'):
        if StableDiffusionPipeline is None or trace is None:
            raise ImportError("Please install 'daam' and 'diffusers' to use DAAMSegmenter.")
            
        print(f"[DAAM] Loading Stable Diffusion pipeline: {model_id}...")
        self.device = device
        # Use public mirror, no token needed
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        self.pipeline.enable_attention_slicing()
        
        # We need the VAE and UNet for the manual forward pass
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        
        print("[DAAM] Pipeline loaded.")

    def predict(self, image_pil, prompt, size=512):
        """
        Run DAAM on a real image by adding noise and tracing the restoration.
        """
        # Clear GPU cache between calls to avoid stale state
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        w, h = image_pil.size
        
        # 1. Preprocess image for VAE
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        img_tensor = img_tensor.to(self.pipeline.unet.dtype)

        # 2. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 3. Set up scheduler timesteps (REQUIRED for DAAM trace to work properly)
        # This must be called before add_noise to configure the scheduler
        self.scheduler.set_timesteps(50, device=self.device)
        
        # 4. Add Noise
        # Reference uses index 49/50 -> almost 0 noise.
        # We'll use a very small timestep effectively.
        noise = torch.randn_like(latents)
        # Using t=21 to match reference index 49 (Verified)
        timestep = torch.tensor([21], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # Extract concept from "a photo of a {class}" or just use last word
        concept = ""
        if prompt.startswith("a photo of a "):
            concept = prompt[len("a photo of a "):].strip(".").strip()
        elif prompt.startswith("a "):
            concept = prompt[2:].strip(".").strip()
        
        if not concept:
            # heuristic: last word
            concept = prompt.split()[-1]

        # Augment prompt to emphasize concept (Reference does this, potentially mismatching, but we align both)
        # Reference adds background concepts which act as interference/distractors.
        background_concepts = ["background", "floor", "tree", "person", "grass", "face"]
        
        # "a photo of a dog" -> "a photo of a dog, a dog, a background, a floor..."
        background_str = ", ".join([f"a {bc}" for bc in background_concepts])
        augmented_prompt = f"{prompt}, a {concept}, {background_str}"
        
        # 4. Prepare Embeddings WITHOUT CFG (Matching Reference)
        # Reference implementation does NOT concat unconditional embeddings, effectively running guidance=1.0
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # No Unconditional embedding / No concatenation
        # Reference runs UNet on single batch item (conditional only)
        prompt_embeds = text_embeddings
        latent_model_input = noisy_latents  # No concat with itself

        # 5. Trace and Forward Pass
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                # Perform forward pass (conditional only)
                _ = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds
                ).sample
            
            # 6. Extract Heatmap
            # Use augmented_prompt for decoding
            global_heat_map = tc.compute_global_heat_map(prompt=augmented_prompt)
            
            heatmap = None
            try:
                # Try exact concept match first
                word_heat_map = global_heat_map.compute_word_heat_map(concept)
                heatmap = word_heat_map.heatmap 
            except Exception:
                # Fallback: manually select tokens that are NOT special or "a", "photo", "of"
                # This is much safer than global average
                try:
                    # Tokenize prompt to see indices
                    tokens = self.tokenizer.encode(prompt)
                    # common IDs: 49406 (start), 49407 (end), 320(a), 1125(photo), 539(of)
                    # We want to avoid these if possible.
                    
                    # Alternatively, just grab the heatmaps for the concept words
                    # Split concept into words (if multi-word class like "great white shark")
                    sub_words = concept.split()
                    sub_heatmaps = []
                    for sw in sub_words:
                        try:
                            whm = global_heat_map.compute_word_heat_map(sw).heatmap
                            sub_heatmaps.append(whm)
                        except:
                            pass
                    
                    if sub_heatmaps:
                        # Average the heatmaps of the constituent words
                        heatmap = torch.stack(sub_heatmaps).mean(0)
                        
                except Exception:
                    pass

            # Final Fallback: use average of last few tokens (likely the object)
            if heatmap is None and hasattr(global_heat_map, 'heat_maps'):
                 # heat_maps shape: [num_tokens, H, W]
                 # The prompt is "a photo of a {class}". Class is at the end.
                 # Skip start(0), a(1), photo(2), of(3), a(4)... class(5..N), end(-1)
                 # So we take 5:-1
                 if global_heat_map.heat_maps.shape[0] > 6:
                     heatmap = global_heat_map.heat_maps[5:-1].mean(0)
                 else:
                     heatmap = global_heat_map.heat_maps.mean(0)
            
            if heatmap is None:
                # Should not happen unless heat_maps is missing
                 heatmap = torch.zeros((h, w))

        # 7. Post-process
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
        # Reference uses 'nearest' neighbor interpolation which gives blocky results -> lower mIoU
        heatmap = F.interpolate(heatmap, size=(h, w), mode='nearest')
        heatmap = heatmap.squeeze() # [h, w]
        
        # Normalize [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()

    def predict_key_space_omp(
        self,
        image_pil,
        prompt: str,
        target_concept: str,
        competing_concepts: List[str],
        size: int = 512,
        omp_beta: float = 1.0
    ) -> torch.Tensor:
        """
        Run DAAM with heat map orthogonalization (Post-hoc OMP).
        
        This method computes heat maps for the target concept and competing concepts
        separately using the standard predict() method, then orthogonalizes the 
        target heat map against the competing heat maps.
        
        The approach:
        1. Get heat map for target concept using predict()
        2. Get heat maps for each competing concept using predict()
        3. Subtract the projection of target onto competitors (orthogonalization)
        
        Args:
            omp_beta: Strength of orthogonalization (0.0=None, 1.0=Full, >1.0=Aggressive)
        """
        # 1. Get target heat map using standard DAAM predict()
        target_heatmap = self.predict(image_pil, prompt, size=size)
        
        if not competing_concepts or omp_beta == 0.0:
            # No orthogonalization needed
            return target_heatmap
        
        # 2. Get competing heat maps using standard DAAM predict()
        competing_heatmaps = []
        for comp_concept in competing_concepts:
            # Use prompt that includes the competing concept
            comp_prompt = f"a photo of a {comp_concept}."
            try:
                comp_hm = self.predict(image_pil, comp_prompt, size=size)
                competing_heatmaps.append(comp_hm)
            except Exception as e:
                # Skip if we can't get heat map for this concept
                print(f"[DAAM OMP] Warning: Could not get heat map for '{comp_concept}': {e}")
                continue
        
        # 3. Orthogonalize target against competitors
        if not competing_heatmaps:
            return target_heatmap
        
        heatmap = self._orthogonalize_heatmap(target_heatmap, competing_heatmaps, omp_beta)
        
        # Normalize to [0, 1] after orthogonalization
        heatmap = heatmap.clamp(min=0)  # ReLU to remove negative values
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _orthogonalize_heatmap(
        self, 
        target: torch.Tensor, 
        competitors: List[torch.Tensor], 
        beta: float
    ) -> torch.Tensor:
        """
        Orthogonalize target heat map against competing heat maps.
        
        Uses Gram-Schmidt-like projection to remove components of target
        that align with competitors.
        
        Args:
            target: Target heat map [H, W]
            competitors: List of competing heat maps [H, W]
            beta: Orthogonalization strength (0=none, 1=full projection)
        
        Returns:
            Orthogonalized heat map [H, W]
        """
        # Flatten heat maps for projection
        target_flat = target.flatten().float()
        
        result = target_flat.clone()
        
        for comp in competitors:
            comp_flat = comp.flatten().float()
            
            # Normalize competitor for projection
            comp_norm = comp_flat / (comp_flat.norm() + 1e-8)
            
            # Compute projection of target onto competitor
            projection = (target_flat @ comp_norm) * comp_norm
            
            # Subtract weighted projection
            result = result - beta * projection
        
        # Reshape back
        return result.reshape(target.shape)
