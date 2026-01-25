import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from diffusers import StableDiffusionPipeline
    from daam import trace
except ImportError as e:
    print(f"Warning: diffusers or daam not installed. DAAMSegmenter will fail if initialized. Error: {e}")
    StableDiffusionPipeline = None
    trace = None

class DAAMSegmenter:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-base", device='cuda'):
        if StableDiffusionPipeline is None or trace is None:
            raise ImportError("Please install 'daam' and 'diffusers' to use DAAMSegmenter.")
            
        print(f"[DAAM] Loading Stable Diffusion pipeline: {model_id}...")
        self.device = device
        # Use token=True to ensure we use the logged-in token. 
        # Note: use_auth_token is deprecated in favor of token=True in newer diffusers, 
        # but use_auth_token is what matched reference. Let's use `token=True` for modern diffusers.
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, token=True).to(device)
        except TypeError:
             # Fallback for older diffusers
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
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
        
        Args:
            image_pil: PIL Image
            prompt: Text prompt (e.g. "a photo of a cat")
            size: processing size (default 512 for SD)
            
        Returns:
            heatmap: torch.Tensor [H_original, W_original] normalized to [0, 1]
        """
        w, h = image_pil.size
        
        # 1. Preprocess image for VAE
        # Resize to 512x512 (standard for SD v2) implementation
        img_resized = image_pil.resize((size, size), resample=Image.BICUBIC)
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = img_arr * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        # VAE expects float32 or float16 depending on model, usually float32 is safer unless fp16 model
        # The pipeline is likely float32 unless specified otherwise. Let's match pipeline dtype.
        dtype = self.pipeline.unet.dtype
        img_tensor = img_tensor.to(dtype)

        # 2. Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # 3. Add Noise
        # Using timestep ~50 (out of 1000) or similar low noise level as per reference
        # Reference uses index 49 out of 50 inference steps, which is very little noise?
        # Actually in the reference 'run_daam_sd2.py': 
        #   noise = randn_tensor(...)
        #   init_latents = self.pipeline.scheduler.add_noise(init_latents, noise, timestep)
        #   where timestep is gathered from scheduler.timesteps
        # We will replicate a standard "add small noise" approach. 
        # If we use the scheduler directly with integer timestep:
        noise = torch.randn_like(latents)
        # Timestep: Low noise to keep structure. Reference used index 1 (or close to 0) in reverse?
        # Let's pick a timestep that corresponds to "almost clean". 
        # Standard SD scheduler usually goes 999 -> 0. 
        # A small timestep (e.g. 100) means little noise. 
        # Reference used `timestep = torch.tensor([10], device=self.device)` sort of logic or 
        # scheduler.timesteps[len(metrics)-1]. 
        # Let's stick to a fixed small timestep for "segmentation of real image"
        timestep = torch.tensor([50], device=self.device)  # 50/1000 is small noise
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # 4. Prepare Embeddings
        # We need the full prompt embedding
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # 5. Trace and Forward Pass
        # We trace the pipeline, but we manually call unet to ensure we control inputs
        # Actually daam.trace(pipeline) patches the UNet in the pipeline.
        # So we can just call self.unet inside the context.
        
        with trace(self.pipeline) as tc:
            with torch.no_grad():
                _ = self.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # 6. Extract Heatmap
            # DAAM accumulates attention.
            # We want the heatmap for the object class.
            # The prompt is usually "a photo of a {class}". 
            # We want the heat map for the class word.
            # Simple heuristic: last word if it's "a photo of a {class}"
            # Or we can just use the global heatmap if the prompt is specific.
            # But the reference uses `compute_word_heat_map(concept_word)`.
            
            # Let's extract the object name from the prompt.
            # Prompt format from benchmark: "a photo of a {class}."
            # We can try to parse it or just take the global heatmap.
            # If we take global, it highlights everything in prompt.
            # Ideally we pass the concept word.
            
            # Attempt to parse concept from "a photo of a {class}."
            if prompt.startswith("a photo of a "):
                concept = prompt[len("a photo of a "):].strip(".").strip()
            else:
                # Fallback: use the whole prompt or last word
                concept = prompt.split()[-1]

            global_heat_map = tc.compute_global_heat_map()
            # DAAM handles token matching. 
            try:
                word_heat_map = global_heat_map.compute_word_heat_map(concept)
                heatmap = word_heat_map.heatmap # [H, W] tensor
            except Exception:
                # Fallback if specific word not found (e.g. tokenizer mismatch)
                # Use global heatmap
                heatmap = global_heat_map.heatmap

        # 7. Post-process
        # Heatmap is already 512x512 (default daam size matches latent*stride?) 
        # Actually DAAM returns image-sized heatmap if configured, or latent sized.
        # Let's interpolate to be safe and match input PIL size.
        
        heatmap = heatmap.unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze() # [h, w]
        
        # Normalize [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.cpu()
