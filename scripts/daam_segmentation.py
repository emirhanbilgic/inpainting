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

        # 3. Add Noise
        # Reference uses index 49/50 -> almost 0 noise.
        # We'll use a very small timestep effectively.
        noise = torch.randn_like(latents)
        # Using t=1 explicitly for low noise
        timestep = torch.tensor([1], device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)

        # Extract concept from "a photo of a {class}" or just use last word
        concept = ""
        if prompt.startswith("a photo of a "):
            concept = prompt[len("a photo of a "):].strip(".").strip()
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
