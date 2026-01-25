from diffusers import StableDiffusionPipeline
import torch

try:
    pipe = StableDiffusionPipeline.from_pretrained("Manojb/stable-diffusion-2-base")
    pipe.scheduler.set_timesteps(50)
    print(f"Timesteps: {pipe.scheduler.timesteps}")
    print(f"Index 49: {pipe.scheduler.timesteps[49]}")
except Exception as e:
    print(f"Error: {e}")
