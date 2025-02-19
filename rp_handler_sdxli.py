from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import torch
from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.utils import load_image



from PIL import Image

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print(f"SETUP ---- B {datetime.now()}");

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

inference_steps = 8
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

device = "cuda"

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Set the scheduler
pipe.to(device)
sdxl_pipe = pipe

def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    print(f"RUN WITH prompt:{prompt}, negative_prompt:{negative_prompt}, inference_steps:{inference_steps}")

    generated_images = sdxl_pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=inference_steps, 
        guidance_scale=0,
        width=job_input['width'],
        height=job_input['height']
    ).images

    print(f"RUN ---- B {datetime.now()}");

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_name = f"{job_id}-generated-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths