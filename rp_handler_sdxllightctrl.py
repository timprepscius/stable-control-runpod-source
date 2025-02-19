from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os

print(f"SETUP ---- B {datetime.now()}");

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, EulerDiscreteScheduler, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np

print(f"SETUP ---- C {datetime.now()}");

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
)

inference_steps = 8
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Set the scheduler
pipe.to("cuda")

print(f"SETUP ---- D {datetime.now()}");

pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- E {datetime.now()}");


def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    generated_images = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=inference_steps, 
        image=pose_image,
        guidance_scale=0
    ).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_path = f"/tmp/generated-{job_id}-{i}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths


