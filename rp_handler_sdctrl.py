from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os

print(f"SETUP ---- B {datetime.now()}");

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import numpy as np

print(f"SETUP ---- C {datetime.now()}");

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
)

# Load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)

# Use a VAE for improved quality (optional but recommended for SDXL)
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Set the scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

print(f"SETUP ---- D {datetime.now()}");

pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- E {datetime.now()}");


def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    generated_images = pipe(prompt, negative_prompt, image=pose_image).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_path = f"/tmp/generated-{job_id}-{i}.png"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths


