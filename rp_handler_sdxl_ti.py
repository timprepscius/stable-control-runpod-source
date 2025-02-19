from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
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

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_pipe.scheduler.config, timestep_spacing="trailing")

print(f"SETUP ---- C0 {datetime.now()}");

reference_adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
    torch_dtype=torch.float16
).to(device)

print(f"SETUP ---- C1 {datetime.now()}");

pose_adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
    torch_dtype=torch.float16
).to(device)

print(f"SETUP ---- C2 {datetime.now()}");

vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler")

modify_pose_pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    base, 
    vae=vae, 
    adapter=[reference_adapter, pose_adapter], 
    scheduler=euler_a, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

print(f"SETUP ---- E {datetime.now()}");
pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- F {datetime.now()}");

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

    print(f"RUN ---- C {datetime.now()}");

    pose_image_sized = pose_image.resize((job_input['width'], job_input['height']))

    posed_images = []
    for generated_image in generated_images:
        results = modify_pose_pipe(
            prompt=prompt,
            negative_prompt="",
            image=[generated_image, pose_image_sized],    # The original generated image
            guidance_scale=0,
            width=job_input['width'],
            height=job_input['height']
        ).images
        
        print(f"RUN ---- D {datetime.now()}");

        for result in results:
            posed_images.append(result)


    posed_images.append(pose_image)

    for i, sample in enumerate(posed_images):
        output_name = f"{job_id}-posed-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths