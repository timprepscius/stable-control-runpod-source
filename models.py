import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


device = "cuda"

def make_sdxli(inference_steps=8):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    return pipe

def make_generate_sdxliti(inference_steps=8):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    ).to(device)

    print(f"SETUP ---- C2 {datetime.now()}");

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
    # scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        adapter=pose_adapter, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )

    return pipe

def make_modify_sdxlti(inference_steps=8):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

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

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler")

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        vae=vae, 
        adapter=MultiAdapter([reference_adapter, pose_adapter]), 
        scheduler=scheduler, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)    

