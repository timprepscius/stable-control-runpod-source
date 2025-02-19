from datetime import datetime

import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

device = "cuda"

def load_test():
    with open("test_input.json", "r") as f:
        test_input_json = json.load(f)

    test_input_json["id"] = "test_id"

    return test_input_json

def make_sdxl_ctrl_pose(inference_steps=60, device=device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
    )

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base, controlnet=controlnet, torch_dtype=torch.float16
    )

    # Use a VAE for improved quality (optional but recommended for SDXL)
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Set the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device is not None:
        pipe.to(device)

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None

    return pipe

def make_sdxli_ctrl_pose(inference_steps=60, device=device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
    )

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base, unet=unet, controlnet=controlnet, torch_dtype=torch.float16
    )

    # Use a VAE for improved quality (optional but recommended for SDXL)
    pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Set the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device is not None:
        pipe.to(device)

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None

    return pipe    

def make_sdxli(inference_steps=8, device=device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0

    if device is not None:
        pipe.to(device)

    return pipe

def make_sdxli_ti_pose(inference_steps=8, device=device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C2 {datetime.now()}");

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler", timestep_spacing="trailing")
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

    pipe.inference_steps = inference_steps  
    pipe.override_guidance_scale = 0

    if device is not None:
        pipe.to(device)

    return pipe

def make_sdxl_ti_pose(inference_steps=60, device=device):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C2 {datetime.now()}");

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler")

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        vae=vae, 
        adapter=pose_adapter, 
        scheduler=scheduler, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )  

    pipe.inference_steps = inference_steps  
    pipe.override_guidance_scale = None

    if device is not None:
        pipe.to(device)

    return pipe

def make_sdxl_ti_sketch_pose(inference_steps=40):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    reference_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C1 {datetime.now()}");

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

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
    )    

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None

    if device is not None:
        pipe.to(device)

    return pipe

def make_ti_pose(model_type, device=device):
    if model_type == "sdxl":
        return model_sdxl_ti_pose(device=device)
    if model_type == "sdxl-lightning"
        return model_sdxli_ti_pose(device=device)

    return None

def make_ctrl_pose(model_type, device=device):
    if model_type == "sdxl":
        return model_sdxl_ctrl_pose(device=device)
    if model_type == "sdxl-lightning":
        return model_sdxli_ctrl_pose(device=device)

    return None


