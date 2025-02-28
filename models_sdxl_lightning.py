from datetime import datetime

import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image

from models_common import *


def make_sdxli_ctrl_pose(inference_steps=8, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
    )

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base, unet=unet, controlnet=controlnet, torch_dtype=torch.float16
    )

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "UniPCMultistepScheduler")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sdxl_lightning_ctrl_pose_vae_{model['vae']}_scheduler_{model['scheduler']}"

    def runner(p):
        return pipe(
            prompt=p["prompt"], 
            negative_prompt=p["negative_prompt"], 
            num_inference_steps=pipe.inference_steps, 
            guidance_scale=p["guidance_scale"] if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
            width=p['width'],
            height=p['height']
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)

    return pipe   

def make_sdxli(inference_steps=8, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "EulerDiscreteScheduler")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sdxl_lightning_vae_{model['vae']}_scheduler_{model['scheduler']}"

    def runner(p):
        return pipe(
            prompt=p["prompt"], 
            negative_prompt=p["negative_prompt"], 
            num_inference_steps=pipe.inference_steps, 
            guidance_scale=p["guidance_scale"] if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
            width=p['width'],
            height=p['height']
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)

    return pipe

def make_sdxli_ti_pose(inference_steps=8, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C2 {datetime.now()}");

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        unet=unet,
        adapter=pose_adapter, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "EulerAncestralDiscreteScheduler")

    pipe.inference_steps = inference_steps  
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sdxl_lightning_ti_pose_vae_{model['vae']}_scheduler_{model['scheduler']}"

    def runner(p):
        return pipe(
            prompt=p["prompt"], 
            negative_prompt=p["negative_prompt"], 
            num_inference_steps=pipe.inference_steps, 
            guidance_scale=p["guidance_scale"] if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
            width=p['width'],
            height=p['height']
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)

    return pipe



