from datetime import datetime

import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image

from models_common import *

def make_sdxl_ctrl_pose(inference_steps=60, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
    )

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base, controlnet=controlnet, torch_dtype=torch.float16
    )

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "UniPCMultistepScheduler")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None
    pipe.human_name = f"sdxl_ctrl_pose_vae_{model['vae']}_scheduler_{model['scheduler']}"

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


def make_sdxl(inference_steps=60, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "EulerDiscreteScheduler")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None
    pipe.human_name = f"sdxl_vae_{model['vae']}_scheduler_{model['scheduler']}"

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

def make_sdxl_turbo(inference_steps=1, device=device, model=empty_model):
    base = "stabilityai/sdxl-turbo"

    pipe = AutoPipelineForText2Image.from_pretrained(
        base, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )

    set_vae(model, pipe, None)
    set_scheduler(model, pipe, None)

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sdxl_turbo_{model['vae']}_scheduler_{model['scheduler']}"

    def runner(p):
        steps = dict_value_or_default(p, "inference_steps", pipe.inference_steps)

        return pipe(
            prompt=p["prompt"], 
            num_inference_steps=steps, 
            guidance_scale=0.0,
            width=p['width'],
            height=p['height']
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)


    return pipe        

def make_sdxl_ti_pose(inference_steps=60, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-xl-base-1.0"

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-openpose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C2 {datetime.now()}");

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        adapter=pose_adapter, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )  

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "EulerAncestralDiscreteScheduler")

    pipe.inference_steps = inference_steps  
    pipe.override_guidance_scale = None
    pipe.human_name = f"sdxl_ti_pose_vae_{model['vae']}_scheduler_{model['scheduler']}"

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

def make_sdxl_ti_sketch_pose(inference_steps=40, model=empty_model):
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

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        vae=vae, 
        adapter=MultiAdapter([reference_adapter, pose_adapter]), 
        scheduler=scheduler, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )    

    set_vae(model, pipe, "madebyollin")
    set_scheduler(model, pipe, "EulerAncestralDiscreteScheduler")

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = None
    pipe.human_name = f"sdxl_ti_sketch_pose_vae_{model['vae']}_scheduler_{model['scheduler']}"

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



