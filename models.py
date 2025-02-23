from datetime import datetime

import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image

# from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
# from diffusers import StableDiffusion3Pipeline

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json

import uuid

device = "cuda"

def load_test():
    with open("test_input.json", "r") as f:
        test_input_json = json.load(f)

    test_input_json["id"] = str(uuid.uuid4())

    return test_input_json

empty_model = { "model": None, "vae": None, "scheduler": None }


def dict_value_or_default(m, k, d):
    if k in m:
        return m[k]

    return d

def value_or_default(v, d):
    if v is not None:
        return v

    return d

def make_scheduler(m, pipe, default=None):
    scheduler = value_or_default(m["scheduler"], default)
    if scheduler == "UniPCMultistepScheduler":
        return UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if scheduler == "EulerDiscreteScheduler":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    if scheduler == "EulerAncestralDiscreteScheduler":
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        return EulerAncestralDiscreteScheduler.from_pretrained(base, subfolder="scheduler", timestep_spacing="trailing")

    return None

def set_scheduler(m, pipe, default=None):
    scheduler = make_scheduler(m, pipe, default)
    if scheduler is not None:
        pipe.scheduler = scheduler

def make_vae(m, pipe, default=None):
    vae = value_or_default(m["vae"], default)
    if vae == "madebyollin":
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

def set_vae(m, pipe, default=None):
    vae = make_vae(m, pipe, default)
    if vae is not None:
        pipe.vae = vae


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

def make_sd3_turbo(inference_steps=4, device=device, model=empty_model):
    base = "stabilityai/stable-diffusion-3.5-large-turbo"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16
    )
    # pipeline.enable_model_cpu_offload()


    set_vae(model, pipe, None)
    set_scheduler(model, pipe, None)

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sd3_turbo_{model['vae']}_scheduler_{model['scheduler']}"

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

def parse_model_string(model_string):
    components = model_string.split("+")
    result = {"model": components[0], "vae": None, "scheduler": None}
    
    for comp in components[1:]:
        if "=" in comp:
            key, value = comp.split("=", 1)
            result[key] = value
    
    print(result)
    return result

def make_ti_pose(model_string, device=device):
    m = parse_model_string(model_string);
    model_type = m["model"]
    if model_type == "sdxl":
        return make_sdxl_ti_pose(device=device, model=m)
    if model_type == "sdxl-lightning":
        return make_sdxli_ti_pose(device=device, model=m)

    return None

def make_ctrl_pose(model_string, device=device):
    m = parse_model_string(model_string);
    model_type = m["model"]

    if model_type == "sdxl":
        return make_sdxl_ctrl_pose(device=device, model=m)
    if model_type == "sdxl-lightning":
        return make_sdxli_ctrl_pose(device=device, model=m)

    return None

def make(model_string, device=device):
    m = parse_model_string(model_string);
    model_type = m["model"]

    if model_type == "sdxl":
        return make_sdxl(device=device, model=m)
    if model_type == "sdxl-lightning":
        return make_sdxli(device=device, model=m)
    if model_type == "sdxl-turbo":
        return make_sdxl_turbo(device=device, model=m)
    if model_type == "sd3-turbo":
        return make_sd3_turbo(device=device, model=m)

    return None




