from datetime import datetime

import os
import torch

from diffusers import MultiAdapter, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers import AutoPipelineForText2Image

from transformers import BitsAndBytesConfig

from diffusers import SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline

from models_common import *

def make_sd3_turbo(inference_steps=4, device=device, model=empty_model):
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    
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
            num_inference_steps=pipe.inference_steps, 
            guidance_scale=0.0,
            max_sequence_length=512,
            width=p['width'],
            height=p['height']
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)

    return pipe        




