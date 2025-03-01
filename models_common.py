from datetime import datetime

import os
import torch
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


def parse_model_string(model_string):
    components = model_string.split("+")
    result = {"model": components[0], "vae": None, "scheduler": None}
    
    for comp in components[1:]:
        if "=" in comp:
            key, value = comp.split("=", 1)
            result[key] = value
    
    print(result)
    return result



