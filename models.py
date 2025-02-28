from datetime import datetime

import os
import torch

from models_common import *
from models_flux import *

import uuid

device = "cuda"

def load_test():
    with open("test_input.json", "r") as f:
        test_input_json = json.load(f)

    test_input_json["id"] = str(uuid.uuid4())

    return test_input_json

empty_model = { "model": None, "vae": None, "scheduler": None }


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
    if model_type == "flux-schnell"
        return make_flux_schnell(device=device, model=m)
    if model_type == "flux-schnell-offline"
        return make_flux_schnell(device=device, model=m)

    return None




