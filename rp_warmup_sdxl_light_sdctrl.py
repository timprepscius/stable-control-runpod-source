import torch
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetImg2ImgPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

try:
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")

    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16
    )    

    controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )


except Exception as e:
    print(e)

print("finishing without reporting error")