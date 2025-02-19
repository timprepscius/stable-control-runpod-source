import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, UniPCMultistepScheduler

try:
    inference_steps = 8
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors" # Use the correct ckpt for your step setting!

    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    hf_hub_download(repo, ckpt)

    print(f"SETUP ---- C0 {datetime.now()}");

    pose_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-pose-sdxl-1.0", 
        torch_dtype=torch.float16
    )

    print(f"SETUP ---- C2 {datetime.now()}");

    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        base, 
        unet=unet,
        vae=vae, 
        adapter=pose_adapter, 
        scheduler=euler_a, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )

except Exception as e:
    print(e)

print("finishing without reporting error")