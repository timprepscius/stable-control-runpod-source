from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import torch
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetImg2ImgPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from PIL import Image

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print(f"SETUP ---- B {datetime.now()}");

device = "cuda" if torch.cuda.is_available() else "cpu"

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_pipe.scheduler.config, timestep_spacing="trailing")

print(f"SETUP ---- C {datetime.now()}");

# controlnet = ControlNetModel.from_pretrained(
#     "thibaud/controlnet-openpose-sdxl-1.0",
#     torch_dtype=torch.float16
# ).to(device)

# print(f"SETUP ---- CD {datetime.now()}");

# controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     controlnet=controlnet,
#     torch_dtype=torch.float16
# ).to(device)


print(f"SETUP ---- E {datetime.now()}");

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024))
    return image

pose_image_path = "pose_1.png"
pose_image = preprocess_image(pose_image_path)

print(f"SETUP ---- E {datetime.now()}");

def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''

    print(f"RUN ---- B {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    generated_images = sdxl_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=8).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_name = f"{job_id}-generated-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    # print(f"RUN ---- C {datetime.now()}");

    # posed_images = []
    # for generated_image in generated_images:
    #     results = controlnet_pipe(
    #         prompt=prompt,
    #         negative_prompt=negative_prompt,
    #         image=generated_image,    # The original generated image
    #         control_image=pose_image, # The extracted pose
    #         strength=0.8,             # Controls how much the original image is altered
    #         num_inference_steps=30,
    #         guidance_scale=job_input['guidance_scale'],
    #         width=job_input['width'],
    #         height=job_input['height']
    #     ).images
        
    #     print(f"RUN ---- D {datetime.now()}");

    #     for result in results:
    #         posed_images.append(result)

    # for i, sample in enumerate(posed_images):
    #     output_name = f"{job_id}-posed-{i}"
    #     output_path = f"/tmp/{output_name}.jpg"
    #     sample.save(output_path)
    #     output_paths.append(output_path)

    return output_paths

