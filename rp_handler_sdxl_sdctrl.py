from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os

import runpod
import io
#from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA

print(f"SETUP ---- B {datetime.now()}");

import torch
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image
import numpy as np

print(f"SETUP ---- C0 {datetime.now()}");

device = "cuda" if torch.cuda.is_available() else "cpu"

sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to(device)

print(f"SETUP ---- C1 {datetime.now()}");

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
).to(device)

print(f"SETUP ---- C2 {datetime.now()}");

controlnet_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)


print(f"SETUP ---- D {datetime.now()}");

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024))
    return image

    # image = np.array(image)
    # image = np.mean(image, axis=2).astype(np.uint8)  # Convert to grayscale (Canny edge input)
    # image = Image.fromarray(image)
    # return image

pose_image_path = "pose_1.png"
pose_image = preprocess_image(pose_image_path)

print(f"SETUP ---- E {datetime.now()}");


def process(job):
    print(f"RUN ---- A {datetime.now()}");

    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    print(f"RUN ---- B {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    generated_images = sdxl_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_path = f"/tmp/generated-{i}.png"
        sample.save(output_path)
        output_paths.append(output_path)

    posed_images = []
    for generated_image in generated_images:
        results = controlnet_pipe(
            image=generated_image,    # The original generated image
            control_image=pose_image, # The extracted pose
            strength=0.8,             # Controls how much the original image is altered
            num_inference_steps=30
        )      
        
        for result in results:
            posed_images.append(result)

    for i, sample in enumerate(posed_images):
        output_path = f"/tmp/posed-{i}.png"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths

def run(job):
    print(f"RUN ---- START {datetime.now()}");

    result = process(job)

    job_output = []

    for index, img_path in enumerate(result):
        image_url = rp_upload.upload_image(job['id'], img_path, index)

        job_output.append({
            "image": image_url,
            "path": img_path,
            "seed": validated_input['seed'] + index
        })

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    print(f"RUN ---- END {datetime.now()}");

    return job_output

if __name__ == '__main__':
    runpod.serverless.start({"handler": run})

