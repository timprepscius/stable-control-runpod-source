from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import torch

import models
from diffusers.utils import load_image

print(f"SETUP ---- B {datetime.now()}");

sdxl_pipe = models.make_sdxl_ti_pose()

pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- F {datetime.now()}");

def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    pose_image_sized = pose_image.resize((job_input['width'], job_input['height']))

    print(f"RUN WITH prompt:{prompt}, negative_prompt:{negative_prompt}, inference_steps:{inference_steps}")

    generated_images = sdxl_pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=inference_steps, 
        guidance_scale=0,
        image=pose_image_sized,
        width=job_input['width'],
        height=job_input['height']
    ).images

    print(f"RUN ---- B {datetime.now()}");

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_name = f"{job_id}-generated-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths