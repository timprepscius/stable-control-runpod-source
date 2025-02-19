from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os

print(f"SETUP ---- B {datetime.now()}");

import torch
from diffusers.utils import load_image
import models

print(f"SETUP ---- C {datetime.now()}");

pipe = models.make_sdxl_ctrl_pose()
inference_steps = pipe.inference_steps
override_guidance_scale = None

pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- E {datetime.now()}");


def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    pose_image_sized = pose_image.resize((job_input['width'], job_input['height']))
    guidance_scale = job_input['guidance_scale']

    generated_images = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=inference_steps, 
        image=pose_image,
        guidance_scale=guidance_scale if override_guidance_scale is None else override_guidance_scale,
        image=pose_image_sized,
        width=job_input['width'],
        height=job_input['height']
    ).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_path = f"/tmp/generated-{job_id}-{i}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths

if __name__ == '__main__':
    test = models.load_test()
    process(test['id'], test['input'])


