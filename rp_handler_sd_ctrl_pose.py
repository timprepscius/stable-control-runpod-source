from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os
import sys

from diffusers.utils import load_image
import models

print(f"SETUP ---- C {datetime.now()}");

def make_env(model_type="sdxl"):
    print(f"SETUP ---- B {datetime.now()}");

    pipe = models.make_ctrl_pose(model_type)

    pose_image_path = "pose_1.png"
    pose_image = load_image(pose_image_path)

    print(f"SETUP ---- F {datetime.now()}");

    return { "pipe": pipe, "pose_image": pose_image }


def process(env, job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    pipe = env["pipe"]
    pose_image = env["pose_image"]

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    pose_image_sized = pose_image.resize((job_input['width'], job_input['height']))
    guidance_scale = job_input['guidance_scale']

    generated_images = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=pipe.inference_steps, 
        guidance_scale=guidance_scale if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
        image=pose_image_sized,
        width=job_input['width'],
        height=job_input['height'],
        controlnet_conditioning_scale=0.1,
        control_guidance_start=0.2,
        # control_guidance_end=0.4
    ).images

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_path = f"/tmp/generated-{job_id}-{pipe.human_name}-{i}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths

if __name__ == '__main__':
    test = models.load_test()

    model_type = "sdxl" if len(sys.argv) < 2 else sys.argv[-1]
    env = make_env(model_type) 
    process(env, test['id'], test['input'])


