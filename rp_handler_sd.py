from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import sys
from diffusers.utils import load_image
import models

def make_env(model_type="sdxl"):
    print(f"SETUP ---- B {datetime.now()}");

    pipe = models.make(model_type)

    print(f"SETUP ---- F {datetime.now()}");

    return { "pipe": pipe }


def process(env, job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    pipe = env["pipe"]

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    guidance_scale = job_input['guidance_scale']

    generated_images = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=pipe.inference_steps, 
        guidance_scale=guidance_scale if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
        width=job_input['width'],
        height=job_input['height']
    ).images

    print(f"RUN ---- B {datetime.now()}");

    output_paths = []
    for i, sample in enumerate(generated_images):
        output_name = f"{job_id}-generated-{pipe.human_name}-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths

if __name__ == '__main__':
    test = models.load_test()

    model_type = "sdxl" if len(sys.argv) < 2 else sys.argv[-1]
    env = make_env(model_type) 
    process(env, test['id'], test['input'])
