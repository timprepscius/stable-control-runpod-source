from datetime import datetime
print(f"SETUP ---- A {datetime.now()}");

import os
import torch
from diffusers.utils import load_image

print(f"SETUP ---- B {datetime.now()}");

sdxl_pipe = models.make_sdxli()
modify_pose_pipe = models.make_sdxl_ti_sketch_pose()

print(f"SETUP ---- E {datetime.now()}");
pose_image_path = "pose_1.png"
pose_image = load_image(pose_image_path)

print(f"SETUP ---- F {datetime.now()}");

def process(job_id, job_input):
    print(f"RUN ---- A {datetime.now()}");

    prompt = job_input['prompt']
    negative_prompt = job_input['negative_prompt']
    print(f"RUN WITH prompt:{prompt}, negative_prompt:{negative_prompt}, inference_steps:{inference_steps}")

    generated_images = sdxl_pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=inference_steps, 
        guidance_scale=0,
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

    print(f"RUN ---- C {datetime.now()}");

    pose_image_sized = pose_image.resize((job_input['width'], job_input['height']))

    posed_images = []
    for generated_image in generated_images:
        results = modify_pose_pipe(
            prompt=prompt,
            negative_prompt="",
            image=[generated_image, pose_image_sized],    # The original generated image
            guidance_scale=0,
            width=job_input['width'],
            height=job_input['height']
        ).images
        
        print(f"RUN ---- D {datetime.now()}");

        for result in results:
            posed_images.append(result)


    posed_images.append(pose_image)

    for i, sample in enumerate(posed_images):
        output_name = f"{job_id}-posed-{i}"
        output_path = f"/tmp/{output_name}.jpg"
        sample.save(output_path)
        output_paths.append(output_path)

    return output_paths


if __name__ == '__main__':
    test = models.load_test()
    process(test['id'], test['input'])
