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

    generated_images = pipe.runner(job_input).images

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
