from datetime import datetime

#######################################
#######################################

print(f"SETUP ---- 0A {datetime.now()}");

import os
os.system("git pull")

import uuid

worker_id = str(uuid.uuid4());
print(f"WORKER_ID: {worker_id}")

#######################################
#######################################

print(f"SETUP ---- 0B {datetime.now()}");

import runpod
import io
#from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schema import INPUT_SCHEMA


#######################################
#######################################

print(f"SETUP ---- 0C {datetime.now()}");

from rp_handler_sdxl_light_sdctrl import process

#######################################
#######################################

print(f"SETUP ---- 0D {datetime.now()}");

import base64
def b64of(fileName):
    with open(fileName, "rb") as f:
       out_data = f.read()

    return base64.b64encode(out_data).decode("utf-8")

def run(job):
    print(f"RUN ---- START {datetime.now()}");
    print(job)

    job_input = job['input']
    job_id = job['id']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    result = process(job_id, validated_input)

    job_output = []

    for index, img_path in enumerate(result):
        data = b64of(img_path)
        # image_url = rp_upload.upload_image(job['id'], img_path, index)

        job_output.append({
            "image": data,
            "path": img_path,
            "parameters": job["input"]
        })


    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    print(f"RUN ---- END {datetime.now()}");

    refresh_worker = job_input['restart'] == worker_id

    return { "refresh_worker": refresh_worker, "job_results": job_output }

if __name__ == '__main__':
    runpod.serverless.start({"handler": run})

