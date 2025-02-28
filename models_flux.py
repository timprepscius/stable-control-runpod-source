from models_common import *

from diffusers import FluxPipeline

def make_flux_schnell(inference_steps=4, device=device, model=empty_model):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    if model["model"] == "flux-schnell-offline":
        pipe.enable_model_cpu_offload()  # Offload model to CPU to save VRAM; remove if sufficient GPU memory is available

    set_vae(model, pipe, None)
    set_scheduler(model, pipe, None)

    pipe.inference_steps = inference_steps
    pipe.override_guidance_scale = 0
    pipe.human_name = f"sdxl_vae_{model['vae']}_scheduler_{model['scheduler']}"

    def runner(p):
        return pipe(
            prompt=p["prompt"], 
            negative_prompt=p["negative_prompt"], 
            num_inference_steps=pipe.inference_steps, 
            max_sequence_length=256,
            guidance_scale=p["guidance_scale"] if pipe.override_guidance_scale is None else pipe.override_guidance_scale,
            width=p['width'],
            height=p['height']
            # generator=torch.Generator("cpu").manual_seed(0)
        )

    pipe.runner = runner

    if device is not None:
        pipe.to(device, torch.float16)


    return pipe   





