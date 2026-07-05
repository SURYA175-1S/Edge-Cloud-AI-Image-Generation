from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import base64
from io import BytesIO

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

print("Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# FIX: use stable scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cpu")

print("Model ready")

@app.post("/generate")
def generate(prompt: Prompt):

    result = pipe(
        prompt.prompt,
        num_inference_steps=15,
        guidance_scale=7.5
    )

    image = result.images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return {"image": img_str}