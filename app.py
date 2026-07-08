from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import base64
from io import BytesIO

app = FastAPI(title="Edge-Cloud AI Image Generation API")


class Prompt(BaseModel):
    prompt: str


print("Loading Stable Diffusion Model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe = pipe.to("cpu")

pipe.enable_attention_slicing()
pipe.safety_checker = None

print("Model Loaded Successfully")


@app.get("/")
def home():
    return {
        "message": "Edge-Cloud AI Image Generation API is Running"
    }


@app.post("/generate")
def generate(prompt: Prompt):

    result = pipe(
        prompt.prompt,
        height=384,
        width=384,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    image = result.images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "image": image_base64
    }
