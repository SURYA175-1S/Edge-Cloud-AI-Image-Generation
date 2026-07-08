import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource(show_spinner=False)
def load_model():
    # Load Stable Diffusion model for CPU
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        dtype=torch.float32  # float32 works on CPU
    )
    pipe = pipe.to("cpu")  # CPU only
    pipe.safety_checker = None
    pipe.enable_attention_slicing()  # Save memory
    return pipe

pipe = load_model()

st.title("Faster AI Scene Rendering Demo (CPU Optimized)")

prompt = st.text_input("Enter a scene description:")

if st.button("Generate Scene") and prompt:
    with st.spinner("Generating scene... This should be faster now."):
        # Lower resolution for faster CPU generation
        image = pipe(
            prompt,
            height=384,             # lower resolution
            width=384,              # lower resolution
            guidance_scale=7.5,
            num_inference_steps=20  # fewer steps → faster
        ).images[0]

        st.image(image, caption=prompt)
