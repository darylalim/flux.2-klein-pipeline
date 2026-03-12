import os
import random

import streamlit as st
import torch
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.bfloat16


@st.cache_resource
def _get_pipe():
    device, dtype = _detect_device()

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=dtype,
        use_safetensors=True,
        token=hf_token,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pipe = _get_pipe()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    return image, seed


if __name__ == "__main__":
    st.set_page_config(page_title="FLUX.2 Klein", layout="centered")

    st.markdown(
        "# [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)"
    )

    prompt = st.text_input("Prompt", placeholder="Enter your prompt")

    with st.expander("Advanced Settings"):
        seed_val = st.slider(
            "Seed",
            min_value=0,
            max_value=MAX_SEED,
            value=0,
            step=1,
        )

        randomize_seed = st.checkbox("Randomize seed", value=True)

        col1, col2 = st.columns(2)
        with col1:
            width = st.slider(
                "Width",
                min_value=512,
                max_value=MAX_IMAGE_SIZE,
                value=1024,
                step=32,
            )
        with col2:
            height = st.slider(
                "Height",
                min_value=512,
                max_value=MAX_IMAGE_SIZE,
                value=1024,
                step=32,
            )

        col3, col4 = st.columns(2)
        with col3:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
            )
        with col4:
            num_inference_steps = st.slider(
                "Number of inference steps",
                min_value=1,
                max_value=20,
                value=4,
                step=1,
            )

    if st.button("Run", type="primary"):
        with st.spinner("Generating..."):
            image, used_seed = infer(
                prompt,
                seed_val,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
            )
        st.session_state.result_image = image
        st.session_state.result_seed = used_seed if randomize_seed else None

    if "result_image" in st.session_state:
        st.image(st.session_state.result_image)
        if st.session_state.result_seed is not None:
            st.caption(f"Seed: {st.session_state.result_seed}")
