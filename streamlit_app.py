import os
import random

import streamlit as st
import torch
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv
from transformers import GenerationConfig, pipeline as transformers_pipeline

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440

REPO_ID_DISTILLED = "black-forest-labs/FLUX.2-klein-4B"
REPO_ID_BASE = "black-forest-labs/FLUX.2-klein-base-4B"


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.bfloat16


def _load_pipe(repo_id):
    device, dtype = _detect_device()

    pipe = Flux2KleinPipeline.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        use_safetensors=True,
        token=hf_token,
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


@st.cache_resource
def _get_pipe_distilled():
    return _load_pipe(REPO_ID_DISTILLED)


@st.cache_resource
def _get_pipe_base():
    return _load_pipe(REPO_ID_BASE)


PIPES = {
    "Distilled (4 steps)": _get_pipe_distilled,
    "Base (50 steps)": _get_pipe_base,
}

DEFAULT_STEPS = {"Distilled (4 steps)": 4, "Base (50 steps)": 50}
DEFAULT_CFG = {"Distilled (4 steps)": 1.0, "Base (50 steps)": 4.0}


@st.cache_resource
def _get_llm():
    device, dtype = _detect_device()

    return transformers_pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dtype=dtype,
        device=device,
    )


UPSAMPLE_SYSTEM_PROMPT = (
    "You are a prompt engineer. Rewrite the user's text into a detailed, "
    "vivid image generation prompt. Keep it under 100 words. Output only "
    "the enhanced prompt, nothing else."
)


def upsample_prompt(prompt):
    try:
        llm = _get_llm()
        messages = [
            {"role": "system", "content": UPSAMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        generation_config = GenerationConfig(
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        result = llm(messages, generation_config=generation_config)
        enhanced = result[0]["generated_text"][-1]["content"].strip()
        if not enhanced:
            return prompt
        return enhanced
    except Exception:
        st.warning("Prompt enhancement failed. Using original prompt.")
        return prompt


def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=None,
    num_inference_steps=None,
    mode="Distilled (4 steps)",
    image_list=None,
):
    if guidance_scale is None:
        guidance_scale = DEFAULT_CFG[mode]
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_STEPS[mode]

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pipe = PIPES[mode]()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    pipe_kwargs = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
        "generator": generator,
    }

    if image_list is not None:
        pipe_kwargs["image"] = image_list

    with torch.inference_mode():
        image = pipe(**pipe_kwargs).images[0]

    return image, seed


if __name__ == "__main__":
    st.set_page_config(page_title="FLUX.2 Klein", layout="centered")

    st.markdown(
        "# [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)"
    )

    prompt = st.text_input("Prompt", placeholder="Enter your prompt")

    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("enhanced_prompt", None)
        st.session_state.pop("enhanced_prompt_area", None)

    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt)
        st.session_state.enhanced_prompt = enhanced

    if "enhanced_prompt" in st.session_state:
        final_prompt = st.text_area(
            "Enhanced Prompt",
            value=st.session_state.enhanced_prompt,
            key="enhanced_prompt_area",
        )
    else:
        final_prompt = prompt

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
                final_prompt,
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
