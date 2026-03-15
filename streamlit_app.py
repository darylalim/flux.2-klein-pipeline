import os
import random

import streamlit as st
import torch
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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
def _get_vlm():
    device, dtype = _detect_device()
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        torch_dtype=dtype,
    ).to(device)
    return processor, model


UPSAMPLE_PROMPT_TEXT_ONLY = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent. Keep the enhanced prompt under 120 "
    "words.\n\n"
    "Guidelines:\n"
    "- Add concrete visual specifics: textures, materials, lighting, "
    "shadows, and spatial relationships.\n"
    "- Put ALL text that should appear in the image in quotation marks "
    "(signs, labels, screens, etc.) - without quotes, the model generates "
    "gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

UPSAMPLE_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words, ~30 for brief requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid vague words like "
    '"whimsical" or "cascading")\n'
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Turn negatives into positives "
    '("don\'t change X" becomes "keep X")\n'
    '- Make abstractions concrete ("futuristic" becomes '
    '"glowing cyan neon, metallic panels")\n\n'
    "Output only the final instruction in plain text and nothing else."
)


def upsample_prompt(prompt, image_list=None):
    try:
        processor, model = _get_vlm()
        device, _ = _detect_device()
        system_prompt = (
            UPSAMPLE_PROMPT_WITH_IMAGES if image_list else UPSAMPLE_PROMPT_TEXT_ONLY
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        ]
        if image_list:
            user_content = [
                *[{"type": "image"} for _ in image_list],
                {"type": "text", "text": prompt},
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            )
        prompt_text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        processor_kwargs = {"text": prompt_text, "return_tensors": "pt"}
        if image_list:
            processor_kwargs["images"] = image_list
        inputs = processor(**processor_kwargs).to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        enhanced = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()
        if not enhanced:
            return prompt
        return enhanced
    except Exception:
        st.warning("Prompt enhancement failed. Using original prompt.")
        return prompt


def _dimensions_from_images(image_list):
    """Calculate output dimensions matching the aspect ratio of the first input image."""
    w, h = image_list[0].size
    if w == 0 or h == 0:
        return 1024, 1024
    aspect = w / h
    if aspect >= 1:
        new_w = 1024
        new_h = round(1024 / aspect / 32) * 32
    else:
        new_h = 1024
        new_w = round(1024 * aspect / 32) * 32
    return max(512, min(MAX_IMAGE_SIZE, new_w)), max(512, min(MAX_IMAGE_SIZE, new_h))


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

    uploaded_files = st.file_uploader(
        "Input images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]

    _upload_key = (
        tuple((f.name, f.file_id) for f in uploaded_files) if uploaded_files else ()
    )
    if "prev_uploads" not in st.session_state:
        st.session_state.prev_uploads = ()
    if _upload_key != st.session_state.prev_uploads:
        st.session_state.prev_uploads = _upload_key
        if image_list:
            _w, _h = _dimensions_from_images(image_list)
            st.session_state.width_slider = _w
            st.session_state.height_slider = _h
        else:
            st.session_state.width_slider = 1024
            st.session_state.height_slider = 1024

    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("enhanced_prompt", None)
        st.session_state.pop("enhanced_prompt_area", None)

    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt, image_list=image_list)
        st.session_state.enhanced_prompt = enhanced

    if "enhanced_prompt" in st.session_state:
        final_prompt = st.text_area(
            "Enhanced Prompt",
            value=st.session_state.enhanced_prompt,
            key="enhanced_prompt_area",
        )
    else:
        final_prompt = prompt

    mode = st.radio(
        "Mode",
        options=["Distilled (4 steps)", "Base (50 steps)"],
        horizontal=True,
    )

    if "prev_mode" not in st.session_state:
        st.session_state.prev_mode = mode
    if mode != st.session_state.prev_mode:
        st.session_state.prev_mode = mode
        st.session_state.guidance_scale_slider = DEFAULT_CFG[mode]
        st.session_state.steps_slider = DEFAULT_STEPS[mode]

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
                key="width_slider",
            )
        with col2:
            height = st.slider(
                "Height",
                min_value=512,
                max_value=MAX_IMAGE_SIZE,
                value=1024,
                step=32,
                key="height_slider",
            )

        col3, col4 = st.columns(2)
        with col3:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=0.0,
                max_value=10.0,
                value=DEFAULT_CFG["Distilled (4 steps)"],
                step=0.1,
                key="guidance_scale_slider",
            )
        with col4:
            num_inference_steps = st.slider(
                "Number of inference steps",
                min_value=1,
                max_value=100,
                value=DEFAULT_STEPS["Distilled (4 steps)"],
                step=1,
                key="steps_slider",
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
                mode=mode,
                image_list=image_list,
            )
        st.session_state.result_image = image
        st.session_state.result_seed = used_seed if randomize_seed else None

    if "result_image" in st.session_state:
        st.image(st.session_state.result_image)
        if st.session_state.result_seed is not None:
            st.caption(f"Seed: {st.session_state.result_seed}")
