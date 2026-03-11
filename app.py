import os
import random
import warnings

import gradio as gr
import torch
from dotenv import load_dotenv

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message=".*CUDA is not available.*Disabling autocast.*"
    )
    from diffusers import Flux2KleinPipeline

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

MAX_SEED = 2_147_483_647
MAX_IMAGE_SIZE = 1440

_pipe = None


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.bfloat16


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    device, dtype = _detect_device()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
        _pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=dtype,
            use_safetensors=True,
            token=hf_token,
        )
    if device == "cuda":
        _pipe.enable_model_cpu_offload()
    else:
        _pipe.to(device)
    return _pipe


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


with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            " # [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)"
        )
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, variant="primary")

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    value=1.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples=[
                "A capybara wearing a suit holding a sign that reads Hello World"
            ],
            inputs=[prompt],
        )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch(css="#col-container { margin: 0 auto; max-width: 640px; }")
