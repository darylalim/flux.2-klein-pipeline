# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates images from text prompts using the [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (4B parameters) from Black Forest Labs via Hugging Face Diffusers. Includes optional prompt upsampling using [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) to enhance prompts before generation.

## Setup

```bash
uv sync
```

No license acceptance is required — FLUX.2 Klein is Apache 2.0 licensed. Optionally create a `.env` file with `HF_TOKEN=<your-token>` for authenticated Hugging Face access.

## Running

```bash
uv run streamlit run streamlit_app.py
```

## Architecture

Everything lives in `streamlit_app.py`, structured in four sections:

1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU). All devices use bfloat16 to match the model's native dtype. `_get_pipe()` loads the pipeline once via `@st.cache_resource`. On CUDA, it uses `enable_model_cpu_offload()` to reduce VRAM usage; on MPS/CPU, it uses `pipe.to(device)`.
2. **Inference** — `infer()` takes prompt, seed, dimensions (512-1440px), guidance scale (default 1.0), and inference steps (default 4). FLUX.2 Klein does not support negative prompts. Runs under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used.
3. **Prompt upsampling** — `_get_llm()` loads SmolLM2-1.7B-Instruct via `transformers.pipeline`, cached with `@st.cache_resource`. `upsample_prompt()` sends the user's prompt to the LLM using a chat message format and returns the enhanced text. The LLM is loaded lazily on first use.
4. **UI** — Streamlit interface behind `if __name__ == "__main__"` with text input, enhance prompt button, run button, image output, and an expander with advanced settings. Inference triggers on button click.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_streamlit_app.py  # Run a single test file
```

## Gotchas

### Diffusers / FLUX.2 Klein

- **`from_pretrained` requires `torch_dtype`, not `dtype`.** The `Flux2KleinPipeline.from_pretrained` API requires `torch_dtype`. Passing `dtype` causes it to be silently ignored. The `torch_dtype` deprecation warning originated from transformers, not diffusers — diffusers handles the translation internally as of the fix in huggingface/diffusers#12841.
- **On CUDA, use `enable_model_cpu_offload()` instead of `pipe.to(device)`.** This offloads model components to CPU when not in use, reducing VRAM requirements (~13GB). On MPS/CPU, use `pipe.to(device)` since CPU offload is CUDA-only.
- **FLUX.2 Klein does not support negative prompts.** The FLUX pipeline does not accept a `negative_prompt` parameter.
- **diffusers is installed from git.** `Flux2KleinPipeline` requires the latest diffusers from the main branch. The lockfile pins the exact commit for reproducibility. Switch to a PyPI release once `Flux2KleinPipeline` ships in a stable version.

### Transformers / SmolLM2

- **SmolLM2-Instruct requires the chat message format.** Use `messages=[{"role": "system", ...}, {"role": "user", ...}]` with `transformers.pipeline`, not raw text. The response is structured as `[{"generated_text": [{"role": "assistant", "content": "..."}]}]`.
- **`transformers.pipeline` uses `dtype`, not `torch_dtype`.** The `transformers` library has deprecated `torch_dtype` in favor of `dtype`. This is the opposite of diffusers, which requires `torch_dtype`.
- **Use `GenerationConfig` instead of loose generation kwargs.** Passing `max_new_tokens`, `do_sample`, etc. as keyword arguments alongside a model's built-in `generation_config` is deprecated. Wrap them in a `GenerationConfig` object instead.

### General

- **The generator is pinned to CPU, not the inference device.** The model card example uses `torch.Generator(device="cuda")`, but we use `device="cpu"` for cross-device compatibility. MPS generators have known reliability issues, and CPU generators produce equivalent results across all backends.
- **Do not pin `sentencepiece==0.1.99`.** That version has no pre-built wheel for macOS ARM64. The current unpinned version works.
- **Both models share memory.** FLUX.2 Klein (~8GB) and SmolLM2-1.7B (~3.4GB) in bfloat16 require ~11.4GB combined. The LLM is loaded lazily — it only consumes memory after the user clicks "Enhance Prompt".
