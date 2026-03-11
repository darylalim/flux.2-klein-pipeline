# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file Gradio web application that generates images from text prompts using the [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (4B parameters) from Black Forest Labs via Hugging Face Diffusers.

## Setup

```bash
uv sync
```

No license acceptance is required — FLUX.2 Klein is Apache 2.0 licensed. Optionally create a `.env` file with `HF_TOKEN=<your-token>` for authenticated Hugging Face access.

## Running

```bash
uv run python app.py
```

## Architecture

Everything lives in `app.py`, structured in three sections:

1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU). All devices use bfloat16 to match the model's native dtype. `_get_pipe()` lazily loads the pipeline on first inference. On CUDA, it uses `enable_model_cpu_offload()` to reduce VRAM usage; on MPS/CPU, it uses `pipe.to(device)`.
2. **Inference** — `infer()` takes prompt, seed, dimensions (512-1440px), guidance scale (default 1.0), and inference steps (default 4). FLUX.2 Klein does not support negative prompts. Runs under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used.
3. **UI** — Text input, run button, image output, and an accordion with advanced settings. Inference triggers on button click or prompt submission.

## Commands

```bash
uv run ruff check .              # Lint
uv run ruff check --fix .        # Lint with auto-fix
uv run ruff format .             # Format
uv run ruff format --check .     # Check formatting only
uv run ty check .                # Type check
uv run pytest                    # Run all tests
uv run pytest tests/test_app.py  # Run a single test file
```

## Gotchas

- **`from_pretrained` requires `torch_dtype`, not `dtype`.** The `Flux2KleinPipeline.from_pretrained` API requires `torch_dtype`. Passing `dtype` causes it to be silently ignored.
- **On CUDA, use `enable_model_cpu_offload()` instead of `pipe.to(device)`.** This offloads model components to CPU when not in use, reducing VRAM requirements (~13GB). On MPS/CPU, use `pipe.to(device)` since CPU offload is CUDA-only.
- **Do not remove the `warnings.catch_warnings()` blocks.** The block around the diffusers import suppresses a `torch.autocast(device_type="cuda")` warning on non-CUDA machines. The block in `_get_pipe()` suppresses the T5 tokenizer `add_prefix_space` and `torch_dtype` deprecation warnings.
- **FLUX.2 Klein does not support negative prompts.** The FLUX pipeline does not accept a `negative_prompt` parameter.
- **The generator is pinned to CPU, not the inference device.** The model card example uses `torch.Generator(device="cuda")`, but we use `device="cpu"` for cross-device compatibility. MPS generators have known reliability issues, and CPU generators produce equivalent results across all backends.
- **Do not pin `sentencepiece==0.1.99`.** That version has no pre-built wheel for macOS ARM64. The current unpinned version works.
- **diffusers is installed from git.** `Flux2KleinPipeline` requires the latest diffusers from the main branch. The lockfile pins the exact commit for reproducibility. Switch to a PyPI release once `Flux2KleinPipeline` ships in a stable version.
