# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images using [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4B parameters) from Black Forest Labs. Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for editing workflows. Optional vision-aware prompt upsampling via [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) — the VLM can see uploaded images when enhancing editing prompts. Includes pre-built example prompts with bundled images.

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

Everything lives in `streamlit_app.py`, structured in five sections:

1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU), always bfloat16, cached with `@lru_cache`. `_load_pipe()` loads a pipeline by repo ID; `_get_pipe_distilled()` and `_get_pipe_base()` are `@st.cache_resource`-cached getters. `PIPES` maps mode names to getter functions. `MODE_DEFAULTS` holds per-mode defaults (Distilled: 4 steps, CFG 1.0; Base: 50 steps, CFG 4.0). On CUDA, uses `enable_model_cpu_offload()`; on MPS/CPU, uses `pipe.to(device)`.
2. **Examples** — `EXAMPLES` is a list of 5 dicts (`label`, `prompt`, `images`): 4 text-only prompts and 1 multi-image editing example with bundled `.webp` files in `examples/`.
3. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct via `AutoProcessor` + `AutoModelForImageTextToText`, cached with `@st.cache_resource`, returns a `(processor, model)` tuple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list=None)` selects the system prompt based on whether images are provided, builds multimodal list-of-dicts messages, and extracts output by slicing generated token IDs to exclude the input. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance, already_enhanced)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true and `already_enhanced` is false, returns `(prompt, was_enhanced)` tuple.
4. **Inference** — `infer()` takes prompt, seed, dimensions (512–1440px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Runs under `torch.inference_mode()` with a CPU-pinned generator. When `progress_callback` is provided, wraps it in a `callback_on_step_end` for per-step progress reporting. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 512–1440).
5. **UI** — Behind `if __name__ == "__main__"`: text input, example buttons, file uploader, enhance prompt button, mode radio, run button, image output, advanced settings expander. Advanced settings include an "Auto-enhance prompt" checkbox that runs `upsample_prompt()` before generation (skipped if already manually enhanced). A `st.progress` bar shows per-step inference progress. Example buttons populate prompt and images via session state; `_clear_enhancement()` removes all enhancement-related keys when context changes (new prompt, new example). Width/height sliders auto-update to match uploaded image aspect ratio; guidance scale and steps update when mode changes.

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

- **`from_pretrained` requires `torch_dtype`, not `dtype`.** Passing `dtype` is silently ignored.
- **On CUDA, use `enable_model_cpu_offload()` instead of `pipe.to(device)`.** Reduces VRAM to ~13GB. On MPS/CPU, use `pipe.to(device)` since CPU offload is CUDA-only.
- **FLUX.2 Klein does not support negative prompts.**
- **diffusers is installed from git.** `Flux2KleinPipeline` requires the latest main branch. Switch to PyPI once it ships in a stable release.
- **Both variants support `image=` for editing.** Pass a list of PIL Images. The pipeline preprocesses inputs (aligns to VAE multiples, resizes if >1MP). Width/height sliders control output dimensions, not input sizing.
- **Base uses different defaults than Distilled.** Base: 50 steps, CFG 4.0. Distilled: 4 steps, CFG 1.0.
- **`callback_on_step_end` must return the `callback_kwargs` dict.** The pipeline pops `latents` and `prompt_embeds` from the return value. Returning `None` or an empty dict causes the pipeline to lose track of latents.

### Transformers / SmolVLM

- **SmolVLM uses `AutoProcessor` + `AutoModelForImageTextToText`, not `transformers.pipeline`.** The processor handles both tokenization and image preprocessing.
- **`AutoModelForImageTextToText.from_pretrained` uses `torch_dtype`, like diffusers.** This is the `PreTrainedModel.from_pretrained` parameter, not the `transformers.pipeline` `dtype` parameter.
- **All message `content` must use list-of-dicts format.** SmolVLM's chat template requires `[{"type": "text", "text": "..."}]`, not plain strings. Applies to system and user messages on all paths.
- **`batch_decode` returns the full sequence including the input prompt.** Slice output to exclude input tokens: `output_ids[:, inputs["input_ids"].shape[1]:]`.
- **Pass sampling parameters directly to `model.generate()`.** Use `max_new_tokens`, `do_sample`, `temperature`, `top_p` as keyword arguments. No `GenerationConfig` wrapper needed.

### General

- **The generator is pinned to CPU, not the inference device.** MPS generators have reliability issues; CPU generators produce equivalent results across all backends.
- **Do not pin `sentencepiece==0.1.99`.** No pre-built wheel for macOS ARM64. The current unpinned version works.
- **All models share memory.** FLUX.2 Klein Distilled (~8GB) + Base (~8GB) + SmolVLM-500M (~1.2GB) in bfloat16 = ~17.2GB peak. All loaded lazily via `@st.cache_resource`.
