# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images using [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4B parameters) from Black Forest Labs. Runs natively on Apple Silicon via [mflux](https://github.com/filipstrand/mflux) (diffusion) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (VLM). Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for editing workflows. Optional vision-aware prompt upsampling via [SmolVLM-500M-Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) — the VLM can see uploaded images when enhancing editing prompts. Includes pre-built example prompts with bundled images.

## Setup

```bash
uv sync
```

Requires Apple Silicon (M1+) and Python 3.12+.

## Running

```bash
uv run streamlit run streamlit_app.py
```

## Architecture

Everything lives in `streamlit_app.py`, structured in five sections:

1. **Model initialization** — `Flux2Klein(model_config=...)` from mflux creates models directly; MLX manages Apple Silicon unified memory automatically. `_get_model_distilled()` and `_get_model_base()` are `@st.cache_resource`-cached getters. `MODELS` maps mode names to getter functions. `MODE_DEFAULTS` holds per-mode defaults (Distilled: 4 steps, CFG 1.0; Base: 50 steps, CFG 4.0).
2. **Examples** — `EXAMPLES` is a list of 5 dicts (`label`, `prompt`, `images`): 4 text-only prompts and 1 multi-image editing example with bundled `.webp` files in `examples/`.
3. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct-bf16 via `mlx_vlm.load()`, cached with `@st.cache_resource`, returns a `(model, processor, config)` triple. Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (text-to-image, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (image editing, concrete language, preserve unchanged elements). `upsample_prompt(prompt, image_list=None)` selects the system prompt based on whether images are provided, formats messages via `mlx_vlm.prompt_utils.apply_chat_template`, and generates via `mlx_vlm.generate()`. Loaded lazily on first use. `_resolve_prompt(prompt, image_list, auto_enhance, already_enhanced)` wraps the auto-enhance decision: enhances only when `auto_enhance` is true and `already_enhanced` is false, returns `(prompt, was_enhanced)` tuple.
4. **Inference** — `infer()` takes prompt, seed, dimensions (512–1440px), mode, optional `image_list`, and optional `progress_callback`. Defaults resolve from `MODE_DEFAULTS[mode]`. Calls `model.generate_image()` from mflux. When `progress_callback` is provided, registers an `InLoopCallback` via `model.callbacks.register()` for per-step progress reporting. `_dimensions_from_images()` calculates output dimensions from the first uploaded image's aspect ratio (larger side 1024, rounds to 32, clamps 512–1440).
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

### mflux / FLUX.2 Klein

- **Models are created via `Flux2Klein(model_config=ModelConfig.flux2_klein_4b())`.** No repo ID strings or token parameters.
- **MLX manages device placement automatically.** Apple Silicon unified memory is used directly.
- **`generate_image()` returns a single PIL Image directly.**
- **Progress callbacks use `model.callbacks.register()`.** Register an object with a `call_in_loop(self, t, seed, prompt, latents, config, time_steps)` method.
- **`image_paths` accepts PIL Image objects at runtime.** Despite the type annotation (`list[Path | str]`), `ImageUtil.load_image()` handles PIL objects via isinstance check.
- **The guidance parameter is named `guidance`**, not `guidance_scale`.
- **FLUX.2 Klein does not support negative prompts.**
- **Base uses different defaults than Distilled.** Base: 50 steps, CFG 4.0. Distilled: 4 steps, CFG 1.0.

### mlx-vlm / SmolVLM

- **Use `mlx_vlm.load()` to get `(model, processor)` and `mlx_vlm.utils.load_config()` for config.** Config is required by `apply_chat_template`.
- **`mlx_vlm.generate()` handles tokenization and decoding internally.** Access result via `result.text`.
- **`apply_chat_template` takes `num_images` instead of embedding image tokens in messages.** Pass images as a flat list to the `image` parameter of `generate()`.
- **`temperature > 0` implies sampling.** Use `temperature=0.0` for greedy decoding.
- **Use `max_tokens`** to control output length.

### General

- **Apple Silicon is the primary target.** The app uses MLX (mflux + mlx-vlm) which requires Apple Silicon or Linux CUDA. CPU-only and Windows are not supported.
- **All models share memory via MLX unified memory.** FLUX.2 Klein Distilled + Base + SmolVLM in bfloat16. All loaded lazily via `@st.cache_resource`.
