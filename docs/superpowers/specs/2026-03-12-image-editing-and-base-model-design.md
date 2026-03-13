# Image Editing and Base Model Toggle — Design Spec

## Overview

Add two features to the FLUX.2 Klein Streamlit app:

1. **Base model toggle** — support both Distilled (4 steps) and Base (50 steps) FLUX.2 Klein variants with a mode selector
2. **Image editing** — multi-image upload for editing/compositing via the pipeline's `image=` parameter

## Requirements

- 32GB unified memory (Apple Silicon); both models loaded lazily via `@st.cache_resource` (~8GB each in bfloat16 weights, ~16GB combined in system RAM). On CUDA, both use `enable_model_cpu_offload()` so weights live in CPU RAM and move to GPU on demand.
- Multi-image upload (not single image)
- Dual system prompts for upsampling (text-only vs. image-editing)
- Single-file architecture preserved (everything in `streamlit_app.py`)

## Design

### 1. Model Loading

Two cached pipeline functions, one per model variant:

```python
REPO_ID_DISTILLED = "black-forest-labs/FLUX.2-klein-4B"
REPO_ID_BASE = "black-forest-labs/FLUX.2-klein-base-4B"

@st.cache_resource
def _get_pipe_distilled():
    # loads REPO_ID_DISTILLED

@st.cache_resource
def _get_pipe_base():
    # loads REPO_ID_BASE
```

A `PIPES` dict maps mode names to getter functions. Constants define per-mode defaults:

```python
PIPES = {
    "Distilled (4 steps)": _get_pipe_distilled,
    "Base (50 steps)": _get_pipe_base,
}
DEFAULT_STEPS = {"Distilled (4 steps)": 4, "Base (50 steps)": 50}
DEFAULT_CFG = {"Distilled (4 steps)": 1.0, "Base (50 steps)": 4.0}
```

Pipelines are loaded lazily on first call via `@st.cache_resource`, then cached permanently. Both use the same device detection and loading logic as the current `_get_pipe()`.

### 2. Image Editing

Multi-image uploader using `st.file_uploader`:

```python
uploaded_files = st.file_uploader(
    "Input images (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
```

Uploaded files are converted to PIL Images and always passed as a list (even for a single image, for consistency). When images are present, they are passed to the pipeline via `image=image_list`. When no images are uploaded, `image` is omitted and behavior is identical to today (text-to-image). Both the Distilled and Base variants support `image=` — the feature is available in all modes.

The width/height sliders control the output image dimensions. Input images are preprocessed independently by the pipeline (aligned to VAE multiples, resized if total pixel area exceeds 1024 x 1024 (~1 megapixel)) — they are not resized to match the slider values. No additional image preprocessing is needed in our code.

The uploader sits between the prompt input and the mode selector.

### 3. Dual System Prompts

Replace the single `UPSAMPLE_SYSTEM_PROMPT` with two:

```python
UPSAMPLE_PROMPT_TEXT_ONLY = (
    "You are a prompt engineer. Rewrite the user's text into a detailed, "
    "vivid image generation prompt. Keep it under 100 words. Output only "
    "the enhanced prompt, nothing else."
)

UPSAMPLE_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words). Specify what changes and "
    "what stays the same. Use concrete language. Output only the final "
    "instruction, nothing else."
)
```

`upsample_prompt()` gains a `has_images` parameter to select the appropriate system prompt. The text-only prompt is the current one (unchanged). The image-editing prompt is adapted from the HF Space but simplified for SmolLM2 (text-only LLM — it won't see the images, just the editing request).

### 4. Updated `infer()` and UI

**`infer()` changes:**

- Add `mode` parameter (str, default `"Distilled (4 steps)"`) — selects pipeline via `PIPES[mode]()`
- Add `image_list` parameter (list of PIL Images or None)
- Change `num_inference_steps` and `guidance_scale` defaults to `None`. When `None`, look up from `DEFAULT_STEPS[mode]` and `DEFAULT_CFG[mode]` respectively. This ensures callers get correct defaults for the selected mode without needing to pass them explicitly.
- Pass `image=image_list` to pipeline when images are provided

**UI layout (top to bottom):**

1. Title
2. Prompt text input
3. Image uploader (new) — `st.file_uploader` with `accept_multiple_files=True` (placed before Enhance Prompt so `uploaded_files` is available for `has_images`)
4. Enhance Prompt button + enhanced prompt text area
5. Mode radio (new) — "Distilled (4 steps)" / "Base (50 steps)"
6. Advanced Settings expander — seed, dimensions, guidance scale, inference steps
7. Run button + result display

**Mode switching behavior:**

When the user changes mode, the inference steps and guidance scale sliders reset to that mode's defaults (overriding any user customization). This is done by writing the new defaults into `st.session_state` keys for the sliders.

**Slider range update:**

The inference steps slider `max_value` increases from 20 to 100 to accommodate the Base model's default of 50 steps.

**Upsampling context:**

"Enhance Prompt" passes `has_images=True/False` to `upsample_prompt()` based on whether images are uploaded.

## Changes Summary

| Area | Current | After |
|------|---------|-------|
| Models | 1 (Distilled) | 2 (Distilled + Base) |
| `_get_pipe()` | Single function | `_get_pipe_distilled()` + `_get_pipe_base()` |
| Image input | None | Multi-image upload |
| `infer()` params | prompt, seed, dimensions, CFG, steps | + mode, image_list |
| System prompts | 1 | 2 (text-only, image-editing) |
| `upsample_prompt()` | `(prompt)` | `(prompt, has_images)` |
| Memory usage | ~8GB (one model) + ~3.4GB (SmolLM2) | ~16GB (two models) + ~3.4GB (SmolLM2) ≈ ~19.4GB peak |

## Testing

Tests need to cover:

- Both pipeline getter functions load with correct repo IDs
- `infer()` selects the correct pipeline based on mode
- `infer()` passes `image=` when images are provided, omits it when not
- `upsample_prompt()` uses correct system prompt based on `has_images`
- Default steps/CFG values per mode
- Existing tests updated to accommodate new `mode` and `image_list` parameters on `infer()`
- Device detection and seed handling tests remain valid (unchanged logic)

## Documentation

Update `CLAUDE.md` to reflect:

- Two model variants and their repo IDs
- Updated `infer()` signature (new `mode` and `image_list` params)
- Dual system prompts
- Updated memory requirements (~16GB for both models)
- Image editing via `image=` parameter
