# Examples and Vision-Aware Upsampling Design

## Overview

Two features adapted from the official BFL FLUX.2 Klein demo:

1. **Pre-built examples** — 4 text-only prompts + 1 image-editing example with bundled images, presented as buttons below the prompt input.
2. **Vision-aware prompt upsampling** — Replace SmolLM2-1.7B (text-only) with SmolVLM-500M-Instruct (multimodal) so the upsampling model can see uploaded images when enhancing editing prompts.

## VLM Model Swap

Replace SmolLM2-1.7B-Instruct with SmolVLM-500M-Instruct as the single upsampling model for both text-only and image+text prompts.

### `_get_vlm()` replaces `_get_llm()`

Cached VLM loader using `@st.cache_resource`. Returns a `(processor, model)` tuple.

```python
@st.cache_resource
def _get_vlm():
    device, dtype = _detect_device()
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        torch_dtype=dtype,
    ).to(device)
    return processor, model
```

- `AutoProcessor` handles both tokenization and image preprocessing
- `AutoModelForVision2Seq` is the model class for SmolVLM
- Uses `torch_dtype` (like diffusers, unlike `transformers.pipeline` which uses `dtype`)
- ~1.2GB in bfloat16 (down from ~3.4GB for SmolLM2-1.7B)

### `upsample_prompt()` changes

The `has_images` parameter is removed. Replaced by an `image_list` parameter (optional list of PIL Images). System prompt is selected based on `if image_list` instead.

**Message format:** SmolVLM's chat template requires all message `content` values to use the list-of-dicts format `[{"type": "text", "text": "..."}]`, not plain strings. This applies to system messages, user messages, and all paths.

**Output extraction:** `model.generate()` + `processor.batch_decode()` returns the full sequence including the input prompt. To extract only the generated text, slice the output IDs to exclude input tokens before decoding: `output_ids[:, inputs["input_ids"].shape[1]:]`.

**Sampling parameters:** Carry over the existing sampling config (`do_sample=True`, `temperature=0.7`, `top_p=0.9`) as keyword arguments to `model.generate()` alongside `max_new_tokens=150`. The `GenerationConfig` wrapper is no longer needed since we call `model.generate()` directly.

**When images are present:**

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": UPSAMPLE_PROMPT_WITH_IMAGES}]},
    {
        "role": "user",
        "content": [
            {"type": "image"},  # one placeholder per image
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    },
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt_text, images=image_list, return_tensors="pt").to(device)
output_ids = model.generate(
    **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9,
)
result = processor.batch_decode(
    output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
)[0]
```

**When text-only:**

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": UPSAMPLE_PROMPT_TEXT_ONLY}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt_text, return_tensors="pt").to(device)
output_ids = model.generate(
    **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9,
)
result = processor.batch_decode(
    output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
)[0]
```

System prompt selection based on `if image_list` — `UPSAMPLE_PROMPT_WITH_IMAGES` vs `UPSAMPLE_PROMPT_TEXT_ONLY`.

Error handling unchanged — try/except around the entire VLM call, fall back to original prompt on any exception. Empty output also falls back to original prompt.

### Import changes

Remove:
- `from transformers import GenerationConfig, pipeline as transformers_pipeline`

Add:
- `from transformers import AutoModelForVision2Seq, AutoProcessor`

## Pre-built Examples

### Data

4 text-only prompts + 1 image-editing example:

```python
EXAMPLES = [
    {
        "label": "Gradient Vase",
        "prompt": "Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088",
        "images": None,
    },
    {
        "label": "Cat Sticker",
        "prompt": "A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose. The design should have smooth rounded lines with black outlines and soft gradient shading with pink cheeks.",
        "images": None,
    },
    {
        "label": "Capybara in Rain",
        "prompt": "Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo",
        "images": None,
    },
    {
        "label": "Berlin TV Tower",
        "prompt": "Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire.",
        "images": None,
    },
    {
        "label": "Multi-image Edit",
        "prompt": "The person from image 1 is petting the cat from image 2, the bird from image 3 is next to them",
        "images": ["examples/person.webp", "examples/cat.webp", "examples/bird.webp"],
    },
]
```

### Bundled images

3 small permissively-licensed `.webp` files in an `examples/` directory, <100KB each. Sourced from CC0/public domain.

### UI

Buttons in a row below the prompt input. Clicking a text-only example sets `st.session_state.example_prompt`. Clicking the image-editing example also sets `st.session_state.example_images` (list of PIL Images loaded from the bundled files).

Since `st.file_uploader` can't be programmatically populated, the image-editing example loads images directly into `image_list` via session state and displays them with `st.image()` previews, bypassing the uploader widget.

**Example images replace, not merge with, uploaded files.** When the user clicks the image-editing example, `st.session_state.example_images` is set and any previously uploaded files are ignored for that run. If the user then uploads new files via the uploader, `example_images` is cleared (same clearing logic as below).

**State clearing:** `st.session_state.example_prompt` is cleared when the user modifies the prompt text input. `st.session_state.example_images` is cleared when the user uploads new files or when the prompt text changes. This prevents stale example state from persisting after the user starts their own workflow.

## Testing

### VLM tests

`_make_mock_vlm()` returns a `(mock_processor, mock_model)` tuple. The mock processor must support: `apply_chat_template()` → returns a string, `__call__()` → returns a dict-like with `.to()` returning an object with `input_ids` attribute (a tensor), and `batch_decode()` → returns a list of strings. The mock model must support `generate()` → returns a tensor. `_reload_app()` patches `transformers.AutoProcessor` and `transformers.AutoModelForVision2Seq` at the `transformers` module level (matching the existing pattern of patching at import source, e.g., `patch("transformers.AutoProcessor")`).

**`TestVLMInit` (replaces `TestLLMInit`):**
- Verify `_get_vlm()` loads `HuggingFaceTB/SmolVLM-500M-Instruct` for both processor and model
- Verify `torch_dtype` is passed to model (not `dtype`)
- Verify device placement (MPS/CUDA/CPU) — model uses `.to(device)`
- Verify `@st.cache_resource` decoration

**`TestUpsamplePrompt` updates:**
- Verify multimodal message format when `image_list` is provided (messages contain `{"type": "image"}` placeholders, all `content` values use list-of-dicts format)
- Verify images are passed to `processor()` call
- Verify text-only path works (no image placeholders, no images to processor, content still uses list-of-dicts format)
- Verify output extraction slices `output_ids` to exclude input tokens
- Verify sampling parameters are passed to `model.generate()` (`do_sample`, `temperature`, `top_p`)
- Verify `has_images` parameter is removed from signature
- Keep existing tests: empty output fallback, exception fallback, system prompt selection

**Example tests:**
- Verify `EXAMPLES` list structure (each entry has `label`, `prompt`, `images` keys)
- Verify image-editing example references files that exist in `examples/`
- Verify bundled image files can be opened by PIL and are valid images

## Dependencies and File Changes

### Dependencies

No new pip packages. `transformers` already provides `AutoProcessor` and `AutoModelForVision2Seq`. Remove unused `GenerationConfig` import. `sentencepiece` stays.

### New files

- `examples/person.webp` — permissively licensed, <100KB
- `examples/cat.webp` — permissively licensed, <100KB
- `examples/bird.webp` — permissively licensed, <100KB

### CLAUDE.md updates

- Architecture: update "Prompt upsampling" section for VLM (`_get_vlm()` returns `(processor, model)` tuple, multimodal message format with list-of-dicts content, `AutoProcessor` + `AutoModelForVision2Seq`, `image_list` parameter replacing `has_images`, output extraction via input token slicing)
- Gotchas: replace SmolLM2 notes with SmolVLM patterns (`torch_dtype` like diffusers, `processor.apply_chat_template()` for multimodal messages, all message `content` must use `[{"type": "text", "text": ...}]` format). Remove `GenerationConfig` gotcha. Add note about `batch_decode` returning full sequence requiring input token slicing.
- Memory: update from ~3.4GB to ~1.2GB for VLM, peak from ~19.4GB to ~17.2GB

### README updates

- Mention example prompts and vision-aware prompt enhancement
