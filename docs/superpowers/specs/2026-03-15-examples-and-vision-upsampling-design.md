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

Gains an `image_list` parameter (optional list of PIL Images).

**When images are present:**

```python
messages = [
    {"role": "system", "content": UPSAMPLE_PROMPT_WITH_IMAGES},
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
output_ids = model.generate(**inputs, max_new_tokens=150)
result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
```

**When text-only:**

```python
messages = [
    {"role": "system", "content": UPSAMPLE_PROMPT_TEXT_ONLY},
    {"role": "user", "content": prompt},
]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt_text, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_new_tokens=150)
result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
```

System prompt selection unchanged — `UPSAMPLE_PROMPT_WITH_IMAGES` vs `UPSAMPLE_PROMPT_TEXT_ONLY` based on whether images are present.

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

On the next rerun, the prompt input reads from `st.session_state.example_prompt`, and the image handling logic merges `st.session_state.example_images` with any manually uploaded files.

## Testing

### VLM tests

`_make_mock_vlm()` returns a `(mock_processor, mock_model)` tuple. `_reload_app()` patches `AutoProcessor.from_pretrained` and `AutoModelForVision2Seq.from_pretrained` instead of `transformers.pipeline`.

**`TestVLMInit` (replaces `TestLLMInit`):**
- Verify `_get_vlm()` loads `HuggingFaceTB/SmolVLM-500M-Instruct` for both processor and model
- Verify `torch_dtype` is passed to model (not `dtype`)
- Verify device placement (MPS/CUDA/CPU)
- Verify `@st.cache_resource` decoration

**`TestUpsamplePrompt` updates:**
- Verify multimodal message format when `image_list` is provided (messages contain `{"type": "image"}` placeholders)
- Verify images are passed to `processor()` call
- Verify text-only path works (no image placeholders, no images to processor)
- Keep existing tests: empty output fallback, exception fallback, system prompt selection
- Update for `processor.apply_chat_template()` → `model.generate()` → `processor.batch_decode()` pipeline

**Example tests:**
- Verify `EXAMPLES` list structure (each entry has `label`, `prompt`, `images` keys)
- Verify image-editing example references files that exist in `examples/`

## Dependencies and File Changes

### Dependencies

No new pip packages. `transformers` already provides `AutoProcessor` and `AutoModelForVision2Seq`. Remove unused `GenerationConfig` import. `sentencepiece` stays.

### New files

- `examples/person.webp` — permissively licensed, <100KB
- `examples/cat.webp` — permissively licensed, <100KB
- `examples/bird.webp` — permissively licensed, <100KB

### CLAUDE.md updates

- Architecture: update "Prompt upsampling" section for VLM (`_get_vlm()` returns `(processor, model)` tuple, multimodal message format, `AutoProcessor` + `AutoModelForVision2Seq`)
- Gotchas: replace SmolLM2 notes with SmolVLM patterns (`torch_dtype` like diffusers, `processor.apply_chat_template()` for multimodal messages). Remove `GenerationConfig` gotcha.
- Memory: update from ~3.4GB to ~1.2GB for VLM, peak from ~19.4GB to ~17.2GB

### README updates

- Mention example prompts and vision-aware prompt enhancement
