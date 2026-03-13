# Image Editing and Base Model Toggle — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image editing support (multi-image upload) and a base model toggle (Distilled vs Base) to the FLUX.2 Klein Streamlit app.

**Architecture:** Replace the single `_get_pipe()` with two cached pipeline getters (distilled + base) sharing a `_load_pipe()` helper. Add `mode` and `image_list` parameters to `infer()` with sentinel defaults that resolve from per-mode constants. Split the upsampling system prompt into text-only and image-editing variants selected by a `has_images` flag.

**Tech Stack:** Streamlit, PyTorch, Hugging Face Diffusers (`Flux2KleinPipeline`), Hugging Face Transformers (SmolLM2), PIL

**Spec:** `docs/superpowers/specs/2026-03-12-image-editing-and-base-model-design.md`

---

## File Structure

All changes are modifications to existing files:

- **`streamlit_app.py`** — new constants, dual pipeline loading, updated `infer()`, dual system prompts, UI additions (image uploader, mode radio, slider updates)
- **`tests/test_streamlit_app.py`** — update 5 broken tests (reference `_get_pipe()` which is removed), add new tests for mode selection, image list, dual prompts, new constants
- **`CLAUDE.md`** — update architecture docs, memory requirements, `infer()` signature, gotchas

---

## Chunk 1: Backend Changes

### Task 1: Dual Pipeline Loading

Replace `_get_pipe()` with two cached pipeline getters sharing a `_load_pipe()` helper. Add mode constants.

**Files:**
- Modify: `streamlit_app.py:14-40` (constants + pipeline loading)
- Modify: `tests/test_streamlit_app.py:42-51` (TestConstants — add new constant tests)
- Modify: `tests/test_streamlit_app.py:101-154` (TestPipelineInit — update for new function names)
- Modify: `tests/test_streamlit_app.py:475-500` (TestStreamlitApp — update cache_resource tests)

- [ ] **Step 1: Write failing tests for new constants**

Add to `TestConstants` in `tests/test_streamlit_app.py`:

```python
def test_repo_id_distilled(self):
    import streamlit_app

    assert streamlit_app.REPO_ID_DISTILLED == "black-forest-labs/FLUX.2-klein-4B"

def test_repo_id_base(self):
    import streamlit_app

    assert streamlit_app.REPO_ID_BASE == "black-forest-labs/FLUX.2-klein-base-4B"

def test_default_steps(self):
    import streamlit_app

    assert streamlit_app.DEFAULT_STEPS == {
        "Distilled (4 steps)": 4,
        "Base (50 steps)": 50,
    }

def test_default_cfg(self):
    import streamlit_app

    assert streamlit_app.DEFAULT_CFG == {
        "Distilled (4 steps)": 1.0,
        "Base (50 steps)": 4.0,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestConstants -v`
Expected: FAIL — `REPO_ID_DISTILLED`, `REPO_ID_BASE`, `DEFAULT_STEPS`, `DEFAULT_CFG` not defined

- [ ] **Step 3: Write failing tests for dual pipeline getters**

Replace `TestPipelineInit` with `TestPipelineLoading` in `tests/test_streamlit_app.py`:

```python
class TestPipelineLoading:
    def test_distilled_from_pretrained_args(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_distilled()
            mock_cls.from_pretrained.assert_called_once_with(
                "black-forest-labs/FLUX.2-klein-4B",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                token=streamlit_app.hf_token,
            )

    def test_base_from_pretrained_args(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_base()
            mock_cls.from_pretrained.assert_called_once_with(
                "black-forest-labs/FLUX.2-klein-base-4B",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                token=streamlit_app.hf_token,
            )

    def test_pipeline_moved_to_device(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_distilled()
            mock_pipe.to.assert_called_with("cpu")

    def test_pipeline_moved_to_mps_device(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_distilled()
            mock_pipe.to.assert_called_with("mps")

    def test_cpu_offload_on_cuda(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_distilled()
            mock_pipe.enable_model_cpu_offload.assert_called()
            mock_pipe.to.assert_not_called()
```

- [ ] **Step 4: Write failing tests for cache_resource decorators**

Update `TestStreamlitApp` — replace `test_get_pipe_uses_cache_resource` with two tests:

```python
def test_get_pipe_distilled_uses_cache_resource(self):
    """Verify _get_pipe_distilled is decorated with @st.cache_resource."""
    with (
        patch("diffusers.Flux2KleinPipeline"),
        patch("transformers.pipeline"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        import streamlit_app

        importlib.reload(streamlit_app)
        assert hasattr(streamlit_app._get_pipe_distilled, "clear")

def test_get_pipe_base_uses_cache_resource(self):
    """Verify _get_pipe_base is decorated with @st.cache_resource."""
    with (
        patch("diffusers.Flux2KleinPipeline"),
        patch("transformers.pipeline"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        import streamlit_app

        importlib.reload(streamlit_app)
        assert hasattr(streamlit_app._get_pipe_base, "clear")
```

- [ ] **Step 5: Run all new tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestPipelineLoading tests/test_streamlit_app.py::TestStreamlitApp::test_get_pipe_distilled_uses_cache_resource tests/test_streamlit_app.py::TestStreamlitApp::test_get_pipe_base_uses_cache_resource -v`
Expected: FAIL — `_get_pipe_distilled`, `_get_pipe_base` not defined

- [ ] **Step 6: Implement dual pipeline loading**

In `streamlit_app.py`, replace lines 14-40 (constants + `_get_pipe()`) with:

```python
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

# Temporary alias — infer() still references _get_pipe() until Task 2 updates it.
_get_pipe = _get_pipe_distilled
```

- [ ] **Step 7: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS. The old `TestPipelineInit` and `test_get_pipe_uses_cache_resource` tests have been replaced. The `_get_pipe = _get_pipe_distilled` alias keeps `infer()` and existing `TestInfer` tests working until Task 2 replaces the call with `PIPES[mode]()`.

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add dual pipeline loading for distilled and base models"
```

---

### Task 2: Update infer() for Mode Selection and Sentinel Defaults

Add `mode` and `image_list` parameters to `infer()`. Use sentinel defaults for `guidance_scale` and `num_inference_steps` that resolve from per-mode constants.

**Files:**
- Modify: `streamlit_app.py:85-110` (infer function)
- Modify: `tests/test_streamlit_app.py:157-291` (TestInfer — add new tests)

- [ ] **Step 1: Write failing tests for mode selection and image_list**

Add to `TestInfer` in `tests/test_streamlit_app.py`:

```python
def test_mode_selects_distilled_pipeline(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer("a cat", mode="Distilled (4 steps)")
        mock_cls.from_pretrained.assert_called_with(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            token=streamlit_app.hf_token,
        )

def test_mode_selects_base_pipeline(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer("a cat", mode="Base (50 steps)")
        mock_cls.from_pretrained.assert_called_with(
            "black-forest-labs/FLUX.2-klein-base-4B",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            token=streamlit_app.hf_token,
        )

def test_base_mode_default_steps_and_cfg(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer("a cat", mode="Base (50 steps)")
        mock_pipe.assert_called_once_with(
            prompt="a cat",
            guidance_scale=4.0,
            num_inference_steps=50,
            width=1024,
            height=1024,
            generator=ANY,
        )

def test_explicit_params_override_mode_defaults(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer(
            "a cat",
            mode="Base (50 steps)",
            guidance_scale=2.0,
            num_inference_steps=10,
        )
        mock_pipe.assert_called_once_with(
            prompt="a cat",
            guidance_scale=2.0,
            num_inference_steps=10,
            width=1024,
            height=1024,
            generator=ANY,
        )

def test_partial_override_steps_only(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer(
            "a cat",
            mode="Base (50 steps)",
            num_inference_steps=10,
        )
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 10
        assert call_kwargs["guidance_scale"] == 4.0

def test_image_list_passed_to_pipeline(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer("edit this", image_list=images)
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["image"] is images

def test_no_image_key_when_none(self):
    mock_pipe = _make_mock_pipe()
    streamlit_app, _ = _reload_app(mock_pipe)
    with (
        patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        streamlit_app.infer("a cat")
        call_kwargs = mock_pipe.call_args[1]
        assert "image" not in call_kwargs
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestInfer::test_mode_selects_distilled_pipeline tests/test_streamlit_app.py::TestInfer::test_mode_selects_base_pipeline tests/test_streamlit_app.py::TestInfer::test_base_mode_default_steps_and_cfg tests/test_streamlit_app.py::TestInfer::test_image_list_passed_to_pipeline tests/test_streamlit_app.py::TestInfer::test_no_image_key_when_none -v`
Expected: FAIL — `infer()` does not accept `mode` or `image_list`

- [ ] **Step 3: Remove temporary alias and implement updated infer()**

Delete the `_get_pipe = _get_pipe_distilled` alias added in Task 1 (it was a temporary bridge). Then replace `infer()` in `streamlit_app.py` with:

```python
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
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS. Existing `TestInfer` tests pass because the default mode is "Distilled (4 steps)" which resolves to the same defaults (steps=4, CFG=1.0) as the old hard-coded values.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add mode selection and image list support to infer()"
```

---

### Task 3: Dual System Prompts

Replace the single `UPSAMPLE_SYSTEM_PROMPT` with text-only and image-editing variants. Add `has_images` parameter to `upsample_prompt()`.

**Files:**
- Modify: `streamlit_app.py:55-82` (system prompts + upsample_prompt)
- Modify: `tests/test_streamlit_app.py:352-472` (TestUpsamplePrompt — add new tests)

- [ ] **Step 1: Write failing tests for dual system prompts**

Add a new expected prompt constant and new tests to `tests/test_streamlit_app.py`:

```python
EXPECTED_SYSTEM_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words). Specify what changes and "
    "what stays the same. Use concrete language. Output only the final "
    "instruction, nothing else."
)
```

Add to `TestUpsamplePrompt`:

```python
def test_uses_text_only_prompt_without_images(self):
    mock_pipe = _make_mock_pipe()
    mock_llm = _make_mock_llm()
    streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
    with (
        patch("streamlit_app.transformers_pipeline") as mock_tp,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_tp.return_value = mock_llm
        streamlit_app.upsample_prompt("a cat", has_images=False)
        messages = mock_llm.call_args[0][0]
        assert messages[0] == {
            "role": "system",
            "content": EXPECTED_SYSTEM_PROMPT,
        }

def test_uses_image_editing_prompt_with_images(self):
    mock_pipe = _make_mock_pipe()
    mock_llm = _make_mock_llm()
    streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
    with (
        patch("streamlit_app.transformers_pipeline") as mock_tp,
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_tp.return_value = mock_llm
        streamlit_app.upsample_prompt("make it blue", has_images=True)
        messages = mock_llm.call_args[0][0]
        assert messages[0] == {
            "role": "system",
            "content": EXPECTED_SYSTEM_PROMPT_WITH_IMAGES,
        }
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestUpsamplePrompt::test_uses_text_only_prompt_without_images tests/test_streamlit_app.py::TestUpsamplePrompt::test_uses_image_editing_prompt_with_images -v`
Expected: FAIL — `upsample_prompt()` does not accept `has_images`

- [ ] **Step 3: Implement dual system prompts**

In `streamlit_app.py`, replace `UPSAMPLE_SYSTEM_PROMPT` and `upsample_prompt()` with:

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


def upsample_prompt(prompt, has_images=False):
    try:
        llm = _get_llm()
        system_prompt = (
            UPSAMPLE_PROMPT_WITH_IMAGES if has_images else UPSAMPLE_PROMPT_TEXT_ONLY
        )
        messages = [
            {"role": "system", "content": system_prompt},
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
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS. Existing `TestUpsamplePrompt` tests pass because `has_images` defaults to `False`, using `UPSAMPLE_PROMPT_TEXT_ONLY` which has the same text as the old `UPSAMPLE_SYSTEM_PROMPT`.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add dual system prompts for text-only and image-editing modes"
```

---

## Chunk 2: UI and Documentation

### Task 4: UI Changes

Add image uploader, mode radio, update slider ranges, wire mode switching and image list to `infer()`.

**Files:**
- Modify: `streamlit_app.py:1-8` (add PIL import)
- Modify: `streamlit_app.py:113-209` (UI section)

- [ ] **Step 1: Add PIL import**

Add `from PIL import Image` to the imports in `streamlit_app.py`:

```python
import os
import random

import streamlit as st
import torch
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv
from PIL import Image
from transformers import GenerationConfig, pipeline as transformers_pipeline
```

- [ ] **Step 2: Add image uploader after prompt input**

**Spec deviation:** The spec places the image uploader after the Enhance Prompt button, but `uploaded_files` must be defined before it's referenced in `has_images=bool(uploaded_files)`. The image uploader is placed before the Enhance Prompt button instead.

In the `if __name__ == "__main__"` block, after the prompt text input (after line 120) and before the prompt change detection (`if "last_prompt" not in st.session_state`), insert the image uploader:

```python
    uploaded_files = st.file_uploader(
        "Input images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]
```

- [ ] **Step 3: Update "Enhance Prompt" to pass has_images**

Change the enhance prompt button handler from:

```python
    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt)
        st.session_state.enhanced_prompt = enhanced
```

to:

```python
    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt, has_images=bool(uploaded_files))
        st.session_state.enhanced_prompt = enhanced
```

- [ ] **Step 4: Add mode radio after enhanced prompt section**

After the enhanced prompt text area (after the `else: final_prompt = prompt` block), add the mode radio and mode-switching logic:

```python
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
```

The mode radio is placed after the enhanced prompt section and before the Advanced Settings expander. When the user switches modes, session state keys for the sliders are updated before the sliders render, so the sliders pick up the new defaults. The `value=` parameter on each slider only applies on first render; after that, the session state key takes precedence.

Final UI order:

1. Title
2. Prompt text input
3. Image uploader
4. Enhance Prompt button + enhanced prompt area
5. Mode radio
6. Advanced Settings expander
7. Run button + result

- [ ] **Step 5: Update slider ranges and add session state keys**

Update the Advanced Settings sliders. Changes:
- Guidance scale slider: add `key="guidance_scale_slider"`, increase `max_value` from `5.0` to `10.0` (allows experimentation beyond mode defaults)
- Inference steps slider: add `key="steps_slider"`, increase `max_value` from `20` to `100` (required for Base model's 50-step default)

```python
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
```

- [ ] **Step 6: Update Run button to pass mode and image_list**

Change the Run button handler to use keyword arguments for `mode` and `image_list` (avoids positional fragility with 9 parameters):

```python
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
```

- [ ] **Step 7: Run linting and type checking**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: PASS

- [ ] **Step 8: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add image uploader, mode radio, and updated sliders to UI"
```

---

### Task 5: Update CLAUDE.md

Update project documentation to reflect the new features.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Project Overview**

Update the Project Overview to mention image editing capability and both model variants. The current overview only describes text-to-image generation with the Distilled model.

- [ ] **Step 2: Update Architecture section**

In the Architecture section of CLAUDE.md, update section 1 (Model initialization) to describe:
- Two model variants: Distilled (`FLUX.2-klein-4B`, 4 steps, CFG 1.0) and Base (`FLUX.2-klein-base-4B`, 50 steps, CFG 4.0)
- `_load_pipe()` shared helper, `_get_pipe_distilled()` and `_get_pipe_base()` cached getters
- `PIPES` dict for mode-based lookup, `DEFAULT_STEPS` and `DEFAULT_CFG` constants

Update section 2 (Inference) to describe:
- `infer()` accepts `mode` (selects pipeline) and `image_list` (list of PIL Images for editing)
- Sentinel defaults: `guidance_scale=None` and `num_inference_steps=None` resolve from `DEFAULT_CFG[mode]` and `DEFAULT_STEPS[mode]`
- When `image_list` is provided, passes `image=image_list` to the pipeline

Update section 3 (Prompt upsampling) to describe:
- Two system prompts: `UPSAMPLE_PROMPT_TEXT_ONLY` (generation) and `UPSAMPLE_PROMPT_WITH_IMAGES` (editing)
- `upsample_prompt()` accepts `has_images` to select the appropriate prompt

Update section 4 (UI) to describe:
- Image uploader for multi-image input
- Mode radio for Distilled vs Base selection
- Slider defaults update on mode change

- [ ] **Step 3: Update Gotchas section**

Add under Diffusers / FLUX.2 Klein:
- **Both Distilled and Base variants support `image=` for editing.** Pass a list of PIL Images to the pipeline's `image` parameter. Input images are preprocessed by the pipeline (aligned to VAE multiples, resized if total pixel area exceeds ~1 megapixel). Width/height sliders control output dimensions, not input image sizing.
- **The Base model uses different defaults than Distilled.** Base: 50 steps, guidance scale 4.0. Distilled: 4 steps, guidance scale 1.0. The `infer()` function resolves these from `DEFAULT_STEPS[mode]` and `DEFAULT_CFG[mode]` when not explicitly provided.

Update the shared memory gotcha:
- **Both models and the LLM share memory.** FLUX.2 Klein Distilled (~8GB) + Base (~8GB) + SmolLM2-1.7B (~3.4GB) in bfloat16 require ~19.4GB peak. Both diffusion models are loaded lazily via `@st.cache_resource`. The LLM is loaded lazily on first "Enhance Prompt" use.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for image editing and base model features"
```
