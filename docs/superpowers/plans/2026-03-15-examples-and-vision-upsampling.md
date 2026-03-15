# Examples and Vision-Aware Upsampling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SmolLM2-1.7B with SmolVLM-500M-Instruct for vision-aware prompt upsampling, and add pre-built example prompts with bundled images.

**Architecture:** Single-file app (`streamlit_app.py`) with all logic. SmolVLM-500M replaces the text-only LLM with a multimodal VLM using `AutoProcessor` + `AutoModelForVision2Seq`. Examples are a constant list rendered as buttons.

**Tech Stack:** Streamlit, PyTorch, Hugging Face Transformers (AutoProcessor, AutoModelForVision2Seq), Pillow

**Spec:** `docs/superpowers/specs/2026-03-15-examples-and-vision-upsampling-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `streamlit_app.py` | Modify | Replace `_get_llm` → `_get_vlm`, rewrite `upsample_prompt`, add `EXAMPLES`, add example buttons UI |
| `tests/test_streamlit_app.py` | Modify | Replace LLM mocks/tests with VLM mocks/tests, add example tests |
| `examples/person.webp` | Create | Bundled CC0 example image |
| `examples/cat.webp` | Create | Bundled CC0 example image |
| `examples/bird.webp` | Create | Bundled CC0 example image |
| `CLAUDE.md` | Modify | Update architecture, gotchas, memory for VLM + examples |
| `README.md` | Modify | Update features and memory |
| `pyproject.toml` | Modify | Update description |

---

## Chunk 1: VLM Model Swap (Test Infrastructure + _get_vlm + upsample_prompt)

### Task 1: Update test infrastructure

**Files:**
- Modify: `tests/test_streamlit_app.py:1-39`

- [ ] **Step 1: Replace `_make_mock_llm` with `_make_mock_vlm`**

Replace lines 15-21 with:

```python
class _ToableDict(dict):
    """A dict subclass with a .to() method that returns self, mimicking processor output."""

    def to(self, device):
        return self


def _make_mock_vlm():
    """Create a mock VLM (processor + model) pair."""
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "formatted prompt"
    input_ids = torch.tensor([[1, 2, 3]])
    mock_inputs = _ToableDict(
        {"input_ids": input_ids, "attention_mask": torch.tensor([[1, 1, 1]])}
    )
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["enhanced prompt"]

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.to.return_value = mock_model

    return mock_processor, mock_model
```

Note: `_ToableDict` is a dict subclass that supports both `**unpacking` (for `model.generate(**inputs, ...)`) and `inputs["input_ids"]` subscript access (for output slicing). A plain `MagicMock` cannot be unpacked with `**`.

- [ ] **Step 2: Update `_reload_app` signature and patches**

Replace lines 24-39 with:

```python
def _reload_app(
    mock_pipe, *, mock_vlm=None, mps_available=False, cuda_available=False
):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch("diffusers.Flux2KleinPipeline") as mock_cls,
        patch("transformers.AutoProcessor") as mock_ap,
        patch("transformers.AutoModelForVision2Seq") as mock_vm,
        patch("torch.backends.mps.is_available", return_value=mps_available),
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("streamlit.cache_resource", lambda f: f),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        if mock_vlm is not None:
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
        import streamlit_app

        importlib.reload(streamlit_app)
        return streamlit_app, mock_cls
```

- [ ] **Step 3: Run tests to verify nothing breaks yet**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py -x -q`

Expected: Tests that don't use the LLM still pass. LLM-related tests (TestLLMInit, TestUpsamplePrompt) will fail because they reference `mock_llm` — that's expected and will be fixed in Tasks 2-3.

- [ ] **Step 4: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: replace LLM mock infrastructure with VLM mocks"
```

---

### Task 2: Write and implement `_get_vlm` (TDD)

**Files:**
- Modify: `tests/test_streamlit_app.py:610-665` (replace TestLLMInit)
- Modify: `streamlit_app.py:9,65-74` (replace imports + _get_llm)

- [ ] **Step 1: Write failing `TestVLMInit` tests**

Replace `TestLLMInit` class (lines 610-665) with:

```python
class TestVLMInit:
    def test_vlm_loads_correct_model(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app._get_vlm()
            mock_ap.from_pretrained.assert_called_once_with(
                "HuggingFaceTB/SmolVLM-500M-Instruct"
            )
            mock_vm.from_pretrained.assert_called_once_with(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.bfloat16,
            )

    def test_vlm_device_cpu(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app._get_vlm()
            mock_model.to.assert_called_with("cpu")

    def test_vlm_device_mps(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm, mps_available=True)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app._get_vlm()
            mock_model.to.assert_called_with("mps")

    def test_vlm_device_cuda(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(
            mock_pipe, mock_vlm=mock_vlm, cuda_available=True
        )
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app._get_vlm()
            mock_model.to.assert_called_with("cuda")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestVLMInit -v`

Expected: FAIL — `_get_vlm` does not exist yet.

- [ ] **Step 3: Update imports in `streamlit_app.py`**

Replace line 9:
```python
from transformers import GenerationConfig, pipeline as transformers_pipeline
```
with:
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
```

- [ ] **Step 4: Replace `_get_llm` with `_get_vlm` in `streamlit_app.py`**

Replace lines 65-74:
```python
@st.cache_resource
def _get_llm():
    device, dtype = _detect_device()

    return transformers_pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dtype=dtype,
        device=device,
    )
```
with:
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

- [ ] **Step 5: Run `TestVLMInit` to verify tests pass**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestVLMInit -v`

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: replace _get_llm with _get_vlm for SmolVLM-500M-Instruct"
```

---

### Task 3: Rewrite `upsample_prompt` and its tests (TDD)

**Files:**
- Modify: `tests/test_streamlit_app.py:668-846` (TestUpsamplePrompt + expected constants)
- Modify: `streamlit_app.py:108-131` (upsample_prompt function)

- [ ] **Step 1: Replace `TestUpsamplePrompt` with VLM-based tests**

Replace lines 699-846 (the entire `TestUpsamplePrompt` class) with:

```python
class TestUpsamplePrompt:
    def test_chat_message_format_text_only(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("a cat")
            messages = mock_processor.apply_chat_template.call_args[0][0]
            assert messages[0] == {
                "role": "system",
                "content": [{"type": "text", "text": EXPECTED_SYSTEM_PROMPT}],
            }
            assert messages[1] == {
                "role": "user",
                "content": [{"type": "text", "text": "a cat"}],
            }
            kwargs = mock_processor.apply_chat_template.call_args[1]
            assert kwargs["add_generation_prompt"] is True

    def test_chat_message_format_with_images(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("make it blue", image_list=images)
            messages = mock_processor.apply_chat_template.call_args[0][0]
            assert messages[0] == {
                "role": "system",
                "content": [
                    {"type": "text", "text": EXPECTED_SYSTEM_PROMPT_WITH_IMAGES}
                ],
            }
            assert messages[1]["role"] == "user"
            content = messages[1]["content"]
            assert content[0] == {"type": "image"}
            assert content[1] == {"type": "image"}
            assert content[2] == {"type": "text", "text": "make it blue"}

    def test_images_passed_to_processor(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("edit", image_list=images)
            call_kwargs = mock_processor.call_args[1]
            assert call_kwargs["images"] is images

    def test_no_images_passed_to_processor_for_text_only(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("a cat")
            call_kwargs = mock_processor.call_args[1]
            assert "images" not in call_kwargs

    def test_generation_kwargs(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("a cat")
            gen_kwargs = mock_model.generate.call_args[1]
            assert gen_kwargs["max_new_tokens"] == 150
            assert gen_kwargs["do_sample"] is True
            assert gen_kwargs["temperature"] == 0.7
            assert gen_kwargs["top_p"] == 0.9

    def test_extracts_and_strips_output(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        mock_processor, mock_model = mock_vlm
        mock_processor.batch_decode.return_value = ["  A majestic feline  "]
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_empty_output_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        mock_processor, mock_model = mock_vlm
        mock_processor.batch_decode.return_value = [""]
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_whitespace_only_output_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        mock_processor, mock_model = mock_vlm
        mock_processor.batch_decode.return_value = ["   "]
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_exception_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        mock_processor, mock_model = mock_vlm
        mock_model.generate.side_effect = RuntimeError("OOM")
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("streamlit_app.st") as mock_st,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"
            mock_st.warning.assert_called_once_with(
                "Prompt enhancement failed. Using original prompt."
            )

    def test_empty_image_list_uses_text_only_path(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForVision2Seq") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app.upsample_prompt("a cat", image_list=[])
            messages = mock_processor.apply_chat_template.call_args[0][0]
            assert messages[0] == {
                "role": "system",
                "content": [{"type": "text", "text": EXPECTED_SYSTEM_PROMPT}],
            }
            call_kwargs = mock_processor.call_args[1]
            assert "images" not in call_kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestUpsamplePrompt -v`

Expected: FAIL — `upsample_prompt` still uses old LLM-based implementation.

- [ ] **Step 3: Rewrite `upsample_prompt` in `streamlit_app.py`**

Replace lines 108-131:
```python
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
with:
```python
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
```

- [ ] **Step 4: Update the UI call site**

In `streamlit_app.py` line 233, replace:
```python
            enhanced = upsample_prompt(prompt, has_images=bool(uploaded_files))
```
with:
```python
            enhanced = upsample_prompt(prompt, image_list=image_list)
```

- [ ] **Step 5: Run `TestUpsamplePrompt` to verify tests pass**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestUpsamplePrompt -v`

Expected: All 12 tests PASS.

- [ ] **Step 6: Run full test suite**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py -v`

Expected: `TestStreamlitApp` tests that reference `transformers.pipeline` or `_get_llm` will fail. All other tests should PASS.

- [ ] **Step 7: Fix `TestStreamlitApp` cache resource test**

In `tests/test_streamlit_app.py`, replace `test_get_llm_uses_cache_resource` (lines 876-887):
```python
    def test_get_llm_uses_cache_resource(self):
        """Verify _get_llm is decorated with @st.cache_resource (not passthrough)."""
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.pipeline"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_llm, "clear")
```
with:
```python
    def test_get_vlm_uses_cache_resource(self):
        """Verify _get_vlm is decorated with @st.cache_resource (not passthrough)."""
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForVision2Seq"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_vlm, "clear")
```

Also update the other two `TestStreamlitApp` tests that patch `transformers.pipeline`. Replace `patch("transformers.pipeline")` with `patch("transformers.AutoProcessor"), patch("transformers.AutoModelForVision2Seq")` in `test_get_pipe_distilled_uses_cache_resource` (line 854) and `test_get_pipe_base_uses_cache_resource` (line 867).

For `test_get_pipe_distilled_uses_cache_resource`, replace:
```python
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.pipeline"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
```
with:
```python
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForVision2Seq"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
```

Same change for `test_get_pipe_base_uses_cache_resource`.

- [ ] **Step 8: Run full test suite to verify all pass**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py -v`

Expected: ALL tests PASS.

- [ ] **Step 9: Lint and format**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: rewrite upsample_prompt for SmolVLM-500M multimodal support"
```

---

## Chunk 2: Pre-built Examples

### Task 4: Add example images

**Files:**
- Create: `examples/person.webp`
- Create: `examples/cat.webp`
- Create: `examples/bird.webp`

- [ ] **Step 1: Create `examples/` directory**

```bash
mkdir -p "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline/examples"
```

- [ ] **Step 2: Generate placeholder images with Pillow**

Since we need permissively-licensed images, create simple placeholder images using Pillow (solid color with text label). These can be swapped for real CC0 images later.

Create a small script and run it:

```bash
cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run python -c "
from PIL import Image, ImageDraw
for name, color in [('person', (70, 130, 180)), ('cat', (255, 165, 0)), ('bird', (34, 139, 34))]:
    img = Image.new('RGB', (512, 512), color)
    draw = ImageDraw.Draw(img)
    draw.text((200, 240), name, fill='white')
    img.save(f'examples/{name}.webp', 'WEBP', quality=80)
"
```

- [ ] **Step 3: Verify images exist and are valid**

```bash
cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && ls -la examples/ && uv run python -c "
from PIL import Image
for name in ['person', 'cat', 'bird']:
    img = Image.open(f'examples/{name}.webp')
    print(f'{name}.webp: {img.size}, {img.mode}')
"
```

- [ ] **Step 4: Commit**

```bash
git add examples/
git commit -m "feat: add placeholder example images for multi-image editing demo"
```

---

### Task 5: Add EXAMPLES constant and tests (TDD)

**Files:**
- Modify: `tests/test_streamlit_app.py` (add TestExamples class at end)
- Modify: `streamlit_app.py` (add EXAMPLES constant after DEFAULT_CFG)

- [ ] **Step 1: Write failing example tests**

Add at the end of `tests/test_streamlit_app.py`:

```python
class TestExamples:
    def test_examples_list_structure(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        assert isinstance(streamlit_app.EXAMPLES, list)
        assert len(streamlit_app.EXAMPLES) == 5
        for example in streamlit_app.EXAMPLES:
            assert "label" in example
            assert "prompt" in example
            assert "images" in example

    def test_text_only_examples_have_no_images(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        for example in streamlit_app.EXAMPLES[:4]:
            assert example["images"] is None

    def test_image_example_has_valid_paths(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        image_example = streamlit_app.EXAMPLES[4]
        assert image_example["images"] is not None
        assert len(image_example["images"]) == 3

    def test_bundled_images_are_valid(self):
        import os

        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        image_example = streamlit_app.EXAMPLES[4]
        for path in image_example["images"]:
            assert os.path.exists(path), f"Missing: {path}"
            img = Image.open(path)
            assert img.size[0] > 0 and img.size[1] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestExamples -v`

Expected: FAIL — `EXAMPLES` does not exist.

- [ ] **Step 3: Add `EXAMPLES` constant to `streamlit_app.py`**

Add after `DEFAULT_CFG` (after line 62):

```python
EXAMPLES = [
    {
        "label": "Gradient Vase",
        "prompt": (
            "Create a vase on a table in living room, the color of the vase is "
            "a gradient of color, starting with #02eb3c color and finishing with "
            "#edfa3c. The flowers inside the vase have the color #ff0088"
        ),
        "images": None,
    },
    {
        "label": "Cat Sticker",
        "prompt": (
            "A kawaii die-cut sticker of a chubby orange cat, featuring big "
            "sparkly eyes and a happy smile with paws raised in greeting and a "
            "heart-shaped pink nose. The design should have smooth rounded lines "
            "with black outlines and soft gradient shading with pink cheeks."
        ),
        "images": None,
    },
    {
        "label": "Capybara in Rain",
        "prompt": (
            "Soaking wet capybara taking shelter under a banana leaf in the "
            "rainy jungle, close up photo"
        ),
        "images": None,
    },
    {
        "label": "Berlin TV Tower",
        "prompt": (
            "Photorealistic infographic showing the complete Berlin TV Tower "
            "(Fernsehturm) from ground base to antenna tip, full vertical view "
            "with entire structure visible including concrete shaft, metallic "
            "sphere, and antenna spire."
        ),
        "images": None,
    },
    {
        "label": "Multi-image Edit",
        "prompt": (
            "The person from image 1 is petting the cat from image 2, the bird "
            "from image 3 is next to them"
        ),
        "images": ["examples/person.webp", "examples/cat.webp", "examples/bird.webp"],
    },
]
```

- [ ] **Step 4: Run example tests to verify they pass**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py::TestExamples -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Lint and format**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add EXAMPLES constant with 4 text-only + 1 multi-image example"
```

---

### Task 6: Add example buttons to UI

**Files:**
- Modify: `streamlit_app.py` (UI section, after prompt input)

**Key design note:** When an example button is clicked, we must update both `last_prompt` and the text input's session state key so that (a) the text input widget displays the example prompt on rerun, and (b) the prompt-change detection does not immediately clear the example state. Without this, the example prompt self-clears on the very rerun that sets it.

- [ ] **Step 1: Add a session state key to the prompt text input**

Replace line 196:
```python
    prompt = st.text_input("Prompt", placeholder="Enter your prompt")
```
with:
```python
    prompt = st.text_input("Prompt", placeholder="Enter your prompt", key="prompt_input")
```

This lets us programmatically set the text input value via `st.session_state.prompt_input`.

- [ ] **Step 2: Add example buttons after the prompt text input**

After the text input line, before the file uploader, add:

```python
    example_cols = st.columns(len(EXAMPLES))
    for i, example in enumerate(EXAMPLES):
        with example_cols[i]:
            if st.button(example["label"], key=f"example_{i}"):
                st.session_state.prompt_input = example["prompt"]
                st.session_state.last_prompt = example["prompt"]
                if example["images"]:
                    st.session_state.example_images = [
                        Image.open(p) for p in example["images"]
                    ]
                else:
                    st.session_state.pop("example_images", None)
                st.session_state.pop("enhanced_prompt", None)
                st.session_state.pop("enhanced_prompt_area", None)
                st.rerun()
```

Setting `prompt_input` makes the text input show the example prompt on rerun. Setting `last_prompt` to match prevents the change-detection block from firing and clearing the example state.

- [ ] **Step 3: Update image_list to include example images**

Replace lines 204-206:
```python
    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]
```
with:
```python
    image_list = None
    if uploaded_files:
        image_list = [Image.open(f) for f in uploaded_files]
        st.session_state.pop("example_images", None)
    elif "example_images" in st.session_state:
        image_list = st.session_state.example_images
```

- [ ] **Step 4: Add example image previews**

After the image_list logic and before the upload tracking, add a preview for example images:

```python
    if "example_images" in st.session_state and not uploaded_files:
        st.image(st.session_state.example_images, width=150)
```

- [ ] **Step 5: Update state clearing to also clear example state**

In the prompt change detection block (lines 226-229), add clearing of example state:

Replace:
```python
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("enhanced_prompt", None)
        st.session_state.pop("enhanced_prompt_area", None)
```
with:
```python
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("enhanced_prompt", None)
        st.session_state.pop("enhanced_prompt_area", None)
        st.session_state.pop("example_images", None)
```

Note: we do NOT clear `example_prompt` here because the prompt text input already shows the correct value via `st.session_state.prompt_input`. When the user types something new, `prompt` changes, `last_prompt` updates, and `example_images` is cleared. The text input naturally takes over.

- [ ] **Step 6: Run full test suite**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py -v`

Expected: ALL tests PASS (UI code is behind `if __name__ == "__main__"` guard).

- [ ] **Step 7: Lint and format**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add example buttons with session state management"
```

---

## Chunk 3: Documentation Updates

### Task 7: Update CLAUDE.md, README.md, and pyproject.toml

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `pyproject.toml:4`

- [ ] **Step 1: Update `pyproject.toml` description**

Replace line 4:
```
description = "Generate and edit images with FLUX.2 Klein (Distilled and Base) and optional prompt enhancement via SmolLM2"
```
with:
```
description = "Generate and edit images with FLUX.2 Klein (Distilled and Base) and optional prompt enhancement via SmolVLM"
```

- [ ] **Step 2: Update `CLAUDE.md` Project Overview**

Replace the overview (line 7) to mention SmolVLM instead of SmolLM2:

Replace:
```
FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images from text prompts using two variants of the [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (4B parameters) from Black Forest Labs via Hugging Face Diffusers: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for image editing workflows. Includes optional prompt upsampling using [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) to enhance prompts before generation.
```
with:
```
FLUX.2 Klein Pipeline is a single-file Streamlit web application that generates and edits images from text prompts using two variants of the [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model (4B parameters) from Black Forest Labs via Hugging Face Diffusers: Distilled (4 steps, fast) and Base (50 steps, higher quality). Supports multi-image input for image editing workflows. Includes optional vision-aware prompt upsampling using [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) to enhance prompts before generation — the VLM can see uploaded images when enhancing editing prompts. Includes pre-built example prompts with bundled images.
```

- [ ] **Step 3: Update `CLAUDE.md` Architecture section 3 (Prompt upsampling)**

Replace line 29:
```
3. **Prompt upsampling** — `_get_llm()` loads SmolLM2-1.7B-Instruct via `transformers.pipeline`, cached with `@st.cache_resource`. Two system prompts are defined: `UPSAMPLE_PROMPT_TEXT_ONLY` (for text-to-image generation, with guidelines for adding visual detail and quoting text in images, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (for image editing, with rules for concrete language, preserving unchanged elements, and turning negatives into positives). Both are adapted from the official BFL demo and simplified for SmolLM2-1.7B. `upsample_prompt()` accepts a `has_images` flag to select the appropriate system prompt, then returns the LLM-enhanced text. The LLM is loaded lazily on first use.
```
with:
```
3. **Prompt upsampling** — `_get_vlm()` loads SmolVLM-500M-Instruct via `AutoProcessor` + `AutoModelForVision2Seq`, cached with `@st.cache_resource`. Returns a `(processor, model)` tuple. Two system prompts are defined: `UPSAMPLE_PROMPT_TEXT_ONLY` (for text-to-image generation, with guidelines for adding visual detail and quoting text in images, capped at 120 words) and `UPSAMPLE_PROMPT_WITH_IMAGES` (for image editing, with rules for concrete language, preserving unchanged elements, and turning negatives into positives). Both are adapted from the official BFL demo. `upsample_prompt()` accepts an `image_list` parameter (optional list of PIL Images); when images are provided, they are passed to the VLM so it can see them when enhancing editing prompts. Messages use the multimodal list-of-dicts format required by SmolVLM's chat template. Output is extracted by slicing generated token IDs to exclude the input prompt. The VLM is loaded lazily on first use.
```

- [ ] **Step 4: Update `CLAUDE.md` Architecture section 4 (UI)**

Replace line 30:
```
4. **UI** — Streamlit interface behind `if __name__ == "__main__"` with text input, multi-image file uploader, enhance prompt button, mode radio (Distilled vs Base), run button, image output, and an expander with advanced settings. Width/height sliders auto-update to match the uploaded image's aspect ratio (tracked via session state); slider defaults for guidance scale and steps update automatically when the mode changes. Inference triggers on button click.
```
with:
```
4. **UI** — Streamlit interface behind `if __name__ == "__main__"` with text input, example buttons (4 text-only + 1 multi-image editing), multi-image file uploader, enhance prompt button, mode radio (Distilled vs Base), run button, image output, and an expander with advanced settings. Width/height sliders auto-update to match the uploaded image's aspect ratio (tracked via session state); slider defaults for guidance scale and steps update automatically when the mode changes. Example buttons populate the prompt (and images for the editing example) via session state; example state is cleared when the user modifies the prompt or uploads new files. Inference triggers on button click.
```

- [ ] **Step 5: Replace `CLAUDE.md` Transformers/SmolLM2 gotchas section**

Replace lines 55-59:
```
### Transformers / SmolLM2

- **SmolLM2-Instruct requires the chat message format.** Use `messages=[{"role": "system", ...}, {"role": "user", ...}]` with `transformers.pipeline`, not raw text. The response is structured as `[{"generated_text": [{"role": "assistant", "content": "..."}]}]`.
- **`transformers.pipeline` uses `dtype`, not `torch_dtype`.** The `transformers` library has deprecated `torch_dtype` in favor of `dtype`. This is the opposite of diffusers, which requires `torch_dtype`.
- **Use `GenerationConfig` instead of loose generation kwargs.** Passing `max_new_tokens`, `do_sample`, etc. as keyword arguments alongside a model's built-in `generation_config` is deprecated. Wrap them in a `GenerationConfig` object instead.
```
with:
```
### Transformers / SmolVLM

- **SmolVLM uses `AutoProcessor` + `AutoModelForVision2Seq`, not `transformers.pipeline`.** Load with `AutoProcessor.from_pretrained()` and `AutoModelForVision2Seq.from_pretrained()`. The processor handles both tokenization and image preprocessing.
- **`AutoModelForVision2Seq.from_pretrained` uses `torch_dtype`, like diffusers.** This is the standard `PreTrainedModel.from_pretrained` parameter, not the `transformers.pipeline` `dtype` parameter.
- **All message `content` must use list-of-dicts format.** SmolVLM's chat template requires `[{"type": "text", "text": "..."}]`, not plain strings. This applies to system messages and user messages on all paths (text-only and multimodal).
- **`batch_decode` returns the full sequence including the input prompt.** After `model.generate()`, slice the output to exclude input tokens before decoding: `output_ids[:, inputs["input_ids"].shape[1]:]`.
- **Pass sampling parameters directly to `model.generate()`.** Use `max_new_tokens`, `do_sample`, `temperature`, `top_p` as keyword arguments. No `GenerationConfig` wrapper needed.
```

- [ ] **Step 6: Update `CLAUDE.md` memory gotcha**

Replace line 65:
```
- **Both models and the LLM share memory.** FLUX.2 Klein Distilled (~8GB) + Base (~8GB) + SmolLM2-1.7B (~3.4GB) in bfloat16 require ~19.4GB peak. Both diffusion models are loaded lazily via `@st.cache_resource`. The LLM is loaded lazily on first "Enhance Prompt" use.
```
with:
```
- **Both models and the VLM share memory.** FLUX.2 Klein Distilled (~8GB) + Base (~8GB) + SmolVLM-500M (~1.2GB) in bfloat16 require ~17.2GB peak. Both diffusion models are loaded lazily via `@st.cache_resource`. The VLM is loaded lazily on first "Enhance Prompt" use.
```

- [ ] **Step 7: Update `README.md`**

Replace the full README with updated content referencing SmolVLM, examples, and updated memory:

Replace line 3:
```
Generate and edit images with [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) from Black Forest Labs. Includes optional prompt enhancement using [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
```
with:
```
Generate and edit images with [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) from Black Forest Labs. Includes vision-aware prompt enhancement using [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).
```

Replace lines 7-11:
```
- Text-to-image generation and image editing with FLUX.2 Klein (4B parameters)
- Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality)
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match uploaded image aspect ratio
- Prompt enhancement via SmolLM2-1.7B-Instruct with separate prompts for generation and editing (optional, loaded on first use)
```
with:
```
- Text-to-image generation and image editing with FLUX.2 Klein (4B parameters)
- Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality)
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match uploaded image aspect ratio
- Vision-aware prompt enhancement via SmolVLM-500M-Instruct — sees uploaded images when enhancing editing prompts (optional, loaded on first use)
- Pre-built example prompts with bundled images for quick start
```

Replace line 17:
```
- ~19.4GB RAM peak (both FLUX.2 Klein variants + SmolLM2 in bfloat16)
```
with:
```
- ~17.2GB RAM peak (both FLUX.2 Klein variants + SmolVLM in bfloat16)
```

Replace line 26:
```
Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~3.4GB for SmolLM2).
```
with:
```
Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~1.2GB for SmolVLM).
```

- [ ] **Step 8: Run full test suite**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run pytest tests/test_streamlit_app.py -v`

Expected: ALL tests PASS.

- [ ] **Step 9: Lint and format**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/flux.2-klein-pipeline" && uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md README.md pyproject.toml
git commit -m "docs: update CLAUDE.md, README, and pyproject.toml for VLM and examples"
```
