# Prompt Upsampling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt upsampling using a local SmolLM2-1.7B-Instruct model so users can enhance their prompts before image generation.

**Architecture:** Two new functions (`_get_llm()` and `upsample_prompt()`) added to `streamlit_app.py`. The LLM is loaded lazily via `@st.cache_resource` when the user first clicks "Enhance Prompt". An editable text area shows the enhanced prompt for review before generation.

**Tech Stack:** transformers (already installed), SmolLM2-1.7B-Instruct, Streamlit

**Spec:** `docs/superpowers/specs/2026-03-12-prompt-upsampling-design.md`

---

## Chunk 1: Test infrastructure and `_get_llm()`

### Task 1: Update `_reload_app` test helper

**Files:**
- Modify: `tests/test_streamlit_app.py:15-27`

- [ ] **Step 1: Update `_reload_app` to also patch `transformers.pipeline`**

Add `mock_llm` parameter and patch `transformers.pipeline` inside the existing context manager:

```python
def _reload_app(mock_pipe, *, mock_llm=None, mps_available=False, cuda_available=False):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch("diffusers.Flux2KleinPipeline") as mock_cls,
        patch("transformers.pipeline") as mock_tp,
        patch("torch.backends.mps.is_available", return_value=mps_available),
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("streamlit.cache_resource", lambda f: f),
    ):
        mock_cls.from_pretrained.return_value = mock_pipe
        if mock_llm is not None:
            mock_tp.return_value = mock_llm
        import streamlit_app

        importlib.reload(streamlit_app)
        return streamlit_app, mock_cls
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All existing tests PASS (the new `mock_llm` parameter defaults to `None`, so existing callers are unaffected)

- [ ] **Step 3: Commit**

```bash
git add tests/test_streamlit_app.py
git commit -m "test: update _reload_app to mock transformers.pipeline"
```

---

### Task 2: Add `_get_llm()` and its tests (TDD)

**Files:**
- Modify: `tests/test_streamlit_app.py`
- Modify: `streamlit_app.py:6` (add import), `streamlit_app.py:39` (add function after `_get_pipe`)

- [ ] **Step 1: Write failing tests for `_get_llm()`**

Add a new `_make_mock_llm` helper and `TestLLMInit` class to `tests/test_streamlit_app.py`:

```python
def _make_mock_llm():
    """Create a mock text-generation pipeline."""
    llm = MagicMock()
    llm.return_value = [
        {"generated_text": [{"role": "assistant", "content": "enhanced prompt"}]}
    ]
    return llm


class TestLLMInit:
    def test_llm_loads_correct_model(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            streamlit_app._get_llm()
            mock_tp.assert_called_once_with(
                "text-generation",
                model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                torch_dtype=torch.bfloat16,
                device="cpu",
            )

    def test_llm_device_mps(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        streamlit_app, _ = _reload_app(
            mock_pipe, mock_llm=mock_llm, mps_available=True
        )
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            streamlit_app._get_llm()
            mock_tp.assert_called_once_with(
                "text-generation",
                model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                torch_dtype=torch.bfloat16,
                device="mps",
            )

    def test_llm_device_cuda(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        streamlit_app, _ = _reload_app(
            mock_pipe, mock_llm=mock_llm, cuda_available=True
        )
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_tp.return_value = mock_llm
            streamlit_app._get_llm()
            mock_tp.assert_called_once_with(
                "text-generation",
                model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                torch_dtype=torch.bfloat16,
                device="cuda",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestLLMInit -v`
Expected: FAIL — `_get_llm` not defined / `transformers_pipeline` not found

- [ ] **Step 3: Implement `_get_llm()` in `streamlit_app.py`**

Add the import at the top of `streamlit_app.py` (after the `from dotenv import load_dotenv` line):

```python
from transformers import pipeline as transformers_pipeline
```

Add the function after `_get_pipe()` (after line 39):

```python
@st.cache_resource
def _get_llm():
    device, dtype = _detect_device()

    return transformers_pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        torch_dtype=dtype,
        device=device,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestLLMInit -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Add `@st.cache_resource` verification test**

Add to `TestStreamlitApp`:

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

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestStreamlitApp::test_get_llm_uses_cache_resource -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add _get_llm() for SmolLM2-1.7B-Instruct loading"
```

---

## Chunk 2: `upsample_prompt()` function

### Task 3: Add `upsample_prompt()` and its tests (TDD)

**Files:**
- Modify: `tests/test_streamlit_app.py`
- Modify: `streamlit_app.py` (add function after `_get_llm`)

- [ ] **Step 1: Write failing tests for `upsample_prompt()`**

Add `TestUpsamplePrompt` class to `tests/test_streamlit_app.py`:

```python
EXPECTED_SYSTEM_PROMPT = (
    "You are a prompt engineer. Rewrite the user's text into a detailed, "
    "vivid image generation prompt. Keep it under 100 words. Output only "
    "the enhanced prompt, nothing else."
)


class TestUpsamplePrompt:
    def test_chat_message_format(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            streamlit_app.upsample_prompt("a cat")
            mock_llm.assert_called_once()
            messages = mock_llm.call_args[0][0]
            assert messages[0] == {"role": "system", "content": EXPECTED_SYSTEM_PROMPT}
            assert messages[1] == {"role": "user", "content": "a cat"}

    def test_generation_kwargs(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            streamlit_app.upsample_prompt("a cat")
            kwargs = mock_llm.call_args[1]
            assert kwargs["max_new_tokens"] == 150
            assert kwargs["do_sample"] is True
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9

    def test_extracts_assistant_content(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        mock_llm.return_value = [
            {
                "generated_text": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "a cat"},
                    {"role": "assistant", "content": "  A majestic feline  "},
                ]
            }
        ]
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "A majestic feline"

    def test_empty_output_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        mock_llm.return_value = [
            {
                "generated_text": [
                    {"role": "assistant", "content": ""},
                ]
            }
        ]
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_whitespace_only_output_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        mock_llm.return_value = [
            {
                "generated_text": [
                    {"role": "assistant", "content": "   "},
                ]
            }
        ]
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"

    def test_exception_returns_original(self):
        mock_pipe = _make_mock_pipe()
        mock_llm = _make_mock_llm()
        mock_llm.side_effect = RuntimeError("OOM")
        streamlit_app, _ = _reload_app(mock_pipe, mock_llm=mock_llm)
        with (
            patch("streamlit_app.transformers_pipeline") as mock_tp,
            patch("streamlit_app.st") as mock_st,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_tp.return_value = mock_llm
            result = streamlit_app.upsample_prompt("a cat")
            assert result == "a cat"
            mock_st.warning.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestUpsamplePrompt -v`
Expected: FAIL — `upsample_prompt` not defined

- [ ] **Step 3: Implement `upsample_prompt()` in `streamlit_app.py`**

Add after `_get_llm()`:

```python
UPSAMPLE_SYSTEM_PROMPT = (
    "You are a prompt engineer. Rewrite the user's text into a detailed, "
    "vivid image generation prompt. Keep it under 100 words. Output only "
    "the enhanced prompt, nothing else."
)


def upsample_prompt(prompt):
    try:
        llm = _get_llm()
        messages = [
            {"role": "system", "content": UPSAMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        result = llm(
            messages,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        enhanced = result[0]["generated_text"][-1]["content"].strip()
        if not enhanced:
            return prompt
        return enhanced
    except Exception:
        st.warning("Prompt enhancement failed. Using original prompt.")
        return prompt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestUpsamplePrompt -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors. If there are issues, fix with `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add upsample_prompt() for prompt enhancement"
```

---

## Chunk 3: UI changes and documentation

### Task 4: Add "Enhance Prompt" UI to Streamlit

**Files:**
- Modify: `streamlit_app.py:70-143` (the `if __name__ == "__main__"` block)

- [ ] **Step 1: Add the Enhance Prompt button and enhanced prompt text area**

Replace the section between `prompt = st.text_input(...)` (line 77) and `with st.expander("Advanced Settings"):` (line 79) with:

```python
    prompt = st.text_input("Prompt", placeholder="Enter your prompt")

    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""

    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
        st.session_state.pop("enhanced_prompt", None)
        st.session_state.pop("enhanced_prompt_area", None)

    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing prompt..."):
            enhanced = upsample_prompt(prompt)
        st.session_state.enhanced_prompt = enhanced

    if "enhanced_prompt" in st.session_state:
        final_prompt = st.text_area(
            "Enhanced Prompt",
            value=st.session_state.enhanced_prompt,
            key="enhanced_prompt_area",
        )
    else:
        final_prompt = prompt
```

Then update the "Run" button block to use `final_prompt` instead of `prompt`:

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
            )
        st.session_state.result_image = image
        st.session_state.result_seed = used_seed if randomize_seed else None
```

- [ ] **Step 2: Run full test suite to verify nothing breaks**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS (UI code is behind `if __name__ == "__main__"` and not executed during tests)

- [ ] **Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors. If there are issues, fix with `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 4: Type check**

Run: `uv run ty check .`
Expected: No new errors

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Enhance Prompt button and editable text area"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Architecture section**

Replace the architecture list with all four items:

> 1. **Model initialization** — `_detect_device()` selects hardware (MPS > CUDA > CPU). All devices use bfloat16 to match the model's native dtype. `_get_pipe()` loads the pipeline once via `@st.cache_resource`. On CUDA, it uses `enable_model_cpu_offload()` to reduce VRAM usage; on MPS/CPU, it uses `pipe.to(device)`.
> 2. **Inference** — `infer()` takes prompt, seed, dimensions (512-1440px), guidance scale (default 1.0), and inference steps (default 4). FLUX.2 Klein does not support negative prompts. Runs under `torch.inference_mode()` with a CPU-pinned generator for MPS compatibility. Returns a PIL Image and the seed used.
> 3. **Prompt upsampling** — `_get_llm()` loads SmolLM2-1.7B-Instruct via `transformers.pipeline`, cached with `@st.cache_resource`. `upsample_prompt()` sends the user's prompt to the LLM using a chat message format and returns the enhanced text. The LLM is loaded lazily on first use.
> 4. **UI** — Streamlit interface behind `if __name__ == "__main__"` with text input, enhance prompt button, run button, image output, and an expander with advanced settings. Inference triggers on button click.

- [ ] **Step 2: Add new gotchas**

Add to the Gotchas section:

> - **SmolLM2-Instruct requires the chat message format.** Use `messages=[{"role": "system", ...}, {"role": "user", ...}]` with `transformers.pipeline`, not raw text. The response is structured as `[{"generated_text": [{"role": "assistant", "content": "..."}]}]`.
> - **Both models share memory.** FLUX.2 Klein (~8GB) and SmolLM2-1.7B (~3.4GB) in bfloat16 require ~11.4GB combined. The LLM is loaded lazily — it only consumes memory after the user clicks "Enhance Prompt".

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for prompt upsampling feature"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS

- [ ] **Step 2: Lint, format, and type check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check .`
Expected: No errors

- [ ] **Step 3: Review the full diff**

Run: `git diff main~N..HEAD` (where N is the number of commits in this feature)
Verify: All changes are intentional and match the spec

- [ ] **Step 4: Manual smoke test**

Run: `uv run streamlit run streamlit_app.py`
Verify:
1. "Enhance Prompt" button appears below the prompt input
2. Clicking "Enhance Prompt" with a prompt shows the enhanced text in a text area
3. The enhanced text area is editable
4. Changing the original prompt clears the enhanced text area
5. The "Run" button uses the enhanced prompt when available
6. The "Run" button uses the original prompt when no enhancement exists
