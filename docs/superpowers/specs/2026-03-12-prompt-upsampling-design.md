# Prompt Upsampling Design

## Overview

Add a prompt upsampling feature to the Streamlit app that uses a local LLM to enhance user prompts before image generation. The enhanced prompt is shown in an editable text area for the user to review and modify before generating.

## Model

- **Model**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Loading**: `transformers.pipeline("text-generation", ...)` cached with `@st.cache_resource`
- **Device**: Reuses `_detect_device()` — same device and dtype as the image model
- **No new dependencies**: `transformers` is already installed
- **Lazy loading**: The LLM is only loaded when the user first clicks "Enhance Prompt", not at app startup

## Architecture

Two new components in `streamlit_app.py` (single-file architecture preserved):

### `_get_llm()`

Cached LLM loader using `@st.cache_resource`. Loads SmolLM2-1.7B-Instruct via `transformers.pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=dtype, device=device)`. Device and dtype come from `_detect_device()`.

On CUDA, the LLM uses `device="cuda"` directly (no CPU offload equivalent for `transformers.pipeline`). Combined memory with the image model (~8GB image + ~3.4GB LLM in bfloat16) requires ~11.4GB. This fits consumer GPUs with 12GB+ VRAM. On unified memory systems (MPS), both models share the same pool.

### `upsample_prompt(prompt: str) -> str`

Takes the user's short prompt, sends it to the LLM using the chat message format, and returns the enhanced text.

- **Message format** (chat template required for instruct models):
  ```python
  messages = [
      {"role": "system", "content": "You are a prompt engineer. ..."},
      {"role": "user", "content": prompt},
  ]
  ```
- **System prompt**: "You are a prompt engineer. Rewrite the user's text into a detailed, vivid image generation prompt. Keep it under 100 words. Output only the enhanced prompt, nothing else."
- **Generation config**: `max_new_tokens=150`, `do_sample=True`, `temperature=0.7`, `top_p=0.9`
- **Output parsing**: The chat-mode pipeline returns `[{"generated_text": [{"role": "assistant", "content": "..."}]}]`. Extract `result[0]["generated_text"][-1]["content"]` and strip whitespace.
- **Error handling**: If the LLM returns empty content, return the original prompt unchanged.

## UI Flow

1. **Prompt input** — existing `st.text_input` unchanged
2. **"Enhance Prompt" button** — new button below the prompt input. Shows `st.spinner("Enhancing prompt...")` while running. Calls `upsample_prompt()` and stores the result in `st.session_state.enhanced_prompt`
3. **Enhanced prompt area** — only appears when `st.session_state.enhanced_prompt` is set. Rendered as `st.text_area` with a `key="enhanced_prompt_area"` so Streamlit manages user edits independently of session state
4. **"Run" button** — generates the image. Uses the text area value if the enhanced area is visible, otherwise the original input
5. **Advanced Settings** — unchanged (seed, dimensions, guidance scale, steps)

**State management:**
- When the user changes the original prompt input, clear `st.session_state.enhanced_prompt` so stale enhanced text doesn't persist
- Clicking "Enhance Prompt" again overwrites the previous enhancement
- The `st.text_area` with its own `key` lets Streamlit track user edits without being overwritten on rerun

## Testing

Following existing mock-based test patterns. The `_reload_app` helper will need to also patch `transformers.pipeline` to prevent model downloads during testing.

### `TestLLMInit`

- Verify `_get_llm()` loads `HuggingFaceTB/SmolLM2-1.7B-Instruct` with correct args (model name, `torch_dtype`, `device`)
- Verify it uses `@st.cache_resource` (has `.clear` attribute)
- Verify correct device placement (MPS/CUDA/CPU)

### `TestUpsamplePrompt`

- Verify chat message format is passed correctly (system + user messages)
- Verify generation kwargs are forwarded (`max_new_tokens`, `do_sample`, `temperature`, `top_p`)
- Verify output parsing extracts assistant content from structured response
- Verify empty LLM output falls back to the original prompt
- Verify the returned text is stripped of whitespace

### `TestStreamlitApp` updates

- Verify `_get_llm` uses `@st.cache_resource` (has `.clear` attribute)
- Verify UI doesn't call LLM functions on import

## CLAUDE.md Updates

Update the architecture section to describe the new LLM components and add gotchas:
- Chat template is required for SmolLM2-Instruct (use message format, not raw text)
- Memory considerations when co-loading both models
