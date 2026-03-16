# FLUX.2 Klein Pipeline

Generate and edit images with [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) from Black Forest Labs. Vision-aware prompt enhancement via [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).

## Features

- Text-to-image generation and image editing with FLUX.2 Klein (4B parameters)
- Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality)
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match uploaded image aspect ratio
- Vision-aware prompt enhancement via SmolVLM-500M-Instruct (optional, loaded on first use)
- Auto-enhance prompt: checkbox in advanced settings to enhance before generation
- Per-step progress bar during inference
- Pre-built example prompts with bundled images
- Configurable seed, dimensions, guidance scale, and inference steps

## Requirements

- Python 3.12+
- ~17.2GB RAM peak (both FLUX.2 Klein variants + SmolVLM in bfloat16)

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. (Optional) Create a `.env` file with `HF_TOKEN=<your-token>` for authenticated Hugging Face access
4. Run the application: `uv run streamlit run streamlit_app.py`

Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~1.2GB for SmolVLM).

## Testing

Run the unit tests (no GPU or model download required):

```bash
uv run pytest
```
