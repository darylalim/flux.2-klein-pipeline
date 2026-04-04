# FLUX.2 Klein Pipeline

Generate and edit images with [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) from Black Forest Labs. Runs natively on Apple Silicon via [mflux](https://github.com/filipstrand/mflux). Vision-aware prompt enhancement via [SmolVLM-500M-Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) powered by [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

## Features

- Text-to-image generation and image editing with FLUX.2 Klein (4B parameters)
- Two model variants: Distilled (4 steps, fast) and Base (50 steps, higher quality)
- Native Apple Silicon performance via MLX — no PyTorch required
- Multi-image upload for editing and compositing workflows
- Auto-dimension: width/height sliders adjust to match uploaded image aspect ratio
- Vision-aware prompt enhancement via SmolVLM-500M-Instruct (optional, loaded on first use)
- Auto-enhance prompt: checkbox in advanced settings to enhance before generation
- Per-step progress bar during inference
- Pre-built example prompts with bundled images
- Configurable seed, dimensions, guidance scale, and inference steps

## Requirements

- Apple Silicon Mac (M1+)
- Python 3.12+

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the application: `uv run streamlit run streamlit_app.py`

Models are downloaded automatically on first use (~8GB per FLUX.2 Klein variant, ~1GB for SmolVLM).

## Testing

Run the unit tests (no GPU or model download required):

```bash
uv run pytest
```
