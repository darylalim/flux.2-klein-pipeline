# Text to Image Pipeline

Generate images from text prompts with the [FLUX.2 Klein (4B)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model from Black Forest Labs.

![A capybara wearing a suit holding a sign that reads Hello World](images/example.webp)

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Install dependencies: `uv sync`
3. Run the application: `uv run python app.py`

## Testing

Run the unit tests (no GPU or model download required):

```bash
uv run pytest
```
