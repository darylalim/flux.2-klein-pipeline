import importlib
from unittest.mock import ANY, MagicMock, patch

import torch
from PIL import Image


def _make_mock_llm():
    """Create a mock text-generation pipeline."""
    llm = MagicMock()
    llm.return_value = [
        {"generated_text": [{"role": "assistant", "content": "enhanced prompt"}]}
    ]
    return llm


def _make_mock_pipe():
    """Create a mock diffusion pipeline that returns a dummy image."""
    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (64, 64))]
    return pipe


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


class TestConstants:
    def test_max_seed(self):
        import streamlit_app

        assert streamlit_app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import streamlit_app

        assert streamlit_app.MAX_IMAGE_SIZE == 1440


class TestDetectDevice:
    def test_mps_when_available(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe, mps_available=True)
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device, dtype = streamlit_app._detect_device()
            assert device == "mps"
            assert dtype is torch.bfloat16

    def test_cuda_when_mps_unavailable(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe, cuda_available=True)
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device, dtype = streamlit_app._detect_device()
            assert device == "cuda"
            assert dtype is torch.bfloat16

    def test_cpu_fallback(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device, dtype = streamlit_app._detect_device()
            assert device == "cpu"
            assert dtype is torch.bfloat16

    def test_mps_priority_over_cuda(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(
            mock_pipe, mps_available=True, cuda_available=True
        )
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device, _ = streamlit_app._detect_device()
            assert device == "mps"


class TestPipelineInit:
    def test_from_pretrained_args(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, mock_cls = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls2,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls2.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe()
            mock_cls2.from_pretrained.assert_called_once_with(
                "black-forest-labs/FLUX.2-klein-4B",
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
            streamlit_app._get_pipe()
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
            streamlit_app._get_pipe()
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
            streamlit_app._get_pipe()
            mock_pipe.enable_model_cpu_offload.assert_called()
            mock_pipe.to.assert_not_called()


class TestInfer:
    def test_returns_image_and_seed(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            image, seed = streamlit_app.infer("a cat", seed=42)
            assert isinstance(image, Image.Image)
            assert seed == 42

    def test_forwards_args_to_pipeline(self):
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
                seed=123,
                width=768,
                height=512,
                guidance_scale=3.0,
                num_inference_steps=20,
            )
            mock_pipe.assert_called_once_with(
                prompt="a cat",
                guidance_scale=3.0,
                num_inference_steps=20,
                width=768,
                height=512,
                generator=ANY,
            )

    def test_fixed_seed(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = streamlit_app.infer("a cat", seed=99, randomize_seed=False)
            assert seed == 99

    def test_randomized_seed(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            _, seed = streamlit_app.infer("a cat", seed=42, randomize_seed=True)
            assert 0 <= seed <= streamlit_app.MAX_SEED

    def test_generator_uses_cpu(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.Generator") as mock_gen_cls,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_gen = MagicMock()
            mock_gen.manual_seed.return_value = mock_gen
            mock_gen_cls.return_value = mock_gen
            streamlit_app.infer("a cat", seed=42)
            mock_gen_cls.assert_called_once_with(device="cpu")
            mock_gen.manual_seed.assert_called_once_with(42)

    def test_default_params(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app.infer("a cat")
            mock_pipe.assert_called_once_with(
                prompt="a cat",
                guidance_scale=1.0,
                num_inference_steps=4,
                width=1024,
                height=1024,
                generator=ANY,
            )

    def test_empty_prompt(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            image, seed = streamlit_app.infer("", seed=42)
            assert isinstance(image, Image.Image)
            mock_pipe.assert_called_once_with(
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=4,
                width=1024,
                height=1024,
                generator=ANY,
            )

    def test_uses_inference_mode(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.inference_mode") as mock_inference_mode,
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_cm = MagicMock()
            mock_inference_mode.return_value = mock_cm
            streamlit_app.infer("a cat", seed=42)
            mock_inference_mode.assert_called_once()
            mock_cm.__enter__.assert_called_once()


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


class TestStreamlitApp:
    def test_get_pipe_uses_cache_resource(self):
        """Verify _get_pipe is decorated with @st.cache_resource (not passthrough)."""
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_pipe, "clear")

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

    def test_no_pipe_global(self):
        import streamlit_app

        assert not hasattr(streamlit_app, "_pipe")

    def test_ui_not_executed_on_import(self):
        mock_pipe = _make_mock_pipe()
        with (
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.text_input") as mock_text_input,
            patch("streamlit.button") as mock_button,
        ):
            _reload_app(mock_pipe)
            mock_markdown.assert_not_called()
            mock_text_input.assert_not_called()
            mock_button.assert_not_called()
