import importlib
from unittest.mock import ANY, MagicMock, patch

import torch
from PIL import Image


def _make_mock_pipe():
    """Create a mock diffusion pipeline that returns a dummy image."""
    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (64, 64))]
    return pipe


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


def _reload_app(mock_pipe, *, mock_vlm=None, mps_available=False, cuda_available=False):
    """Reload app module with mocked heavy dependencies and passthrough cache."""
    with (
        patch("diffusers.Flux2KleinPipeline") as mock_cls,
        patch("transformers.AutoProcessor") as mock_ap,
        patch("transformers.AutoModelForImageTextToText") as mock_vm,
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


class TestConstants:
    def test_max_seed(self):
        import streamlit_app

        assert streamlit_app.MAX_SEED == 2_147_483_647

    def test_max_image_size(self):
        import streamlit_app

        assert streamlit_app.MAX_IMAGE_SIZE == 1440

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

    def test_pipes_maps_to_getters(self):
        import streamlit_app

        assert (
            streamlit_app.PIPES["Distilled (4 steps)"]
            is streamlit_app._get_pipe_distilled
        )
        assert streamlit_app.PIPES["Base (50 steps)"] is streamlit_app._get_pipe_base


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

    def test_distilled_pipeline_moved_to_cpu(self):
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

    def test_distilled_pipeline_moved_to_mps(self):
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

    def test_distilled_cpu_offload_on_cuda(self):
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

    def test_base_pipeline_moved_to_cpu(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_base()
            mock_pipe.to.assert_called_with("cpu")

    def test_base_pipeline_moved_to_mps(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_base()
            mock_pipe.to.assert_called_with("mps")

    def test_base_cpu_offload_on_cuda(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            streamlit_app._get_pipe_base()
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

    def test_progress_callback_passed_to_pipeline(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer("a cat", progress_callback=callback)
            call_kwargs = mock_pipe.call_args[1]
            assert "callback_on_step_end" in call_kwargs
            assert callable(call_kwargs["callback_on_step_end"])

    def test_no_callback_when_progress_callback_none(self):
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
            assert "callback_on_step_end" not in call_kwargs

    def test_progress_callback_invoked_with_step_and_total(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            on_step_end = mock_pipe.call_args[1]["callback_on_step_end"]
            result = on_step_end(mock_pipe, 0, 999, {"latents": None})
            callback.assert_called_once_with(1, 4)
            assert isinstance(result, dict)

    def test_progress_callback_returns_callback_kwargs(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer("a cat", progress_callback=callback)
            on_step_end = mock_pipe.call_args[1]["callback_on_step_end"]
            test_kwargs = {"latents": "test_tensor"}
            result = on_step_end(mock_pipe, 2, 500, test_kwargs)
            assert result is test_kwargs

    def test_progress_callback_with_base_mode(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", mode="Base (50 steps)", progress_callback=callback
            )
            on_step_end = mock_pipe.call_args[1]["callback_on_step_end"]
            on_step_end(mock_pipe, 0, 999, {"latents": None})
            callback.assert_called_once_with(1, 50)

    def test_progress_callback_step_counts_across_steps(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=4, progress_callback=callback
            )
            on_step_end = mock_pipe.call_args[1]["callback_on_step_end"]
            for step in range(4):
                on_step_end(mock_pipe, step, 999 - step, {"latents": None})
            assert callback.call_count == 4
            callback.assert_any_call(1, 4)
            callback.assert_any_call(2, 4)
            callback.assert_any_call(3, 4)
            callback.assert_any_call(4, 4)

    def test_progress_callback_with_image_list(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (64, 64))]
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer(
                "edit this", image_list=images, progress_callback=callback
            )
            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["image"] is images
            assert "callback_on_step_end" in call_kwargs

    def test_progress_callback_with_explicit_steps(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        with (
            patch("streamlit_app.Flux2KleinPipeline") as mock_cls,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_cls.from_pretrained.return_value = mock_pipe
            callback = MagicMock()
            streamlit_app.infer(
                "a cat", num_inference_steps=10, progress_callback=callback
            )
            on_step_end = mock_pipe.call_args[1]["callback_on_step_end"]
            on_step_end(mock_pipe, 9, 100, {"latents": None})
            callback.assert_called_once_with(10, 10)


class TestDimensionsFromImages:
    def test_square_image(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (800, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_landscape_image(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (1600, 800))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_portrait_image(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024

    def test_rounds_to_multiple_of_32(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (1000, 700))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w % 32 == 0
        assert h % 32 == 0

    def test_clamps_min_to_512(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (3000, 500))]
        _, h = streamlit_app._dimensions_from_images(images)
        assert h >= 512

    def test_zero_height_returns_default(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (100, 0))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_zero_width_returns_default(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (0, 100))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 1024

    def test_uses_first_image_only(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (1600, 800)), Image.new("RGB", (800, 1600))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_4_3_aspect_ratio(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (1200, 900))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 768

    def test_16_9_aspect_ratio(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (1920, 1080))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 576

    def test_portrait_3_4_aspect_ratio(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (900, 1200))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 768
        assert h == 1024

    def test_extreme_panoramic_clamps_height(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (5000, 500))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 1024
        assert h == 512

    def test_extreme_tall_clamps_width(self):
        mock_pipe = _make_mock_pipe()
        streamlit_app, _ = _reload_app(mock_pipe)
        images = [Image.new("RGB", (500, 5000))]
        w, h = streamlit_app._dimensions_from_images(images)
        assert w == 512
        assert h == 1024


class TestVLMInit:
    def test_vlm_loads_correct_model(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_processor, mock_model = mock_vlm
            mock_ap.from_pretrained.return_value = mock_processor
            mock_vm.from_pretrained.return_value = mock_model
            streamlit_app._get_vlm()
            mock_model.to.assert_called_with("cuda")


EXPECTED_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. "
    "Rewrite user prompts to be more descriptive while strictly preserving "
    "their core subject and intent. Keep the enhanced prompt under 120 "
    "words.\n\n"
    "Guidelines:\n"
    "- Add concrete visual specifics: textures, materials, lighting, "
    "shadows, and spatial relationships.\n"
    "- Put ALL text that should appear in the image in quotation marks "
    "(signs, labels, screens, etc.) - without quotes, the model generates "
    "gibberish.\n\n"
    "Output only the revised prompt and nothing else."
)

EXPECTED_SYSTEM_PROMPT_WITH_IMAGES = (
    "You are an image-editing expert. Convert the user's editing request "
    "into one concise instruction (50-80 words, ~30 for brief requests).\n\n"
    "Rules:\n"
    "- Single instruction only, no commentary\n"
    "- Use clear, analytical language (avoid vague words like "
    '"whimsical" or "cascading")\n'
    "- Specify what changes AND what stays the same (face, lighting, "
    "composition)\n"
    "- Turn negatives into positives "
    '("don\'t change X" becomes "keep X")\n'
    '- Make abstractions concrete ("futuristic" becomes '
    '"glowing cyan neon, metallic panels")\n\n'
    "Output only the final instruction in plain text and nothing else."
)


class TestUpsamplePrompt:
    def test_chat_message_format_text_only(self):
        mock_pipe = _make_mock_pipe()
        mock_vlm = _make_mock_vlm()
        streamlit_app, _ = _reload_app(mock_pipe, mock_vlm=mock_vlm)
        with (
            patch("streamlit_app.AutoProcessor") as mock_ap,
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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
            patch("streamlit_app.AutoModelForImageTextToText") as mock_vm,
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


class TestStreamlitApp:
    def test_get_pipe_distilled_uses_cache_resource(self):
        """Verify _get_pipe_distilled is decorated with @st.cache_resource."""
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForImageTextToText"),
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
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForImageTextToText"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_pipe_base, "clear")

    def test_get_vlm_uses_cache_resource(self):
        """Verify _get_vlm is decorated with @st.cache_resource (not passthrough)."""
        with (
            patch("diffusers.Flux2KleinPipeline"),
            patch("transformers.AutoProcessor"),
            patch("transformers.AutoModelForImageTextToText"),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            import streamlit_app

            importlib.reload(streamlit_app)
            assert hasattr(streamlit_app._get_vlm, "clear")

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
