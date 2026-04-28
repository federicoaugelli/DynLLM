from __future__ import annotations

from pathlib import Path

import pytest

from dynllm.core.config import BackendConfig, BackendType, ModelConfig, ModelType


def test_transformers_model_fields_are_normalized() -> None:
    model = ModelConfig(
        name="qwen25-3b-hf",
        path=Path("/tmp/qwen25"),
        backend=BackendType.transformers,
        device=" XPU ",
        dtype=" BFLOAT16 ",
        vram_mb=1234,
    )

    assert model.model_type == ModelType.llm
    assert model.device == "xpu"
    assert model.dtype == "bfloat16"


def test_transformers_rejects_speech_model_type() -> None:
    with pytest.raises(
        ValueError,
        match="transformers supports model_type=llm and model_type=transcription",
    ):
        ModelConfig(
            name="bad-tts-hf",
            path=Path("/tmp/bad-tts-hf"),
            backend=BackendType.transformers,
            model_type=ModelType.speech,
            vram_mb=1,
        )


def test_backend_config_defaults_transformers_binary() -> None:
    backend = BackendConfig()
    assert backend.transformers_binary == "transformers"
