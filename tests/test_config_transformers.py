from __future__ import annotations

from pathlib import Path

import pytest

from dynllm.core.config import (
    BackendConfig,
    BackendType,
    ModelConfig,
    ModelType,
    TransformersAttentionImplementation,
    TransformersQuantization,
)


def test_transformers_model_fields_are_normalized() -> None:
    model = ModelConfig(
        name="qwen25-3b-hf",
        path=Path("/tmp/qwen25"),
        backend=BackendType.transformers,
        device=" XPU ",
        dtype=" BFLOAT16 ",
        revision=" main ",
        vram_mb=1234,
    )

    assert model.model_type == ModelType.llm
    assert model.device == "xpu"
    assert model.dtype == "bfloat16"
    assert model.revision == "main"
    assert model.quantization == TransformersQuantization.none
    assert model.attn_implementation == TransformersAttentionImplementation.auto


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


def test_transformers_quantization_rejects_speech_model_type() -> None:
    with pytest.raises(
        ValueError,
        match="transformers quantization is currently supported only for model_type=llm",
    ):
        ModelConfig(
            name="bad-whisper-quant",
            path=Path("/tmp/bad-whisper"),
            backend=BackendType.transformers,
            model_type=ModelType.transcription,
            quantization="bnb-8bit",
            vram_mb=1,
        )


def test_transformers_quantization_rejects_float32() -> None:
    with pytest.raises(
        ValueError,
        match="transformers quantization requires dtype to be auto, float16, or bfloat16",
    ):
        ModelConfig(
            name="bad-qwen-quant",
            path=Path("/tmp/bad-qwen"),
            backend=BackendType.transformers,
            quantization="bnb-4bit",
            dtype="float32",
            vram_mb=1,
        )


def test_transformers_revision_rejects_embedded_at_symbol() -> None:
    with pytest.raises(
        ValueError,
        match="revision must not contain '@'; configure it separately",
    ):
        ModelConfig(
            name="bad-qwen-revision",
            path=Path("/tmp/bad-qwen"),
            backend=BackendType.transformers,
            revision="repo@main",
            vram_mb=1,
        )
