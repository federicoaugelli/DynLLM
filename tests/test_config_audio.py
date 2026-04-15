from __future__ import annotations

from pathlib import Path

import pytest

from dynllm.core.config import BackendType, ModelConfig, ModelType


def test_model_config_defaults_audio_fields() -> None:
    model = ModelConfig(
        name="phi3-ov",
        path=Path("/tmp/phi3-ov"),
        backend=BackendType.openvino,
        vram_mb=0,
    )

    assert model.model_type == ModelType.llm
    assert model.target_device == "CPU"
    assert model.vram_mb == 0


def test_target_device_is_normalized() -> None:
    model = ModelConfig(
        name="whisper-ov",
        path=Path("/tmp/whisper-ov"),
        backend=BackendType.openvino,
        model_type=ModelType.transcription,
        target_device=" gpu ",
        vram_mb=0,
    )

    assert model.target_device == "GPU"


def test_llamacpp_rejects_non_llm_model_type() -> None:
    with pytest.raises(ValueError, match="llamacpp only supports model_type=llm"):
        ModelConfig(
            name="bad-whisper",
            path=Path("/tmp/bad-whisper.gguf"),
            backend=BackendType.llamacpp,
            model_type=ModelType.transcription,
            vram_mb=1,
        )
