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
    with pytest.raises(ValueError, match="llamacpp supports model_type=llm, embedding, and rerank"):
        ModelConfig(
            name="bad-whisper",
            path=Path("/tmp/bad-whisper.gguf"),
            backend=BackendType.llamacpp,
            model_type=ModelType.transcription,
            vram_mb=1,
        )


def test_openvino_rejects_speech_model_type() -> None:
    with pytest.raises(ValueError, match="openvino no longer supports model_type=speech"):
        ModelConfig(
            name="bad-speech-ov",
            path=Path("/tmp/bad-speech"),
            backend=BackendType.openvino,
            model_type=ModelType.speech,
            vram_mb=1,
        )


def test_tts_requires_engine_field() -> None:
    with pytest.raises(ValueError, match="tts backend requires 'tts_engine' field"):
        ModelConfig(
            name="bad-tts",
            path=Path("/tmp/bad-tts"),
            backend=BackendType.tts,
            model_type=ModelType.speech,
            vram_mb=1,
        )


def test_tts_rejects_non_speech_model_type() -> None:
    with pytest.raises(ValueError, match="tts backend only supports model_type=speech"):
        ModelConfig(
            name="bad-llm-tts",
            path=Path("/tmp/bad-llm-tts"),
            backend=BackendType.tts,
            model_type=ModelType.llm,
            tts_engine="qwen",
            vram_mb=1,
        )


def test_tts_engine_outside_tts_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="tts_engine is only valid for backend=tts"):
        ModelConfig(
            name="bad-engine",
            path=Path("/tmp/bad-engine"),
            backend=BackendType.openvino,
            tts_engine="qwen",
            vram_mb=1,
        )
