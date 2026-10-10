from __future__ import annotations

from pathlib import Path

import pytest

from dynllm.backends.llamacpp import LlamaCppBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType


def _decision_model(tmp_path: Path, name: str = "clef") -> ModelConfig:
    model_file = tmp_path / f"{name}.gguf"
    model_file.write_bytes(b"GGUF")
    return ModelConfig(
        name=name,
        path=model_file,
        backend=BackendType.llamacpp,
        model_type=ModelType.decision,
        vram_mb=16000,
    )


def test_decision_command_has_no_pooling_flags(tmp_path: Path) -> None:
    backend = LlamaCppBackend(binary="llama-server")
    model = _decision_model(tmp_path)

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[0] == "llama-server"
    assert cmd[cmd.index("--model") + 1] == str(model.path)
    assert cmd[cmd.index("--alias") + 1] == "clef"
    assert "--embedding" not in cmd
    assert "--reranking" not in cmd
    assert "--mmproj" not in cmd


def test_mmproj_flag_emitted_when_set(tmp_path: Path) -> None:
    backend = LlamaCppBackend(binary="llama-server")
    model = _decision_model(tmp_path)
    mmproj = tmp_path / "Clef-mmproj-f16.gguf"
    mmproj.write_bytes(b"GGUF")
    model.mmproj = mmproj

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--mmproj") + 1] == str(mmproj)


def test_extra_args_appended_last(tmp_path: Path) -> None:
    backend = LlamaCppBackend(binary="llama-server")
    model = _decision_model(tmp_path)
    model.extra_args = ["--batch-size", "2048", "--flash-attn"]

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[-3:] == ["--batch-size", "2048", "--flash-attn"]


def test_validate_rejects_missing_model_file(tmp_path: Path) -> None:
    backend = LlamaCppBackend(binary="llama-server")
    model = _decision_model(tmp_path)
    model.path = tmp_path / "missing.gguf"

    with pytest.raises(RuntimeError, match="Model file not found"):
        backend._validate(model)


def test_validate_rejects_missing_mmproj(tmp_path: Path) -> None:
    backend = LlamaCppBackend(binary="llama-server")
    model = _decision_model(tmp_path)
    model.mmproj = tmp_path / "missing-mmproj.gguf"

    with pytest.raises(RuntimeError, match="Multimodal projector file not found"):
        backend._validate(model)


def test_decision_accepted_by_llamacpp_backend() -> None:
    model = ModelConfig(
        name="julia",
        path=Path("/models/Julia-1.gguf"),
        backend=BackendType.llamacpp,
        model_type=ModelType.decision,
        vram_mb=200,
    )
    assert model.model_type == ModelType.decision


def test_mmproj_and_extra_args_rejected_for_non_llamacpp() -> None:
    with pytest.raises(ValueError, match="mmproj"):
        ModelConfig(
            name="bad",
            path=Path("/models/ov"),
            backend=BackendType.openvino,
            model_type=ModelType.llm,
            vram_mb=1024,
            mmproj=Path("/models/proj.gguf"),
        )

    with pytest.raises(ValueError, match="extra_args"):
        ModelConfig(
            name="bad",
            path=Path("/models/ov"),
            backend=BackendType.openvino,
            model_type=ModelType.llm,
            vram_mb=1024,
            extra_args=["--batch-size", "2048"],
        )
