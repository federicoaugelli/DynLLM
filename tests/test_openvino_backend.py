from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynllm.backends.openvino import OpenVINOBackend, _probe_wav
from dynllm.core.config import BackendType, ModelConfig, ModelType


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    def __init__(self, statuses: list[int]) -> None:
        self._statuses = iter(statuses)

    async def post(self, *args, **kwargs):
        return _FakeResponse(next(self._statuses))


def _llm_model(tmp_path: Path, name: str = "gpt-oss") -> ModelConfig:
    model_dir = tmp_path / "OpenVINO" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelConfig(
        name=name,
        path=model_dir,
        backend=BackendType.openvino,
        vram_mb=4096,
        target_device="GPU",
        model_type=ModelType.llm,
    )


def test_llm_command_uses_task_approach(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path)

    cmd = backend._build_command(model, 9101, tmp_path)

    assert "--source_model" in cmd
    assert cmd[cmd.index("--source_model") + 1] == "gpt-oss"
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "text_generation"
    assert "--target_device" in cmd
    assert cmd[cmd.index("--target_device") + 1] == "GPU"
    assert "--model_repository_path" in cmd
    assert cmd[cmd.index("--model_repository_path") + 1] == str(model.path.parent)
    assert "--model_name" in cmd
    assert cmd[cmd.index("--model_name") + 1] == "gpt-oss"
    assert "--rest_port" in cmd
    assert "--port" in cmd
    assert "--config_path" not in cmd


def test_non_llm_command_uses_config_file(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model_dir = tmp_path / "models" / "bge"
    model_dir.mkdir(parents=True)
    model = ModelConfig(
        name="bge-embed",
        path=model_dir,
        backend=BackendType.openvino,
        vram_mb=1024,
        target_device="GPU",
        model_type=ModelType.embedding,
    )

    cmd = backend._build_command(model, 9101, tmp_path)

    assert "--config_path" in cmd
    assert "--task" not in cmd
    assert "--target_device" not in cmd

    config_path = Path(cmd[cmd.index("--config_path") + 1])
    config = json.loads(config_path.read_text())
    model_config = config["model_config_list"][0]["config"]

    assert model_config["name"] == "bge-embed"
    assert model_config["base_path"] == str(model.path)
    assert model_config["target_device"] == "GPU"


def test_validate_rejects_missing_model_path(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = ModelConfig(
        name="ghost",
        path=tmp_path / "does-not-exist",
        backend=BackendType.openvino,
        vram_mb=1,
    )
    with pytest.raises(RuntimeError, match="Model path not found"):
        backend._validate(model)


def test_speculative_decoding_flag(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path, "codellama-7b")
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    model.draft_model = draft_dir
    model.draft_model_vram_mb = 500

    cmd = backend._build_command(model, 9101, tmp_path)

    assert "--draft_source_model" in cmd
    assert cmd[cmd.index("--draft_source_model") + 1] == str(draft_dir)


def test_kv_cache_optimization_flags(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path, "test-model")
    model.kv_cache_precision = "u8"
    model.cache_size = 8
    model.enable_prefix_caching = True

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--kv_cache_precision") + 1] == "u8"
    assert cmd[cmd.index("--cache_size") + 1] == "8"
    assert cmd[cmd.index("--enable_prefix_caching") + 1] == "true"


def test_scheduling_flags(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path, "test-model")
    model.max_num_seqs = 128
    model.max_num_batched_tokens = 4096
    model.dynamic_split_fuse = False

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--max_num_seqs") + 1] == "128"
    assert cmd[cmd.index("--max_num_batched_tokens") + 1] == "4096"
    assert cmd[cmd.index("--dynamic_split_fuse") + 1] == "false"


def test_model_distribution_flag(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path, "test-model")
    model.model_distribution_policy = "TENSOR_PARALLEL"

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--model_distribution_policy") + 1] == "TENSOR_PARALLEL"


def test_all_optimization_flags_together(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = _llm_model(tmp_path, "full-opt")
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    model.draft_model = draft_dir
    model.draft_model_vram_mb = 500
    model.kv_cache_precision = "u8"
    model.cache_size = 8
    model.enable_prefix_caching = True
    model.max_num_seqs = 128
    model.max_num_batched_tokens = 4096
    model.dynamic_split_fuse = True
    model.model_distribution_policy = "PIPELINE_PARALLEL"
    model.tool_parser = "hermes3"
    model.reasoning_parser = "qwen3"
    model.enable_tool_guided_generation = True

    cmd = backend._build_command(model, 9101, tmp_path)

    assert cmd[cmd.index("--draft_source_model") + 1] == str(draft_dir)
    assert cmd[cmd.index("--kv_cache_precision") + 1] == "u8"
    assert cmd[cmd.index("--cache_size") + 1] == "8"
    assert cmd[cmd.index("--enable_prefix_caching") + 1] == "true"
    assert cmd[cmd.index("--max_num_seqs") + 1] == "128"
    assert cmd[cmd.index("--max_num_batched_tokens") + 1] == "4096"
    assert cmd[cmd.index("--dynamic_split_fuse") + 1] == "true"
    assert cmd[cmd.index("--model_distribution_policy") + 1] == "PIPELINE_PARALLEL"
    assert "--tool_parser" in cmd
    assert "--reasoning_parser" in cmd
    assert cmd[cmd.index("--enable_tool_guided_generation") + 1] == "true"


def test_non_llm_model_rejects_optimization_flags() -> None:
    with pytest.raises(ValueError, match="kv_cache_precision"):
        ModelConfig(
            name="bad-model",
            path=Path("/models/test"),
            backend=BackendType.openvino,
            vram_mb=1024,
            target_device="CPU",
            model_type=ModelType.embedding,
            kv_cache_precision="u8",
        )

    with pytest.raises(ValueError, match="draft_model"):
        ModelConfig(
            name="bad-model",
            path=Path("/models/test"),
            backend=BackendType.openvino,
            vram_mb=1024,
            target_device="CPU",
            model_type=ModelType.embedding,
            draft_model=Path("some/relative/path"),
        )


def test_probe_wav_is_one_second_silent_wav() -> None:
    wav = _probe_wav(duration_seconds=1.0)
    assert wav.startswith(b"RIFF")
    assert wav[8:12] == b"WAVE"
    assert b"fmt " in wav
    assert b"data" in wav
    # 1 second * 16000 Hz * 1 channel * 2 bytes + 44 byte header
    assert len(wav) == 16000 * 2 + 44


@pytest.mark.anyio
async def test_transcription_ready_interprets_status_codes() -> None:
    backend = OpenVINOBackend(binary="ovms")

    assert not await backend._transcription_ready(_FakeClient([503]), 9100, "whisper")
    assert not await backend._transcription_ready(_FakeClient([404]), 9100, "whisper")
    assert await backend._transcription_ready(_FakeClient([200]), 9100, "whisper")
    assert await backend._transcription_ready(_FakeClient([202]), 9100, "whisper")
    assert await backend._transcription_ready(_FakeClient([400]), 9100, "whisper")
    assert await backend._transcription_ready(_FakeClient([422]), 9100, "whisper")
