from __future__ import annotations

import json
from pathlib import Path

from dynllm.backends.openvino import OpenVINOBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType


def test_llm_command_uses_task_approach(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model_dir = tmp_path / "OpenVINO" / "gpt-oss"
    model_dir.mkdir(parents=True)
    model = ModelConfig(
        name="gpt-oss",
        path=model_dir,
        backend=BackendType.openvino,
        vram_mb=4096,
        target_device="GPU",
        model_type=ModelType.llm,
    )

    cmd = backend._build_command(model, 9101, model.path, tmp_path)

    assert "--source_model" in cmd
    assert cmd[cmd.index("--source_model") + 1] == "gpt-oss"
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "text_generation"
    assert "--target_device" in cmd
    assert cmd[cmd.index("--target_device") + 1] == "GPU"
    assert "--model_repository_path" in cmd
    assert cmd[cmd.index("--model_repository_path") + 1] == str(model_dir.parent)
    assert "--model_name" in cmd
    assert cmd[cmd.index("--model_name") + 1] == "gpt-oss"
    assert "--rest_port" in cmd
    assert "--port" in cmd
    assert "--config_path" not in cmd


def test_non_llm_command_uses_config_file(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = ModelConfig(
        name="bge-embed",
        path=Path("/models/bge"),
        backend=BackendType.openvino,
        vram_mb=1024,
        target_device="GPU",
        model_type=ModelType.embedding,
    )

    cmd = backend._build_command(model, 9101, model.path, tmp_path)

    assert "--config_path" in cmd
    assert "--task" not in cmd
    assert "--target_device" not in cmd

    config_path = Path(cmd[cmd.index("--config_path") + 1])
    config = json.loads(config_path.read_text())
    model_config = config["model_config_list"][0]["config"]

    assert model_config["name"] == "bge-embed"
    assert model_config["base_path"] == str(model.path)
    assert model_config["target_device"] == "GPU"
