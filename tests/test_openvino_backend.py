from __future__ import annotations

import json
from pathlib import Path

from dynllm.backends.openvino import OpenVINOBackend
from dynllm.core.config import BackendType, ModelConfig


def test_llm_command_uses_config_file_for_target_device(tmp_path: Path) -> None:
    backend = OpenVINOBackend(binary="ovms")
    model = ModelConfig(
        name="Qwen2.5-Coder",
        path=Path("/models/qwen2.5-coder"),
        backend=BackendType.openvino,
        vram_mb=4096,
        target_device="GPU",
        ovms_shape="auto",
    )

    cmd = backend._build_llm_command(model, 9101, model.path, tmp_path)

    assert "--config_path" in cmd
    assert "--target_device" not in cmd

    config_path = Path(cmd[cmd.index("--config_path") + 1])
    config = json.loads(config_path.read_text())
    model_config = config["model_config_list"][0]["config"]

    assert model_config["name"] == "Qwen2.5-Coder"
    assert model_config["base_path"] == str(model.path)
    assert model_config["target_device"] == "GPU"
    assert model_config["shape"] == "auto"
