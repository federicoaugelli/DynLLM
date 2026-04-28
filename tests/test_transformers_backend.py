from __future__ import annotations

from pathlib import Path

from dynllm.backends.transformers import TransformersBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType


def test_transformers_command_uses_local_model_path_and_runtime_flags() -> None:
    backend = TransformersBackend(binary="transformers")
    model = ModelConfig(
        name="qwen25-3b-hf",
        path=Path("/models/Qwen2.5-3B-Instruct"),
        backend=BackendType.transformers,
        model_type=ModelType.llm,
        device="xpu",
        dtype="bfloat16",
        vram_mb=6500,
    )

    cmd = backend._build_command(model, 9101)

    assert cmd == [
        "transformers",
        "serve",
        "/models/Qwen2.5-3B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "9101",
        "--device",
        "xpu",
        "--dtype",
        "bfloat16",
    ]
