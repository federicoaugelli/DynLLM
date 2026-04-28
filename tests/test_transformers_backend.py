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


def test_transformers_command_includes_quantization_and_optional_runtime_flags() -> None:
    backend = TransformersBackend(binary="transformers")
    model = ModelConfig(
        name="qwen25-3b-bnb4",
        path=Path("/models/Qwen2.5-3B-Instruct"),
        backend=BackendType.transformers,
        model_type=ModelType.llm,
        device="cuda",
        dtype="float16",
        quantization="bnb-4bit",
        trust_remote_code=True,
        compile_model=True,
        continuous_batching=True,
        attn_implementation="sdpa",
        model_timeout=120,
        revision="main",
        vram_mb=4200,
    )

    cmd = backend._build_command(model, 9102)

    assert cmd == [
        "transformers",
        "serve",
        "/models/Qwen2.5-3B-Instruct@main",
        "--host",
        "127.0.0.1",
        "--port",
        "9102",
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--quantization",
        "bnb-4bit",
        "--trust-remote-code",
        "--compile",
        "--continuous-batching",
        "--attn-implementation",
        "sdpa",
        "--model-timeout",
        "120",
    ]
