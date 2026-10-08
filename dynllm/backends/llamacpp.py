"""
llama.cpp backend.

Spawns one ``llama-server`` subprocess per loaded GGUF model.  llama-server
exposes an OpenAI-compatible API at ``/v1/``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from dynllm.backends.base import SubprocessBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5


class LlamaCppBackend(SubprocessBackend):
    """Backend that manages llama-server subprocesses for GGUF models."""

    def __init__(self, binary: str = "llama-server") -> None:
        super().__init__(
            binary=binary,
            install_hint=(
                "llama-server binary not found. Build/install llama.cpp or set "
                "backend.llamacpp_binary to an absolute executable path."
            ),
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.llamacpp

    def _validate(self, model: ModelConfig) -> None:
        if not Path(model.path).exists():
            raise RuntimeError(f"Model file not found: {model.path}")

    def _build_command(self, model: ModelConfig, port: int, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "--model",
            str(model.path),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--n-gpu-layers",
            str(model.n_gpu_layers),
            "--ctx-size",
            str(model.context_size),
            "--alias",
            model.name,
            "--log-disable",
        ]

        if model.model_type == ModelType.embedding:
            cmd.extend(["--embedding", "--pooling", "mean"])
        elif model.model_type == ModelType.rerank:
            cmd.extend(["--reranking"])

        return cmd

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        # llama-server returns {"status": "ok"} when ready.
                        if resp.json().get("status") in ("ok", "no slot available"):
                            logger.info("llama-server on port %d is ready", port)
                            return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning("llama-server on port %d did not become ready in time", port)
        return False
