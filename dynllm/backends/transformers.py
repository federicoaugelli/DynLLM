"""
Hugging Face ``transformers serve`` backend.

Spawns one ``transformers serve`` subprocess per loaded model.  DynLLM keeps
the public model alias in its own config and rewrites requests to the backend's
expected model identifier before proxying.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from dynllm.backends.base import SubprocessBackend
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5


class TransformersBackend(SubprocessBackend):
    """Backend that manages ``transformers serve`` subprocesses."""

    def __init__(self, binary: str = "transformers") -> None:
        super().__init__(
            binary=binary,
            install_hint=(
                "transformers binary not found. Install transformers[serving] "
                "in the DynLLM environment or set backend.transformers_binary "
                "to an absolute executable path."
            ),
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.transformers

    def _backend_model_name(self, model: ModelConfig) -> str:
        model_name = str(Path(model.path))
        if model.revision is not None:
            return f"{model_name}@{model.revision}"
        return model_name

    def _validate(self, model: ModelConfig) -> None:
        if not Path(model.path).exists():
            raise RuntimeError(f"Model path not found: {model.path}")

    def _build_command(self, model: ModelConfig, port: int, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "serve",
            self._backend_model_name(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            model.device,
            "--dtype",
            model.dtype,
        ]

        if model.quantization.value != "none":
            cmd.extend(["--quantization", model.quantization.value])
        if model.trust_remote_code:
            cmd.append("--trust-remote-code")
        if model.compile_model:
            cmd.append("--compile")
        if model.continuous_batching:
            cmd.append("--continuous-batching")
        if model.attn_implementation.value != "auto":
            cmd.extend(["--attn-implementation", model.attn_implementation.value])
        if model.model_timeout is not None:
            cmd.extend(["--model-timeout", str(model.model_timeout)])

        return cmd

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        health_url = f"http://127.0.0.1:{port}/health"
        load_url = f"http://127.0.0.1:{port}/load_model"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        break
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(_POLL_INTERVAL)
            else:
                logger.warning(
                    "transformers serve on port %d did not become healthy in time",
                    port,
                )
                return False

            remaining = max(1.0, deadline - asyncio.get_event_loop().time())
            try:
                async with client.stream(
                    "POST",
                    load_url,
                    json={"model": model.name},
                    timeout=httpx.Timeout(
                        connect=5.0, read=remaining, write=30.0, pool=5.0
                    ),
                ) as resp:
                    if resp.status_code >= 400:
                        logger.warning(
                            "transformers serve load probe failed on port %d "
                            "with status %d",
                            port,
                            resp.status_code,
                        )
                        return False
                    async for line in resp.aiter_lines():
                        event = line.strip().lower()
                        if event == "event: ready":
                            logger.info("transformers serve on port %d is ready", port)
                            return True
                        if event == "event: error":
                            logger.warning(
                                "transformers serve on port %d reported a model "
                                "load error",
                                port,
                            )
                            return False
            except httpx.TimeoutException:
                logger.warning(
                    "transformers serve load probe timed out on port %d", port
                )
                return False
            except (httpx.ConnectError, httpx.ReadError) as exc:
                logger.warning(
                    "transformers serve load probe failed on port %d: %s", port, exc
                )
                return False

        logger.warning(
            "transformers serve on port %d did not report ready status", port
        )
        return False
