"""
llama.cpp backend.

Spawns one ``llama-server`` subprocess per loaded GGUF model.
llama-server exposes an OpenAI-compatible API at ``/v1/``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from dynllm.backends.base import Backend
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

# Time between readiness poll attempts
_POLL_INTERVAL = 0.5


class LlamaCppBackend(Backend):
    """Backend that manages llama-server subprocesses for GGUF models."""

    def __init__(self, binary: str = "llama-server") -> None:
        self._binary = self._resolve_binary(binary, "backend.llamacpp_binary")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.llamacpp

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    async def start(self, model: ModelConfig, port: int) -> int:
        """Launch llama-server for *model* and return its PID."""
        model_path = Path(model.path)
        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")

        self._binary = self._require_binary(
            self._binary,
            "llama-server binary not found. Build/install llama.cpp or set backend.llamacpp_binary to an absolute executable path.",
        )

        cmd = [
            self._binary,
            "--model",
            str(model_path),
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
            # Prevent llama-server from printing noisy startup info to stderr
            "--log-disable",
        ]

        logger.info(
            "Starting llama-server for model '%s' on port %d: %s",
            model.name,
            port,
            " ".join(cmd),
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        # Brief wait to detect immediate crashes
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
            # If we get here the process exited immediately
            _, stderr_bytes = await proc.communicate()
            err = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            raise RuntimeError(
                f"llama-server exited immediately for model '{model.name}': {err}"
            )
        except asyncio.TimeoutError:
            # Process is still running – good
            pass

        logger.debug("llama-server for '%s' started with PID %d", model.name, proc.pid)
        return proc.pid

    async def stop(self, pid: int) -> None:
        """Terminate the llama-server process with *pid*."""
        logger.info("Stopping llama-server PID %d", pid)
        await self._terminate_pid(pid)

    async def is_ready(
        self, port: int, model_name: str = "", timeout: float = 60.0, model_type: str = "llm"
    ) -> bool:
        """
        Poll ``GET http://127.0.0.1:<port>/health`` until llama-server
        reports healthy or the timeout expires.
        """
        url = f"http://127.0.0.1:{port}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        # llama-server returns {"status": "ok"} when ready
                        if data.get("status") in ("ok", "no slot available"):
                            logger.info("llama-server on port %d is ready", port)
                            return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning("llama-server on port %d did not become ready in time", port)
        return False
