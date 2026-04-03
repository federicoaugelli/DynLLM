"""
OpenVINO Model Server (OVMS) backend.

Spawns one ``ovms`` subprocess per loaded OpenVINO model directory.
OVMS exposes a REST API; inference is proxied via its OpenAI-compatible
``/v3/`` endpoints (introduced in OVMS 2023+).

Readiness is detected via the KServe Model Readiness endpoint:
  GET /v2/models/<model_name>/ready  →  200 when the model is loaded

OVMS is started with a single-model config generated on the fly so each
subprocess serves exactly one model, mirroring the llama-server approach.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import httpx

from dynllm.backends.base import Backend
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5


class OpenVINOBackend(Backend):
    """Backend that manages OVMS subprocesses for OpenVINO IR model directories."""

    def __init__(self, binary: str = "ovms") -> None:
        self._binary = binary
        # Keep a reference to temp dirs so they are not GC'd while the server runs
        self._temp_dirs: dict[int, tempfile.TemporaryDirectory] = {}  # pid -> tmpdir

    @property
    def backend_type(self) -> BackendType:
        return BackendType.openvino

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    async def start(self, model: ModelConfig, port: int) -> int:
        """
        Write an OVMS model config JSON to a temp directory and launch ovms.

        OVMS expects:
          --model_path   <directory containing the IR files>
          --model_name   <alias>
          --port         <gRPC port>   (we disable gRPC)
          --rest_port    <REST port>
        """
        model_path = Path(model.path)
        if not model_path.exists():
            raise RuntimeError(f"Model directory not found: {model_path}")

        # Build per-model config JSON
        config = {
            "model_config_list": [
                {
                    "config": {
                        "name": model.name,
                        "base_path": str(model_path),
                        **({"shape": model.ovms_shape} if model.ovms_shape else {}),
                    }
                }
            ]
        }

        tmpdir = tempfile.TemporaryDirectory(prefix=f"dynllm_ovms_{model.name}_")
        config_file = Path(tmpdir.name) / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        cmd = [
            self._binary,
            "--config_path",
            str(config_file),
            "--rest_port",
            str(port),
            # Disable gRPC to avoid port conflicts
            "--port",
            "0",
        ]

        logger.info(
            "Starting OVMS for model '%s' on REST port %d: %s",
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
            await asyncio.wait_for(proc.wait(), timeout=1.5)
            _, stderr_bytes = await proc.communicate()
            err = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            tmpdir.cleanup()
            raise RuntimeError(
                f"OVMS exited immediately for model '{model.name}': {err}"
            )
        except asyncio.TimeoutError:
            pass

        # Store temp dir – it must remain alive as long as the subprocess runs
        self._temp_dirs[proc.pid] = tmpdir

        logger.debug("OVMS for '%s' started with PID %d", model.name, proc.pid)
        return proc.pid

    async def stop(self, pid: int) -> None:
        """Terminate the OVMS process with *pid* and clean up its temp config."""
        logger.info("Stopping OVMS PID %d", pid)
        await self._terminate_pid(pid)

        # Clean up temp config directory
        tmpdir = self._temp_dirs.pop(pid, None)
        if tmpdir is not None:
            try:
                tmpdir.cleanup()
            except Exception as exc:
                logger.warning("Failed to clean up temp dir for PID %d: %s", pid, exc)

    async def is_ready(
        self, port: int, model_name: str = "", timeout: float = 120.0
    ) -> bool:
        """
        Poll ``GET /v2/models/<model_name>/ready`` until OVMS confirms the
        model is ready to serve requests, or the timeout expires.

        This is the KServe Model Readiness endpoint that OVMS exposes:
          - 200  → model is loaded and ready
          - 503  → model is still loading
          - 404  → model unknown (should not happen with a correctly configured name)

        Using this endpoint avoids any inference probe entirely, which
        previously caused false 404 responses due to a wrong model name in
        the probe body.

        OVMS can take a while to load large IR models so we use a generous
        default timeout of 120 s.
        """
        if not model_name:
            logger.warning(
                "OVMS is_ready() called without model_name; falling back to /v1/config"
            )
            ready_url = f"http://127.0.0.1:{port}/v1/config"
        else:
            ready_url = f"http://127.0.0.1:{port}/v2/models/{model_name}/ready"

        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(ready_url)
                    if resp.status_code == 200:
                        logger.info(
                            "OVMS on port %d: model '%s' is ready",
                            port,
                            model_name,
                        )
                        return True
                    logger.debug(
                        "OVMS port %d: readiness probe returned %d, retrying…",
                        port,
                        resp.status_code,
                    )
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning(
            "OVMS on port %d: model '%s' did not become ready within %.0fs",
            port,
            model_name,
            timeout,
        )
        return False
