"""
OpenVINO Model Server (OVMS) backend.

Spawns one ``ovms`` subprocess per loaded OpenVINO model directory.
OVMS exposes a REST API; we use its OpenAI-compatible endpoint under
``/v3/`` (introduced in OVMS 2023+) or the mediapipe-based ``/v1/`` path.

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

    async def is_ready(self, port: int, timeout: float = 120.0) -> bool:
        """
        Wait until OVMS is fully ready to serve inference requests.

        Two-phase check:
        1. Poll ``GET /v1/config`` until all model versions report
           ``"state": "AVAILABLE"`` – this confirms the model IR is loaded.
        2. Issue a minimal dummy ``POST /v3/chat/completions`` request to
           warm up the inference path.  OVMS can return 404 or 503 on the
           very first inference call even after ``/v1/config`` says AVAILABLE,
           so we retry until we get any response that is *not* 404/503 (an
           expected 400 "bad request" means the endpoint is live).

        OVMS can take a while to load large IR models so we use a generous
        default timeout of 120 s.
        """
        config_url = f"http://127.0.0.1:{port}/v1/config"
        infer_url = f"http://127.0.0.1:{port}/v3/chat/completions"
        deadline = asyncio.get_event_loop().time() + timeout

        # ------------------------------------------------------------------
        # Phase 1 – wait for /v1/config to report all models AVAILABLE
        # ------------------------------------------------------------------
        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(config_url)
                    if resp.status_code == 200:
                        data = resp.json()
                        all_ready = all(
                            any(
                                v.get("state") == "AVAILABLE" for v in versions.values()
                            )
                            for versions in data.values()
                        )
                        if all_ready:
                            logger.debug(
                                "OVMS port %d: /v1/config reports all models AVAILABLE",
                                port,
                            )
                            break
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)
            else:
                logger.warning("OVMS on port %d did not become ready in time", port)
                return False

        # ------------------------------------------------------------------
        # Phase 2 – probe the actual inference endpoint to ensure it is live.
        # Send a minimal (invalid) payload; any response other than 404/503
        # means the endpoint is accepting requests.
        # ------------------------------------------------------------------
        _PROBE_BODY = b'{"model":"probe","messages":[]}'
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.post(
                        infer_url,
                        content=_PROBE_BODY,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code not in (404, 503, 502):
                        logger.info(
                            "OVMS on port %d is ready (inference endpoint alive, "
                            "probe status %d)",
                            port,
                            resp.status_code,
                        )
                        return True
                    logger.debug(
                        "OVMS port %d: inference probe returned %d, retrying…",
                        port,
                        resp.status_code,
                    )
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning(
            "OVMS on port %d: inference endpoint did not become ready in time", port
        )
        return False
