"""
NeutronStar (``ns-server``) backend.

NeutronStar is a custom MoE inference engine for Intel Arc GPUs.  ``ns-server``
serves exactly one merged GGUF model per process with a minimal OpenAI-
compatible API::

    ns-server --model <dir|model.json|merged.gguf> --host 127.0.0.1 \\
              --port <port> --alias <name> [--max-seq N]

    GET  /health              -> {"status":"ok"}
    GET  /v1/models           -> the single ``--alias`` entry
    POST /v1/chat/completions -> OpenAI chat completions (SSE + JSON)

Notable engine behaviours a proxy must respect:

* The TCP listener is opened only **after** the GGUF has been mmapped and all
  IRs compiled/staged (≈40–70 s cold, faster from the on-disk IR cache), so
  readiness is "port answers ``/health``" and needs a generous budget.
* Single worker, requests serialize; treat one loaded ns model as occupying
  (almost) the whole GPU — ``vram_mb`` should be set high enough that VRAM
  pressure evicts any co-resident model before ns loads.
* Only ``/v1/chat/completions`` exists; ``/v1/completions``, embeddings and
  rerank are rejected by the routes layer.
* ``ns-server`` writes its compile/prefix cache under ``.cache/`` relative to
  its working directory (inherited from DynLLM), which keeps warm starts
  across model reloads.
* A hard low ``max_tokens`` budget truncates into ``reasoning_content`` and
  leaves ``content`` empty; clients should budget generously (≥2048).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from dynllm.backends.base import SubprocessBackend
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 1.0

# ns-server binds the port only once the model is fully compiled and staged:
# cold start ≈70 s, warm (IR cache hit) ≈40 s, first-request prefill adds more.
_READY_TIMEOUT_SECONDS = 300.0

# ns-server aborts at startup with any non-zero value (classic-decode only).
_FORBIDDEN_ENV = {"NS3_SPEC"}


class NeutronStarBackend(SubprocessBackend):
    """Backend that manages one ``ns-server`` subprocess per loaded model."""

    def __init__(self, binary: str = "ns-server") -> None:
        super().__init__(
            binary=binary,
            install_hint=(
                "ns-server binary not found. Build NeutronStar "
                "(`cargo build --release --bin ns-server`, then optionally "
                "scripts/deploy.sh) or set backend.neutronstar_binary to an "
                "absolute executable path."
            ),
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.neutronstar

    @property
    def ready_timeout_seconds(self) -> Optional[float]:
        return _READY_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _validate(self, model: ModelConfig) -> None:
        path = Path(model.path)
        if not path.exists():
            raise RuntimeError(f"Model path not found: {path}")
        if path.is_dir() and not (path / "model.json").is_file():
            raise RuntimeError(
                f"NeutronStar model directory '{path}' does not contain a "
                "model.json workspace descriptor."
            )

    def _build_command(self, model: ModelConfig, port: int, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "--model",
            str(model.path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            model.name,
        ]
        if model.ns_max_seq is not None:
            cmd.extend(["--max-seq", str(model.ns_max_seq)])

        cmd.extend(model.extra_args)
        return cmd

    def _build_env(self, model: ModelConfig) -> Optional[dict[str, str]]:
        if not model.ns_env:
            return None

        env = dict(os.environ)
        for key, value in model.ns_env.items():
            if key in _FORBIDDEN_ENV:
                raise RuntimeError(
                    f"ns_env must not set {key}: ns-server refuses speculative "
                    "decoding and aborts on startup."
                )
            env[key] = value
        return env

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        """Poll ``/health``; the listener only appears once the model is loaded."""
        url = f"http://127.0.0.1:{port}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if self._proc is not None and self._proc.returncode is not None:
                    logger.warning(
                        "ns-server for '%s' died while waiting for readiness "
                        "(exit %s)",
                        model.name,
                        self._proc.returncode,
                    )
                    return False
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        logger.info(
                            "ns-server on port %d is ready (model '%s')",
                            port,
                            model.name,
                        )
                        return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass  # still compiling: the port is not bound yet.

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning(
            "ns-server on port %d did not become ready in %.0fs for '%s'",
            port,
            timeout,
            model.name,
        )
        return False
