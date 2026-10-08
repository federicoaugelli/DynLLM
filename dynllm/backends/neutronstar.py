"""
NeutronStar (`ns-server`) backend — scaffold.

NeutronStar is a custom MoE inference engine for Intel Arc GPUs.  Its
``ns-server`` binary (still under active development) exposes a minimal
OpenAI-compatible API::

    ns-server <merged.gguf> [--host 127.0.0.1] [--port 8080]
              [--alias NAME] [--max-seq N]

    GET  /health              -> {"status":"ok"}
    GET  /v1/models           -> the single ``--alias`` entry
    POST /v1/chat/completions -> OpenAI chat completions (SSE + JSON)

It also needs the bundled OpenVINO runtime on ``LD_LIBRARY_PATH`` and a set of
``NS3_*`` tuning environment variables.

The integration surface (backend type, config binary, registry entry) is wired
up, but launching ``ns-server`` is intentionally left unimplemented until the
server API is stable.  Enable it with ``neutronstar`` in ``enabled_backends``
to start using it as soon as :meth:`NeutronStarBackend.start` lands.
"""

from __future__ import annotations

from dynllm.backends.base import Backend
from dynllm.core.config import BackendType, ModelConfig

_PENDING = (
    "The neutronstar backend is a scaffold: launching ns-server is not "
    "implemented yet. It will be enabled once the NeutronStar server API is "
    "stable."
)


class NeutronStarBackend(Backend):
    """Placeholder backend for NeutronStar ``ns-server`` (see module docstring)."""

    def __init__(self, binary: str = "ns-server") -> None:
        self._binary = binary

    @property
    def backend_type(self) -> BackendType:
        return BackendType.neutronstar

    async def start(self, model: ModelConfig, port: int) -> None:
        raise NotImplementedError(_PENDING)

    async def stop(self) -> None:
        return None

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        return False
