"""Backend registry: maps a :class:`BackendType` to a fresh backend instance."""

from __future__ import annotations

from collections.abc import Callable

from dynllm.backends.base import Backend, InProcessBackend, SubprocessBackend
from dynllm.backends.llamacpp import LlamaCppBackend
from dynllm.backends.neutronstar import NeutronStarBackend
from dynllm.backends.openvino import OpenVINOBackend
from dynllm.backends.privacy_filter import PrivacyFilterBackend
from dynllm.backends.transformers import TransformersBackend
from dynllm.backends.tts import TTSBackend
from dynllm.core.config import BackendType, Settings

_FACTORIES: dict[BackendType, Callable[[Settings], Backend]] = {
    BackendType.llamacpp: lambda s: LlamaCppBackend(s.backend.llamacpp_binary),
    BackendType.openvino: lambda s: OpenVINOBackend(s.backend.ovms_binary),
    BackendType.transformers: lambda s: TransformersBackend(
        s.backend.transformers_binary
    ),
    BackendType.neutronstar: lambda s: NeutronStarBackend(s.backend.neutronstar_binary),
    BackendType.tts: lambda _s: TTSBackend(),
    BackendType.privacy_filter: lambda _s: PrivacyFilterBackend(),
}


class BackendRegistry:
    """Creates one backend instance per loaded model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_enabled(self, backend_type: BackendType) -> bool:
        return backend_type in self._settings.enabled_backends

    def create(self, backend_type: BackendType) -> Backend:
        if not self.is_enabled(backend_type):
            raise RuntimeError(
                f"Backend '{backend_type.value}' is not enabled in the current "
                "configuration."
            )
        factory = _FACTORIES.get(backend_type)
        if factory is None:
            raise RuntimeError(f"Unknown backend '{backend_type.value}'.")
        return factory(self._settings)


__all__ = [
    "Backend",
    "BackendRegistry",
    "InProcessBackend",
    "LlamaCppBackend",
    "NeutronStarBackend",
    "OpenVINOBackend",
    "PrivacyFilterBackend",
    "SubprocessBackend",
    "TTSBackend",
    "TransformersBackend",
]
