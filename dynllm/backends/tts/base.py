from __future__ import annotations

import abc


class TTSEngine(abc.ABC):
    """Abstract base for a TTS model implementation.

    Each engine encapsulates the model-specific loading, unloading, and
    synthesis logic.  Engines are instantiated by ``TTSBackend`` which
    manages their lifecycle and provides the in-process inference endpoint.
    """

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    async def load(self) -> None:
        """Load the model into memory (GPU/CPU). Called once at startup."""

    @abc.abstractmethod
    async def unload(self) -> None:
        """Free the model from memory. Called at shutdown or eviction."""

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """Synthesise speech from *text* and return raw audio bytes."""
