from __future__ import annotations

import abc

from dynllm.backends.gpu import empty_torch_cache

__all__ = ["TTSEngine", "empty_torch_cache"]


class TTSEngine(abc.ABC):
    """A concrete text-to-speech model implementation.

    Engines are synchronous and blocking: they are always driven from
    ``TTSBackend``'s thread pool so the event loop stays responsive.
    """

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Free the model from memory."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """Synthesise speech from *text* and return raw audio bytes."""
