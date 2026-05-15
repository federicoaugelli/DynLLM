from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dynllm.backends.base import Backend
from dynllm.backends.tts.base import TTSEngine
from dynllm.backends.tts.engines import ENGINE_REGISTRY
from dynllm.core.config import BackendType, ModelConfig, ModelType

logger = logging.getLogger(__name__)


class TTSBackend(Backend):
    """In-process TTS backend.

    Unlike subprocess-based backends (llama.cpp, OVMS), TTSBackend loads
    the model directly in-process via a ``TTSEngine`` plugin.  Synthesis
    runs in a thread pool executor to avoid blocking the asyncio event loop.
    """

    def __init__(self) -> None:
        self._engine: TTSEngine | None = None
        self._pid: int = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tts",
        )

    # ------------------------------------------------------------------
    # Backend ABC
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.tts

    async def start(self, model: ModelConfig, port: int) -> int:
        engine_cls = ENGINE_REGISTRY.get(model.tts_engine)
        if engine_cls is None:
            raise RuntimeError(
                f"Unknown TTS engine '{model.tts_engine}'. "
                f"Available: {', '.join(ENGINE_REGISTRY)}"
            )

        engine = engine_cls(
            model_path=str(model.path),
            device=model.target_device.lower(),
        )
        await engine.load()
        self._engine = engine
        # PID is just an internal reference, not an OS PID
        self._pid = id(engine)
        return self._pid

    async def stop(self, pid: int) -> None:
        if self._engine is not None:
            await self._engine.unload()
            self._engine = None

    async def is_ready(
        self,
        port: int,
        model_name: str = "",
        timeout: float = 60.0,
        model_type: str = "llm",
    ) -> bool:
        return self._engine is not None and self._engine.loaded

    # ------------------------------------------------------------------
    # TTS-specific
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        if self._engine is None or not self._engine.loaded:
            raise RuntimeError("TTS model not loaded")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice,
            response_format,
            speed,
        )

    def _synthesize_sync(
        self,
        text: str,
        voice: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        """Synchronous wrapper run in the thread pool."""
        import asyncio

        return asyncio.run(
            self._engine.synthesize(
                text,
                voice=voice,
                response_format=response_format,
                speed=speed,
            )
        )
