from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from dynllm.backends.base import InProcessBackend
from dynllm.backends.tts.base import TTSEngine
from dynllm.backends.tts.engines import ENGINE_REGISTRY
from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)


class TTSBackend(InProcessBackend):
    """In-process TTS backend.

    Unlike the subprocess backends, the model is loaded directly inside the
    DynLLM process via a :class:`TTSEngine` plugin.  Every blocking engine call
    (load, unload, synthesis) runs in a single-thread executor so the event loop
    stays responsive and the engine is never touched concurrently.
    """

    def __init__(self) -> None:
        self._engine: TTSEngine | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.tts

    @property
    def loaded(self) -> bool:
        return self._engine is not None and self._engine.loaded

    async def _load(self, model: ModelConfig) -> None:
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
        await self._run(engine.load)
        self._engine = engine

    async def _unload(self) -> None:
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        await self._run(engine.unload)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        engine = self._engine
        if engine is None or not engine.loaded:
            raise RuntimeError("TTS model not loaded")
        return await self._run(
            engine.synthesize,
            text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )
