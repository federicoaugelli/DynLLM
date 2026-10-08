"""
Orchestrator: model load/unload decisions and VRAM accounting.

Rules:
  1. Before loading a model, check that enough VRAM is free.
  2. If not, evict the *most recently loaded* model (LIFO) until it fits.
  3. If several models fit simultaneously, none are evicted.
  4. A model idle past its effective timeout is unloaded by the scheduler.

Each loaded model owns its own backend instance; all state transitions are
serialised through an internal lock so the on-disk state can never disagree
with the running processes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from dynllm.backends import Backend, BackendRegistry
from dynllm.core.config import ModelConfig, ModelType, Settings
from dynllm.db.manager import StateManager
from dynllm.db.models import ModelStatus

logger = logging.getLogger(__name__)

# Readiness budget per workload; ASR models load slowly.
_READY_TIMEOUT = 60.0
_READY_TIMEOUT_TRANSCRIPTION = 300.0


class RequestTracker:
    """Counts in-flight inference requests per model (non-blocking reads)."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def increment(self, model_name: str) -> None:
        async with self._lock:
            self._counts[model_name] = self._counts.get(model_name, 0) + 1

    async def decrement(self, model_name: str) -> None:
        async with self._lock:
            count = self._counts.get(model_name, 0)
            if count > 1:
                self._counts[model_name] = count - 1
            else:
                self._counts.pop(model_name, None)

    def count(self, model_name: str) -> int:
        return self._counts.get(model_name, 0)


class PortAllocator:
    """Sequential port allocator within a configured range."""

    def __init__(self, start: int, end: int) -> None:
        self._start = start
        self._end = end
        self._next = start
        self._lock = asyncio.Lock()

    async def allocate(self) -> int:
        async with self._lock:
            port = self._next
            self._next = self._next + 1 if self._next < self._end else self._start
            return port


class VRAMManager:
    """Owns backend instances and decides what stays resident in VRAM."""

    def __init__(self, settings: Settings, state: StateManager) -> None:
        self._settings = settings
        self._state = state
        self._registry = BackendRegistry(settings)
        self._ports = PortAllocator(
            settings.backend.port_range_start,
            settings.backend.port_range_end,
        )
        self._lock = asyncio.Lock()
        self._instances: dict[str, Backend] = {}
        self._requests = RequestTracker()

    # ------------------------------------------------------------------
    # Request tracking (delegated to the tracker)
    # ------------------------------------------------------------------

    async def increment_active(self, model_name: str) -> None:
        await self._requests.increment(model_name)

    async def decrement_active(self, model_name: str) -> None:
        await self._requests.decrement(model_name)

    def active_count(self, model_name: str) -> int:
        return self._requests.count(model_name)

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    async def ensure_loaded(self, model: ModelConfig) -> int:
        """Ensure *model* is loaded and return its listening port."""
        async with self._lock:
            state = await self._state.get(model.name)

            if state is not None and state.status == ModelStatus.loaded:
                await self._state.touch(model.name)
                assert state.port is not None
                return state.port

            if state is not None and state.status in (
                ModelStatus.loading,
                ModelStatus.unloading,
            ):
                raise RuntimeError(
                    f"Model '{model.name}' is currently transitioning "
                    f"(status={state.status.value}). Retry shortly."
                )

            if not self._registry.is_enabled(model.backend):
                raise RuntimeError(
                    f"Backend '{model.backend.value}' is not enabled in the "
                    "current configuration."
                )

            await self._evict_for(self._total_vram_mb(model))
            return await self._load(model)

    async def unload(self, model_name: str) -> None:
        """Explicitly unload a model by name."""
        async with self._lock:
            await self._unload_by_name(model_name)

    async def get_port(self, model_name: str) -> Optional[int]:
        """Return the port of a loaded model, or ``None`` if not loaded."""
        state = await self._state.get(model_name)
        if state is not None and state.status == ModelStatus.loaded:
            return state.port
        return None

    def get_instance(self, model_name: str) -> Optional[Backend]:
        """Return the live backend instance for *model_name*, if loaded."""
        return self._instances.get(model_name)

    # ------------------------------------------------------------------
    # Internals (must be called with self._lock held)
    # ------------------------------------------------------------------

    @staticmethod
    def _total_vram_mb(model: ModelConfig) -> int:
        return model.vram_mb + (model.draft_model_vram_mb or 0)

    async def _free_vram(self) -> int:
        used = await self._state.total_loaded_vram()
        return max(0, self._settings.total_vram_mb - used)

    async def _evict_for(self, required_mb: int) -> None:
        """Evict loaded models LIFO until *required_mb* is free.

        Models with in-flight requests are never evicted.
        """
        while await self._free_vram() < required_mb:
            free = await self._free_vram()
            loaded = await self._state.get_loaded()
            evictable = [m for m in loaded if self.active_count(m.name) == 0]
            if not evictable:
                busy = [m.name for m in loaded if self.active_count(m.name) > 0]
                raise RuntimeError(
                    f"Not enough VRAM: need {required_mb} MB but only {free} MB "
                    f"free. All loaded models have active requests and cannot "
                    f"be evicted: {busy}"
                )

            victim = max(evictable, key=lambda m: m.load_order)
            logger.info(
                "VRAM pressure: evicting model '%s' (%d MB) to free space for "
                "incoming request (need %d MB, have %d MB free)",
                victim.name,
                victim.vram_mb,
                required_mb,
                free,
            )
            await self._unload_by_name(victim.name)

    async def _load(self, model: ModelConfig) -> int:
        backend = self._registry.create(model.backend)
        port = await self._ports.allocate()
        load_order = await self._state.next_load_order()
        await self._state.set_loading(
            model.name, model.backend.value, self._total_vram_mb(model)
        )

        try:
            await backend.start(model, port)
        except Exception as exc:
            await self._state.set_error(model.name)
            raise RuntimeError(
                f"Failed to start backend for '{model.name}': {exc}"
            ) from exc

        timeout = (
            _READY_TIMEOUT_TRANSCRIPTION
            if model.model_type == ModelType.transcription
            else _READY_TIMEOUT
        )
        if not await backend.is_ready(port, model, timeout):
            await backend.stop()
            await self._state.set_error(model.name)
            raise RuntimeError(
                f"Backend for model '{model.name}' did not become ready."
            )

        self._instances[model.name] = backend
        await self._state.set_loaded(model.name, backend.pid or 0, port, load_order)
        logger.info(
            "Model '%s' loaded on port %d (PID %s, VRAM %d MB%s)",
            model.name,
            port,
            backend.pid,
            self._total_vram_mb(model),
            f" + {model.draft_model_vram_mb} MB draft"
            if model.draft_model_vram_mb
            else "",
        )
        return port

    async def _unload_by_name(self, model_name: str) -> None:
        state = await self._state.get(model_name)
        if state is None or state.status != ModelStatus.loaded:
            return

        await self._state.set_unloading(model_name)
        backend = self._instances.pop(model_name, None)
        if backend is not None:
            try:
                await backend.stop()
            except Exception as exc:
                logger.warning("Error stopping backend for '%s': %s", model_name, exc)

        await self._state.set_unloaded(model_name)
        logger.info("Model '%s' unloaded", model_name)
