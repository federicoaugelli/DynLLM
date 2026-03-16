"""
VRAM Manager – central orchestrator for model load/unload decisions.

Rules (from the spec):
  1. Before loading a model, check if there is enough free VRAM.
  2. If not enough VRAM: evict the *least-recently loaded* model (LIFO order)
     until enough space is freed.
  3. If multiple models fit simultaneously, none are evicted.
  4. A model that has not been used for ``idle_timeout_seconds`` is
     automatically unloaded (handled by the scheduler, not this class).

This manager is the single point of truth for load/unload operations.
All public methods are async and serialised through an internal lock to
prevent concurrent double-loads or race conditions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from dynllm.backends.base import Backend
from dynllm.backends.llamacpp import LlamaCppBackend
from dynllm.backends.openvino import OpenVINOBackend
from dynllm.core.config import BackendType, ModelConfig, Settings
from dynllm.db.manager import StateManager
from dynllm.db.models import ModelStatus

logger = logging.getLogger(__name__)


class PortAllocator:
    """Simple sequential port allocator within a configured range."""

    def __init__(self, start: int, end: int) -> None:
        self._start = start
        self._end = end
        self._next = start
        self._lock = asyncio.Lock()

    async def allocate(self) -> int:
        async with self._lock:
            port = self._next
            self._next += 1
            if self._next > self._end:
                self._next = self._start
            return port


class VRAMManager:
    """
    Orchestrates model loading, unloading, and VRAM accounting.

    This is intentionally kept stateless with respect to process handles –
    all persistent state is in the StateManager (SQLite). VRAM accounting
    is derived from the persisted ``vram_mb`` fields.
    """

    def __init__(self, settings: Settings, state: StateManager) -> None:
        self._settings = settings
        self._state = state
        self._lock = asyncio.Lock()

        # Instantiate backends for enabled backend types only
        self._backends: dict[BackendType, Backend] = {}
        if BackendType.llamacpp in settings.enabled_backends:
            self._backends[BackendType.llamacpp] = LlamaCppBackend(
                binary=settings.backend.llamacpp_binary
            )
        if BackendType.openvino in settings.enabled_backends:
            self._backends[BackendType.openvino] = OpenVINOBackend(
                binary=settings.backend.ovms_binary
            )

        self._ports = PortAllocator(
            settings.backend.port_range_start,
            settings.backend.port_range_end,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_loaded(self, model: ModelConfig) -> int:
        """
        Ensure *model* is loaded and ready, returning its listening port.

        If the model is already loaded the call is a no-op (beyond updating
        last_used_at).  If not, VRAM is freed as needed and the model is
        loaded.
        """
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
                    f"(status={state.status}). Retry shortly."
                )

            # Check that the requested backend is enabled
            if model.backend not in self._backends:
                raise RuntimeError(
                    f"Backend '{model.backend}' is not enabled in the current "
                    "configuration."
                )

            # Free enough VRAM for the new model
            await self._evict_for(model.vram_mb)

            # Load the model
            port = await self._load(model)
            return port

    async def unload(self, model_name: str) -> None:
        """Explicitly unload a model by name."""
        async with self._lock:
            await self._unload_by_name(model_name)

    async def get_port(self, model_name: str) -> Optional[int]:
        """Return the port for a loaded model, or None if not loaded."""
        state = await self._state.get(model_name)
        if state is not None and state.status == ModelStatus.loaded:
            return state.port
        return None

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    async def _free_vram(self) -> int:
        """Return estimated free VRAM in MB."""
        used = await self._state.total_loaded_vram()
        return max(0, self._settings.total_vram_mb - used)

    async def _evict_for(self, required_mb: int) -> None:
        """
        Evict loaded models in LIFO order until *required_mb* VRAM is free.
        """
        while True:
            free = await self._free_vram()
            if free >= required_mb:
                return

            # Find the most-recently loaded model (highest load_order) to evict
            loaded_models = await self._state.get_loaded()
            if not loaded_models:
                raise RuntimeError(
                    f"Not enough VRAM: need {required_mb} MB but only "
                    f"{free} MB free with no models to evict."
                )

            # LIFO: evict the model with the highest load_order
            victim = max(loaded_models, key=lambda m: m.load_order)
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
        """Load a model and return its port."""
        backend = self._backends[model.backend]
        port = await self._ports.allocate()
        load_order = await self._state.next_load_order()

        await self._state.set_loading(model.name, model.backend.value, model.vram_mb)

        try:
            pid = await backend.start(model, port)
        except Exception as exc:
            await self._state.set_error(model.name)
            raise RuntimeError(
                f"Failed to start backend for '{model.name}': {exc}"
            ) from exc

        ready = await backend.is_ready(port)
        if not ready:
            # Kill the stalled process
            try:
                await backend.stop(pid)
            except Exception:
                pass
            await self._state.set_error(model.name)
            raise RuntimeError(
                f"Backend for model '{model.name}' did not become ready."
            )

        await self._state.set_loaded(model.name, pid, port, load_order)
        logger.info(
            "Model '%s' loaded on port %d (PID %d, VRAM %d MB)",
            model.name,
            port,
            pid,
            model.vram_mb,
        )
        return port

    async def _unload_by_name(self, model_name: str) -> None:
        """Unload a single model by name (lock must already be held)."""
        state = await self._state.get(model_name)
        if state is None or state.status != ModelStatus.loaded:
            return

        await self._state.set_unloading(model_name)
        backend = self._backends.get(BackendType(state.backend))

        if backend is not None and state.pid is not None:
            try:
                await backend.stop(state.pid)
            except Exception as exc:
                logger.warning(
                    "Error stopping backend for '%s' (PID %d): %s",
                    model_name,
                    state.pid,
                    exc,
                )

        await self._state.set_unloaded(model_name)
        logger.info("Model '%s' unloaded", model_name)
