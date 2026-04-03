"""
Idle-timeout scheduler.

Runs a background asyncio task that periodically scans all loaded models
and unloads any that have not been used within their effective idle timeout.

Effective timeout resolution per model:
  1. ``model.unload_time`` if explicitly set in config.
  2. Otherwise, the global ``idle_timeout_seconds``.
  3. If the effective timeout is ``math.inf`` (or -1 in config), the model is
     never auto-unloaded by the scheduler (it can still be evicted by VRAM
     pressure or unloaded manually).
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynllm.core.config import Settings
    from dynllm.core.vram_manager import VRAMManager
    from dynllm.db.manager import StateManager

from dynllm.core.vram_manager import get_active_count
from dynllm.db.models import ModelStatus

logger = logging.getLogger(__name__)

# How often (in seconds) the scheduler wakes up and checks idle models.
# Set to ~10 % of the idle timeout, minimum 10 s.
_DEFAULT_CHECK_INTERVAL = 30.0


class IdleScheduler:
    """
    Background task that auto-unloads idle models.

    Usage::

        scheduler = IdleScheduler(vram_manager, state_manager, idle_timeout=300)
        task = asyncio.create_task(scheduler.run())
        # on shutdown:
        scheduler.stop()
        await task
    """

    def __init__(
        self,
        vram_manager: "VRAMManager",
        state: "StateManager",
        idle_timeout: float,
        settings: "Settings | None" = None,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
    ) -> None:
        self._vram = vram_manager
        self._state = state
        self._idle_timeout = idle_timeout
        self._settings = settings
        self._check_interval = check_interval
        self._running = False

    async def run(self) -> None:
        """Main scheduler loop. Run as an asyncio task."""
        self._running = True
        logger.info(
            "Idle scheduler started (timeout=%ds, check_interval=%ds)",
            self._idle_timeout,
            self._check_interval,
        )
        while self._running:
            await asyncio.sleep(self._check_interval)
            if not self._running:
                break
            await self._check_idle()
        logger.info("Idle scheduler stopped")

    def stop(self) -> None:
        """Signal the scheduler loop to exit after the next sleep."""
        self._running = False

    def _effective_timeout(self, model_name: str) -> float:
        """
        Return the effective idle timeout (in seconds) for *model_name*.

        Resolves in order:
        1. Per-model ``unload_time`` from config (if set).
        2. Global ``idle_timeout_seconds``.
        Returns ``math.inf`` when the model should never be auto-unloaded.
        """
        if self._settings is not None:
            model_cfg = self._settings.model_by_name(model_name)
            if model_cfg is not None and model_cfg.unload_time is not None:
                return model_cfg.unload_time
        return self._idle_timeout

    async def _check_idle(self) -> None:
        """Unload any model whose last_used_at is older than its effective idle timeout."""
        loaded = await self._state.get_loaded()
        now = datetime.now(timezone.utc)

        for model_state in loaded:
            if model_state.last_used_at is None:
                continue

            effective_timeout = self._effective_timeout(model_state.name)

            # math.inf means "never auto-unload"
            if math.isinf(effective_timeout):
                continue

            last_used = model_state.last_used_at
            # Ensure last_used is timezone-aware for comparison
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)

            idle_seconds = (now - last_used).total_seconds()
            if idle_seconds >= effective_timeout:
                # Do not evict a model that is currently serving a request.
                if get_active_count(model_state.name) > 0:
                    logger.debug(
                        "Skipping auto-unload of '%s': has active requests",
                        model_state.name,
                    )
                    continue
                logger.info(
                    "Auto-unloading idle model '%s' (idle for %.0fs, timeout=%.0fs)",
                    model_state.name,
                    idle_seconds,
                    effective_timeout,
                )
                try:
                    await self._vram.unload(model_state.name)
                except Exception as exc:
                    logger.error(
                        "Error auto-unloading model '%s': %s",
                        model_state.name,
                        exc,
                    )
