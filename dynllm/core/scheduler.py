"""
Idle-timeout scheduler.

Runs a background asyncio task that periodically scans all loaded models
and unloads any that have not been used within ``idle_timeout_seconds``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynllm.core.vram_manager import VRAMManager
    from dynllm.db.manager import StateManager

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
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
    ) -> None:
        self._vram = vram_manager
        self._state = state
        self._idle_timeout = idle_timeout
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

    async def _check_idle(self) -> None:
        """Unload any model whose last_used_at is older than idle_timeout."""
        loaded = await self._state.get_loaded()
        now = datetime.now(timezone.utc)

        for model_state in loaded:
            if model_state.last_used_at is None:
                continue

            last_used = model_state.last_used_at
            # Ensure last_used is timezone-aware for comparison
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)

            idle_seconds = (now - last_used).total_seconds()
            if idle_seconds >= self._idle_timeout:
                logger.info(
                    "Auto-unloading idle model '%s' (idle for %.0fs)",
                    model_state.name,
                    idle_seconds,
                )
                try:
                    await self._vram.unload(model_state.name)
                except Exception as exc:
                    logger.error(
                        "Error auto-unloading model '%s': %s",
                        model_state.name,
                        exc,
                    )
