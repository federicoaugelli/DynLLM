"""
Backend abstraction layer.

Every backend must implement the ``Backend`` ABC.  The interface is intentionally
narrow: start a subprocess, stop it, and report whether it is alive.

Actual inference is done by forwarding HTTP requests to the backend's local
server port – the proxy never calls inference methods directly.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import signal
from typing import Optional

from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)


class Backend(abc.ABC):
    """
    Abstract base class for a model-serving backend.

    A Backend is responsible for:
    - Launching a subprocess that exposes an OpenAI-compatible HTTP server.
    - Stopping (and optionally gracefully terminating) that subprocess.
    - Reporting the port the subprocess is bound to.
    """

    @property
    @abc.abstractmethod
    def backend_type(self) -> BackendType:
        """The type identifier for this backend."""

    @abc.abstractmethod
    async def start(self, model: ModelConfig, port: int) -> int:
        """
        Start the backend subprocess for *model* on *port*.

        Returns the PID of the started subprocess.
        Raises ``RuntimeError`` on failure.
        """

    @abc.abstractmethod
    async def stop(self, pid: int) -> None:
        """
        Stop (and wait for) the subprocess with *pid*.

        Should not raise even if the process is already dead.
        """

    @abc.abstractmethod
    async def is_ready(
        self, port: int, model_name: str = "", timeout: float = 60.0, model_type: str = "llm"
    ) -> bool:
        """
        Poll the backend until it is accepting requests or *timeout* expires.

        ``model_name`` is the logical name of the model being served (used by
        backends that require it for their readiness endpoint, e.g. OVMS).

        ``model_type`` discriminates readiness detection (e.g. audio vs LLM).

        Returns True if ready, False otherwise.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all backends
    # ------------------------------------------------------------------

    async def _wait_for_process(self, pid: int, timeout: float = 10.0) -> None:
        """Wait for a process to exit, killing it forcefully if needed."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "kill",
                "-0",
                str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                import os

                os.kill(pid, 0)  # raises OSError if not running
                await asyncio.sleep(0.2)
            except OSError:
                return  # process is gone
        # Force-kill if still alive
        try:
            import os

            os.kill(pid, signal.SIGKILL)
            logger.warning("Force-killed backend subprocess PID %d", pid)
        except OSError:
            pass

    async def _terminate_pid(self, pid: int) -> None:
        """Send SIGTERM then wait; escalate to SIGKILL if needed."""
        import os

        try:
            os.kill(pid, signal.SIGTERM)
            logger.debug("Sent SIGTERM to PID %d", pid)
        except OSError:
            return  # already dead

        await self._wait_for_process(pid, timeout=10.0)
