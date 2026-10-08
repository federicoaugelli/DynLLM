"""
Backend abstraction.

A *backend* owns the runtime resource for exactly one loaded model.  The
orchestrator creates a fresh instance per load and drops it on unload, so a
backend never has to juggle more than one model.

Two families are provided:

``SubprocessBackend``
    Runs an external HTTP server (``llama-server``, ``ovms``, …) as a child
    process and proxies inference to it.
``InProcessBackend``
    Loads the model inside the DynLLM process (``tts``, ``privacy_filter``) and
    answers directly, off-loading blocking work to a thread pool.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from dynllm.core.config import BackendType, ModelConfig

logger = logging.getLogger(__name__)


class Backend(abc.ABC):
    """Lifecycle interface shared by every backend."""

    @property
    @abc.abstractmethod
    def backend_type(self) -> BackendType:
        """The type identifier for this backend."""

    @property
    def pid(self) -> Optional[int]:
        """OS PID of the backing process, or ``None`` for in-process backends."""
        return None

    @abc.abstractmethod
    async def start(self, model: ModelConfig, port: int) -> None:
        """Acquire the resource for *model* and bind it to *port*."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release the resource. Must not raise if it is already gone."""

    @abc.abstractmethod
    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        """Poll until the backend accepts requests or *timeout* expires."""


class SubprocessBackend(Backend):
    """Base class for backends backed by an external HTTP server process."""

    def __init__(self, binary: str, install_hint: str) -> None:
        self._binary = shutil.which(binary) or binary
        self._install_hint = install_hint
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._workdir: Optional[tempfile.TemporaryDirectory] = None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @abc.abstractmethod
    def _build_command(self, model: ModelConfig, port: int, workdir: Path) -> list[str]:
        """Return the argv used to launch the backend for *model*."""

    @abc.abstractmethod
    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        """Poll the backend's readiness endpoint."""

    def _validate(self, model: ModelConfig) -> None:
        """Validate the model before launching (override when a path is required)."""

    async def start(self, model: ModelConfig, port: int) -> None:
        self._binary = self._require_binary()
        self._validate(model)
        self._workdir = tempfile.TemporaryDirectory(
            prefix=f"dynllm_{model.backend.value}_{model.name}_"
        )
        workdir = Path(self._workdir.name)
        cmd = self._build_command(model, port, workdir)

        logger.info(
            "Starting %s for '%s' on port %d: %s",
            self.backend_type.value,
            model.name,
            port,
            " ".join(cmd),
        )

        stderr_path = workdir / "stderr.log"
        with open(stderr_path, "wb") as stderr_file:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_file,
            )

        # Catch binaries that die immediately (bad flags, missing model, …).
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            return

        error = stderr_path.read_text(errors="replace").strip()
        self._cleanup()
        raise RuntimeError(
            f"{self.backend_type.value} exited immediately for model "
            f"'{model.name}': {error}"
        )

    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            logger.info("Stopping %s (PID %d)", self.backend_type.value, proc.pid)
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Force-killing %s (PID %d)", self.backend_type.value, proc.pid
                )
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        self._cleanup()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None

    def _require_binary(self) -> str:
        resolved = shutil.which(self._binary)
        if resolved is not None:
            return resolved

        path = Path(self._binary).expanduser()
        if path.exists() and not path.is_file():
            raise RuntimeError(
                f"Backend binary path is not a file: '{path}'. "
                "Point the config at an executable binary."
            )
        if path.is_file():
            raise RuntimeError(
                f"Backend binary exists but is not executable: '{path}'. "
                "Fix permissions or point the config at an executable file."
            )
        raise RuntimeError(f"{self._install_hint} Configured binary: '{self._binary}'.")


class InProcessBackend(Backend):
    """Base class for backends whose model lives inside the DynLLM process."""

    @property
    @abc.abstractmethod
    def loaded(self) -> bool:
        """Whether the model is currently resident."""

    @abc.abstractmethod
    async def _load(self, model: ModelConfig) -> None:
        """Load the model into memory."""

    @abc.abstractmethod
    async def _unload(self) -> None:
        """Free the model from memory."""

    async def start(self, model: ModelConfig, port: int) -> None:
        await self._load(model)

    async def stop(self) -> None:
        await self._unload()

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        return self.loaded
