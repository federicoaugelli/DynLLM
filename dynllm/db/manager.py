"""
StateManager: thread-safe, async-aware SQLite-backed persistence layer.

All model load/unload decisions are serialised through this manager so the
on-disk database is always consistent with the in-memory state – even if the
process crashes and restarts.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlmodel import Session, SQLModel, create_engine, select

from dynllm.db.models import ModelState, ModelStatus

logger = logging.getLogger(__name__)


class StateManager:
    """Manages persistent model lifecycle state in an SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        url = f"sqlite:///{db_path}"
        # check_same_thread=False is safe here because we serialise all
        # access through an asyncio.Lock.
        self._engine = create_engine(url, connect_args={"check_same_thread": False})
        self._lock = asyncio.Lock()
        SQLModel.metadata.create_all(self._engine)
        logger.info("StateManager initialised with database: %s", db_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, session: Session, name: str) -> ModelState:
        """Return an existing ModelState row or create a new unloaded one."""
        row = session.get(ModelState, name)
        if row is None:
            row = ModelState(name=name)
            session.add(row)
            session.flush()
        return row

    # ------------------------------------------------------------------
    # Public API (all async, all serialised through self._lock)
    # ------------------------------------------------------------------

    async def get(self, name: str) -> Optional[ModelState]:
        """Return a *detached* snapshot of the model state, or None."""
        async with self._lock:
            with Session(self._engine) as session:
                row = session.get(ModelState, name)
                if row is None:
                    return None
                session.expunge(row)
                return row

    async def get_all(self) -> list[ModelState]:
        """Return detached snapshots of all known model states."""
        async with self._lock:
            with Session(self._engine) as session:
                rows = session.exec(select(ModelState)).all()
                for r in rows:
                    session.expunge(r)
                return list(rows)

    async def get_loaded(self) -> list[ModelState]:
        """Return detached snapshots of all currently-loaded models."""
        async with self._lock:
            with Session(self._engine) as session:
                rows = session.exec(
                    select(ModelState).where(ModelState.status == ModelStatus.loaded)
                ).all()
                for r in rows:
                    session.expunge(r)
                return list(rows)

    async def next_load_order(self) -> int:
        """Return the next monotonically increasing load-order value."""
        async with self._lock:
            with Session(self._engine) as session:
                rows = session.exec(select(ModelState)).all()
                if not rows:
                    return 1
                return max(r.load_order for r in rows) + 1

    async def set_loading(self, name: str, backend: str, vram_mb: int) -> None:
        async with self._lock:
            with Session(self._engine) as session:
                row = self._get_or_create(session, name)
                row.status = ModelStatus.loading
                row.backend = backend
                row.vram_mb = vram_mb
                session.add(row)
                session.commit()

    async def set_loaded(self, name: str, pid: int, port: int, load_order: int) -> None:
        async with self._lock:
            with Session(self._engine) as session:
                row = self._get_or_create(session, name)
                row.mark_loaded(pid, port, load_order)
                session.add(row)
                session.commit()

    async def set_unloading(self, name: str) -> None:
        async with self._lock:
            with Session(self._engine) as session:
                row = self._get_or_create(session, name)
                row.status = ModelStatus.unloading
                session.add(row)
                session.commit()

    async def set_unloaded(self, name: str) -> None:
        async with self._lock:
            with Session(self._engine) as session:
                row = self._get_or_create(session, name)
                row.mark_unloaded()
                session.add(row)
                session.commit()

    async def set_error(self, name: str) -> None:
        async with self._lock:
            with Session(self._engine) as session:
                row = self._get_or_create(session, name)
                row.mark_error()
                session.add(row)
                session.commit()

    async def touch(self, name: str) -> None:
        """Update last_used_at for a model (called on every inference)."""
        async with self._lock:
            with Session(self._engine) as session:
                row = session.get(ModelState, name)
                if row is not None:
                    row.touch()
                    session.add(row)
                    session.commit()

    async def total_loaded_vram(self) -> int:
        """Return total VRAM consumed by all loaded/loading models."""
        loaded = await self.get_loaded()
        return sum(r.vram_mb for r in loaded)

    async def heal_stale_states(self) -> None:
        """
        On startup, any model left in 'loading' or 'unloading' state
        from a previous crash is marked as unloaded.
        """
        async with self._lock:
            with Session(self._engine) as session:
                stale_states = [ModelStatus.loading, ModelStatus.unloading]
                rows = session.exec(
                    select(ModelState).where(ModelState.status.in_(stale_states))
                ).all()
                for row in rows:
                    logger.warning(
                        "Healing stale state for model '%s' (%s -> unloaded)",
                        row.name,
                        row.status,
                    )
                    row.mark_unloaded()
                    session.add(row)
                if rows:
                    session.commit()
