"""
SQLModel database models for persistent model state tracking.

Using SQLModel (which wraps SQLAlchemy + Pydantic) to get both
DB persistence and Pydantic validation in one place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ModelStatus(str, Enum):
    """Lifecycle states of a backend model instance."""

    unloaded = "unloaded"
    """Model is not consuming any VRAM."""

    loading = "loading"
    """Backend subprocess is starting up."""

    loaded = "loaded"
    """Model is ready for inference."""

    unloading = "unloading"
    """Backend subprocess is being shut down."""

    error = "error"
    """Model failed to load or encountered a fatal error."""


class ModelState(SQLModel, table=True):
    """
    Persisted record of a model's runtime state.

    One row per configured model; updated in-place as the model
    transitions between lifecycle states.
    """

    __tablename__ = "model_state"

    # Use model name as the primary key since it's the unique identifier
    # given to us by the user config.
    name: str = Field(primary_key=True)

    status: ModelStatus = Field(default=ModelStatus.unloaded)

    backend: str = Field(default="")
    """Backend type string ('llamacpp' or 'openvino')."""

    vram_mb: int = Field(default=0)
    """VRAM claimed by this model while loaded."""

    # Subprocess management
    pid: Optional[int] = Field(default=None)
    """PID of the backend server subprocess, if running."""

    port: Optional[int] = Field(default=None)
    """Local port the backend subprocess is listening on."""

    # Time tracking for LIFO eviction and idle timeout
    loaded_at: Optional[datetime] = Field(default=None)
    """When the model was last successfully loaded."""

    last_used_at: Optional[datetime] = Field(default=None)
    """Timestamp of the most recent inference request."""

    # Load order counter for strict LIFO eviction ordering.
    # Higher value = more recently loaded.
    load_order: int = Field(default=0)

    def touch(self) -> None:
        """Update last_used_at to now (call on each inference request)."""
        self.last_used_at = datetime.now(timezone.utc)

    def mark_loaded(self, pid: int, port: int, load_order: int) -> None:
        now = datetime.now(timezone.utc)
        self.status = ModelStatus.loaded
        self.pid = pid
        self.port = port
        self.loaded_at = now
        self.last_used_at = now
        self.load_order = load_order

    def mark_unloaded(self) -> None:
        self.status = ModelStatus.unloaded
        self.pid = None
        self.port = None
        self.loaded_at = None
        self.load_order = 0

    def mark_error(self) -> None:
        self.status = ModelStatus.error
        self.pid = None
        self.port = None
