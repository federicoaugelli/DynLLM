"""
Configuration system for DynLLM.

Config is loaded from a YAML file (default: config.yaml in the working directory,
overridable via the DYNLLM_CONFIG env var).
"""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class BackendType(str, Enum):
    llamacpp = "llamacpp"
    openvino = "openvino"


class ModelType(str, Enum):
    llm = "llm"
    transcription = "transcription"
    speech = "speech"


class ModelConfig(BaseModel):
    """Declaration of a single model available to the proxy."""

    name: str
    """Unique model identifier used in API requests (e.g. 'llama3-8b')."""

    path: Path
    """Absolute or relative path to the model directory/file."""

    backend: BackendType
    """Which backend should serve this model."""

    model_type: ModelType = ModelType.llm
    """What kind of workload this model serves."""

    vram_mb: int = Field(ge=0)
    """Estimated VRAM this model consumes when loaded, in megabytes."""

    target_device: str = "CPU"
    """OpenVINO target device (e.g. CPU, GPU, NPU)."""

    # --- llama.cpp specific ---
    n_gpu_layers: int = Field(default=-1)
    """Number of layers to offload to GPU (-1 = all). Only used by llamacpp backend."""

    context_size: int = Field(default=4096)
    """Context window size. Only used by llamacpp backend."""

    # --- OVMS specific ---
    ovms_shape: Optional[str] = None
    """Optional shape hint for OVMS (e.g. 'auto'). Only used by openvino backend."""

    # --- Idle unload ---
    unload_time: Optional[float] = None
    """
    Per-model idle timeout in seconds before the model is automatically unloaded.

    * ``null`` / omitted – inherit the global ``idle_timeout_seconds`` setting.
    * positive number    – unload after this many idle seconds.
    * ``-1`` or ``inf``  – never auto-unload this model (it stays loaded until
                           VRAM pressure forces eviction or a manual unload).
    """

    @field_validator("unload_time", mode="before")
    @classmethod
    def parse_unload_time(cls, v: object) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str) and v.lower() in ("inf", "infinity", "never"):
            return math.inf
        val = float(v)  # type: ignore[arg-type]
        if val == -1:
            return math.inf
        if val <= 0:
            raise ValueError(
                "unload_time must be a positive number, -1 (never), or inf"
            )
        return val

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    @field_validator("target_device")
    @classmethod
    def normalize_target_device(cls, v: str) -> str:
        value = v.strip().upper()
        if not value:
            raise ValueError("target_device cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_backend_model_type(self) -> "ModelConfig":
        if self.backend == BackendType.llamacpp and self.model_type != ModelType.llm:
            raise ValueError("llamacpp only supports model_type=llm")
        return self


class ServerConfig(BaseModel):
    """DynLLM proxy server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class BackendConfig(BaseModel):
    """Paths to backend executables."""

    llamacpp_binary: str = "llama-server"
    """Path or name of the llama-server binary."""

    ovms_binary: str = "ovms"
    """Path or name of the OpenVINO Model Server binary."""

    # Port range allocated to backend subprocesses
    port_range_start: int = 9100
    port_range_end: int = 9200


class Settings(BaseModel):
    """Top-level application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    models_dir: Optional[Path] = None
    """Optional base directory for model paths. If set, relative model paths are resolved against this."""

    total_vram_mb: int = Field(default=8192, gt=0)
    """Total GPU VRAM available in megabytes. The proxy will not exceed this."""

    idle_timeout_seconds: int = Field(default=300, gt=0)
    """Seconds of inactivity before a model is automatically unloaded."""

    enabled_backends: list[BackendType] = Field(
        default=[BackendType.llamacpp, BackendType.openvino]
    )
    """Which backends are active. Models whose backend is not listed will be rejected."""

    models: list[ModelConfig] = Field(default_factory=list)

    preload_models: list[str] = Field(default_factory=list)
    """
    Names of models to load automatically on startup.

    Each entry must match a ``name`` field in the ``models`` list.
    Models are loaded in the order listed; VRAM eviction rules apply normally.
    """

    db_path: Path = Field(default=Path("dynllm_state.db"))
    """Path to the SQLite state database."""

    log_level: str = "info"

    @model_validator(mode="after")
    def resolve_model_paths(self) -> "Settings":
        if self.models_dir is not None:
            base = self.models_dir.expanduser().resolve()
            for m in self.models:
                if not m.path.is_absolute():
                    m.path = base / m.path
        return self

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    def model_by_name(self, name: str) -> Optional[ModelConfig]:
        for m in self.models:
            if m.name == name:
                return m
        return None


_settings: Optional[Settings] = None


def load_config(path: Optional[str | Path] = None) -> Settings:
    """Load and validate configuration from a YAML file.

    Resolution order:
    1. ``path`` argument
    2. ``DYNLLM_CONFIG`` environment variable
    3. ``config.yaml`` in the current working directory
    """
    global _settings

    if path is None:
        path = os.environ.get("DYNLLM_CONFIG", "config.yaml")

    config_path = Path(path).expanduser()

    if not config_path.exists():
        # Allow starting with an empty config (no models defined yet).
        _settings = Settings()
        return _settings

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    _settings = Settings.model_validate(raw)
    return _settings


def get_settings() -> Settings:
    """Return the already-loaded settings, raising if not yet loaded."""
    if _settings is None:
        raise RuntimeError(
            "Settings have not been loaded yet. Call load_config() first."
        )
    return _settings
