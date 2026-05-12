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
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class BackendType(str, Enum):
    llamacpp = "llamacpp"
    openvino = "openvino"


class ModelType(str, Enum):
    llm = "llm"
    transcription = "transcription"
    speech = "speech"
    embedding = "embedding"
    rerank = "rerank"
    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"
    ocr = "ocr"
    image_generation = "image_generation"


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

    tool_parser: Optional[str] = None
    """
    Type of parser to use for tool calls extraction from model output.
    Only used by the openvino backend for LLM models.
    Supported values: llama3, hermes3, phi4, mistral, gptoss, qwen3coder, devstral, lfm2
    """

    reasoning_parser: Optional[str] = None
    """
    Type of parser to use for reasoning content extraction from model output.
    Only used by the openvino backend for LLM models.
    Supported values: qwen3, gptoss
    """

    enable_tool_guided_generation: Optional[bool] = None
    """
    When enabled, the model will be guided to follow the tool call schema during generation.
    Only used by the openvino backend for LLM models.
    """

    # --- OpenVINO Speculative Decoding ---
    draft_model: Optional[Path] = None
    """
    Path to the draft model directory (OpenVINO IR) for speculative decoding.
    The draft model must share the same tokenizer as the main model.
    A smaller draft model generates token proposals that the main model validates,
    speeding up inference especially at low concurrency.
    Only used by the openvino backend for LLM models.
    """

    draft_model_vram_mb: Optional[int] = Field(default=None, ge=0)
    """
    Additional VRAM consumed by the draft model in megabytes.
    This is added to ``vram_mb`` for eviction decisions.
    Only used when ``draft_model`` is also set.
    """

    # --- OpenVINO KV Cache Optimization ---
    kv_cache_precision: Optional[str] = None
    """
    KV cache precision. Set to ``"u8"`` to reduce KV cache memory consumption
    by using unsigned 8-bit integers instead of the default float32.
    Only used by the openvino backend for LLM models.
    """

    cache_size: Optional[int] = Field(default=None, ge=0)
    """
    KV cache size in gigabytes. Controls how much memory is allocated for
    the KV cache. Default (0) uses dynamic allocation. Recommended to start
    at 10 GB and adjust based on server logs.
    Only used by the openvino backend for LLM models.
    """

    enable_prefix_caching: Optional[bool] = None
    """
    Enable prompt prefix caching. When enabled, repeated prompt prefixes
    (e.g., system prompts) reuse cached KV entries, avoiding redundant
    prefill computation. Enabled by default in OVMS.
    Only used by the openvino backend for LLM models.
    """

    # --- OpenVINO Scheduling / Batching ---
    max_num_seqs: Optional[int] = Field(default=None, ge=1)
    """
    Maximum number of sequences processed simultaneously in a single batch.
    Higher values increase throughput at the cost of more KV cache memory.
    Default in OVMS: 256.
    Only used by the openvino backend for LLM models.
    """

    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1)
    """
    Maximum number of tokens (prefill + decode) that can be batched together
    in a single scheduler step.
    Only used by the openvino backend for LLM models.
    """

    dynamic_split_fuse: Optional[bool] = None
    """
    Enable the dynamic split-fuse scheduling algorithm, which splits prefill
    and decode operations across batches to maximise GPU utilisation.
    Enabled by default in OVMS.
    Only used by the openvino backend for LLM models.
    """

    # --- OpenVINO Multi-device ---
    model_distribution_policy: Optional[str] = None
    """
    Model distribution policy for multi-socket or multi-GPU setups.
    ``"TENSOR_PARALLEL"`` splits individual tensors across devices.
    ``"PIPELINE_PARALLEL"`` distributes different layers across devices.
    Only used by the openvino backend for LLM models.
    """

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

    @field_validator("kv_cache_precision", mode="before")
    @classmethod
    def validate_kv_cache_precision(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        val = str(v)
        if val not in ("u8",):
            raise ValueError(
                f"kv_cache_precision must be 'u8' or None, got '{val}'"
            )
        return val

    @field_validator("model_distribution_policy", mode="before")
    @classmethod
    def validate_model_distribution_policy(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        val = str(v).strip().upper()
        if val not in ("TENSOR_PARALLEL", "PIPELINE_PARALLEL"):
            raise ValueError(
                f"model_distribution_policy must be 'TENSOR_PARALLEL', "
                f"'PIPELINE_PARALLEL', or None, got '{v}'"
            )
        return val

    @field_validator("draft_model", mode="before")
    @classmethod
    def expand_draft_model_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        return Path(str(v)).expanduser()

    @model_validator(mode="after")
    def validate_backend_model_type(self) -> "ModelConfig":
        if self.backend == BackendType.llamacpp and self.model_type not in (
            ModelType.llm,
            ModelType.embedding,
            ModelType.rerank,
        ):
            raise ValueError(
                "llamacpp supports model_type=llm, embedding, and rerank"
            )
        if self.model_type == ModelType.image_generation and self.backend != BackendType.openvino:
            raise ValueError(
                "image_generation is only supported by the openvino backend"
            )
        if self.backend == BackendType.openvino and self.model_type == ModelType.llm:
            if self.draft_model is not None and not self.draft_model.is_dir():
                raise ValueError(
                    f"draft_model must be a directory, got '{self.draft_model}'"
                )
            if self.draft_model_vram_mb is not None and self.draft_model is None:
                raise ValueError(
                    "draft_model_vram_mb requires draft_model to be set"
                )
        else:
            if self.draft_model is not None:
                raise ValueError(
                    "draft_model is only supported for openvino backend with model_type=llm"
                )
            if self.draft_model_vram_mb is not None:
                raise ValueError(
                    "draft_model_vram_mb is only supported for openvino backend with model_type=llm"
                )
            if self.kv_cache_precision is not None:
                raise ValueError(
                    "kv_cache_precision is only supported for openvino backend with model_type=llm"
                )
            if self.cache_size is not None:
                raise ValueError(
                    "cache_size is only supported for openvino backend with model_type=llm"
                )
            if self.enable_prefix_caching is not None:
                raise ValueError(
                    "enable_prefix_caching is only supported for openvino backend with model_type=llm"
                )
            if self.max_num_seqs is not None:
                raise ValueError(
                    "max_num_seqs is only supported for openvino backend with model_type=llm"
                )
            if self.max_num_batched_tokens is not None:
                raise ValueError(
                    "max_num_batched_tokens is only supported for openvino backend with model_type=llm"
                )
            if self.dynamic_split_fuse is not None:
                raise ValueError(
                    "dynamic_split_fuse is only supported for openvino backend with model_type=llm"
                )
            if self.model_distribution_policy is not None:
                raise ValueError(
                    "model_distribution_policy is only supported for openvino backend with model_type=llm"
                )
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
