"""
Configuration system for DynLLM.

Config is loaded from a YAML file (default: ``config.yaml`` in the working
directory, overridable via the ``DYNLLM_CONFIG`` env var).
"""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from dynllm.core.builtins import BUILTIN_DEFAULTS


class BackendType(str, Enum):
    llamacpp = "llamacpp"
    openvino = "openvino"
    transformers = "transformers"
    tts = "tts"
    privacy_filter = "privacy_filter"
    neutronstar = "neutronstar"
    laya = "laya"


class ModelType(str, Enum):
    llm = "llm"
    decision = "decision"
    transcription = "transcription"
    speech = "speech"
    embedding = "embedding"
    rerank = "rerank"
    classification = "classification"
    detection = "detection"
    segmentation = "segmentation"
    ocr = "ocr"
    image_generation = "image_generation"


class TransformersQuantization(str, Enum):
    none = "none"
    bnb_4bit = "bnb-4bit"
    bnb_8bit = "bnb-8bit"


class TransformersAttentionImplementation(str, Enum):
    auto = "auto"
    eager = "eager"
    sdpa = "sdpa"
    flash_attention_2 = "flash_attention_2"
    flash_attention_3 = "flash_attention_3"
    flex_attention = "flex_attention"


# Which model types each backend can serve (order is used in error messages).
_SUPPORTED_MODEL_TYPES: dict[BackendType, tuple[ModelType, ...]] = {
    BackendType.llamacpp: (
        ModelType.llm,
        ModelType.decision,
        ModelType.embedding,
        ModelType.rerank,
    ),
    BackendType.openvino: (
        ModelType.llm,
        ModelType.transcription,
        ModelType.embedding,
        ModelType.rerank,
        ModelType.detection,
        ModelType.segmentation,
        ModelType.ocr,
        ModelType.image_generation,
    ),
    BackendType.transformers: (
        ModelType.llm,
        ModelType.transcription,
        ModelType.speech,
    ),
    BackendType.tts: (ModelType.speech,),
    BackendType.privacy_filter: (ModelType.classification,),
    BackendType.neutronstar: (ModelType.llm,),
    BackendType.laya: (ModelType.decision,),
}

# Fields that only make sense for the in-process Laya backend.  The mapping
# stores the value used by ``ModelConfig`` when the field is untouched.
_LAYA_ONLY_FIELDS: dict[str, object] = {
    "laya_model": None,
    "laya_preload": False,
    "laya_preload_models": ["english", "multilingual"],
    "laya_max_loaded": 2,
    "laya_default_model": None,
}

_LAYA_DEVICES = frozenset({"auto", "cpu", "cuda", "xpu", "mps"})

# Fields that only make sense for an OpenVINO LLM (speculative decoding,
# KV-cache and scheduling knobs).
_OVMS_LLM_ONLY_FIELDS = (
    "draft_model",
    "draft_model_vram_mb",
    "kv_cache_precision",
    "cache_size",
    "enable_prefix_caching",
    "max_num_seqs",
    "max_num_batched_tokens",
    "dynamic_split_fuse",
    "model_distribution_policy",
)


def _join(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"


class ModelConfig(BaseModel):
    """Declaration of a single model available to the proxy."""

    name: str
    """Unique model identifier used in API requests (e.g. 'llama3-8b')."""

    builtin: Optional[str] = None
    """
    Name of a built-in preset (e.g. ``"laya"``).

    When set, missing fields are filled from
    :data:`dynllm.core.builtins.BUILTIN_DEFAULTS`.  Naming a model after a
    preset (``- name: laya``) implies ``builtin: laya`` unless a different
    backend or an explicit path is declared.
    """

    path: Path
    """Absolute or relative path to the model directory/file."""

    backend: BackendType
    """Which backend should serve this model."""

    model_type: ModelType = ModelType.llm
    """What kind of workload this model serves."""

    vram_mb: int = Field(ge=0)
    """Estimated VRAM this model consumes when loaded, in megabytes."""

    target_device: str = "CPU"
    """Execution device (OpenVINO CPU/GPU/NPU; TTS xpu/cpu)."""

    # --- llama.cpp specific ---
    n_gpu_layers: int = -1
    """Number of layers to offload to GPU (-1 = all). llama.cpp only."""

    context_size: int = 4096
    """Context window size. llama.cpp only."""

    mmproj: Optional[Path] = None
    """
    Path to a multimodal projector GGUF. llama.cpp only.

    Required for decision models that read images/audio (e.g. Clef).  Passed to
    llama-server as ``--mmproj``.
    """

    extra_args: list[str] = Field(default_factory=list)
    """
    Additional raw arguments appended to the backend server command.
    llama.cpp and NeutronStar only.  Useful for flags DynLLM does not model
    explicitly (e.g. ``["--batch-size", "2048"]``).
    """

    # --- OVMS specific ---
    ovms_shape: Optional[str] = None
    """Optional shape hint for OVMS (e.g. 'auto'). OpenVINO only."""

    # --- transformers specific ---
    device: str = "auto"
    """Transformers device ('auto', 'cpu', 'cuda', 'xpu')."""

    dtype: str = "auto"
    """Transformers dtype ('auto', 'float16', 'bfloat16', 'float32')."""

    quantization: TransformersQuantization = TransformersQuantization.none
    """Transformers quantization ('none', 'bnb-4bit', 'bnb-8bit')."""

    trust_remote_code: bool = False
    """Allow custom model code execution in transformers."""

    compile_model: bool = False
    """Enable torch.compile through transformers serve."""

    continuous_batching: bool = False
    """Enable transformers continuous batching for supported LLMs."""

    attn_implementation: TransformersAttentionImplementation = (
        TransformersAttentionImplementation.auto
    )
    """Attention backend for transformers serve."""

    model_timeout: Optional[int] = Field(default=None, gt=0)
    """Backend-side idle timeout in seconds for transformers."""

    revision: Optional[str] = None
    """Optional Hugging Face revision rendered as ``model@revision``."""

    # --- TTS engine specific ---
    tts_engine: str = ""
    """TTS engine name ('qwen', 'supertonic'). Required when backend=tts."""

    # --- Laya decision model (in-process) ---
    laya_model: Optional[str] = None
    """
    Pin a Laya checkpoint (``english``, ``multilingual``, ``typed-decisions``)
    or leave ``None`` to let the Router choose per request.
    """

    laya_preload: bool = False
    """
    Build the configured checkpoints during load instead of on the first
    request.  Lazy loading (the default) keeps startup fast.
    """

    laya_preload_models: list[str] = Field(
        default_factory=lambda: ["english", "multilingual"]
    )
    """Checkpoints preloaded when ``laya_preload`` is true."""

    laya_max_loaded: int = Field(default=2, ge=1)
    """Checkpoints kept resident at once (Laya's default is 2)."""

    laya_default_model: Optional[str] = None
    """Fallback checkpoint for text without usable language evidence."""

    # --- NeutronStar specific ---
    ns_max_seq: Optional[int] = Field(default=None, gt=0)
    """KV capacity passed to ns-server as ``--max-seq`` (server default: 16384)."""

    ns_env: dict[str, str] = Field(default_factory=dict)
    """``NS3_*`` tuning variables exported to the ns-server subprocess."""

    # --- OpenVINO LLM tool calling / reasoning ---
    tool_parser: Optional[str] = None
    """Tool-call parser (llama3, hermes3, phi4, mistral, gptoss, qwen3coder, devstral, lfm2)."""

    reasoning_parser: Optional[str] = None
    """Reasoning-content parser (qwen3, gptoss)."""

    enable_tool_guided_generation: Optional[bool] = None
    """Guide generation to follow the tool-call schema."""

    # --- OpenVINO LLM speculative decoding ---
    draft_model: Optional[Path] = None
    """Path to an OpenVINO IR draft model for speculative decoding."""

    draft_model_vram_mb: Optional[int] = Field(default=None, ge=0)
    """Extra VRAM (MB) used by the draft model; added to ``vram_mb``."""

    # --- OpenVINO LLM KV cache optimization ---
    kv_cache_precision: Optional[str] = None
    """KV cache precision; ``"u8"`` halves KV cache memory."""

    cache_size: Optional[int] = Field(default=None, ge=0)
    """Fixed KV cache size in GB (default: dynamic)."""

    enable_prefix_caching: Optional[bool] = None
    """Cache repeated prompt prefixes (default: enabled in OVMS)."""

    # --- OpenVINO LLM scheduling / batching ---
    max_num_seqs: Optional[int] = Field(default=None, ge=1)
    """Max sequences processed simultaneously (default in OVMS: 256)."""

    max_num_batched_tokens: Optional[int] = Field(default=None, ge=1)
    """Max tokens (prefill + decode) batched in a single scheduler step."""

    dynamic_split_fuse: Optional[bool] = None
    """Split prefill/decode across batches (default: enabled in OVMS)."""

    # --- OpenVINO multi-device ---
    model_distribution_policy: Optional[str] = None
    """``"TENSOR_PARALLEL"`` or ``"PIPELINE_PARALLEL"`` for multi-device setups."""

    # --- Idle unload ---
    unload_time: Optional[float] = None
    """
    Per-model idle timeout in seconds before automatic unload.

    ``null`` inherits the global ``idle_timeout_seconds``; a positive number is
    used as-is; ``-1``/``inf`` means never auto-unload.
    """

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def apply_builtin_defaults(cls, data: Any) -> Any:
        """Fill a built-in preset before normal field validation.

        Inference by name is intentionally conservative: it only applies when
        the declaration names a preset, does not point at an explicit ``path``
        and does not select a different backend.  That keeps a user model that
        happens to be called ``laya`` but points at a GGUF untouched.
        """
        if not isinstance(data, dict):
            return data

        explicit = data.get("builtin")
        name = data.get("name")
        preset_name = explicit or name
        if not preset_name:
            return data

        defaults = BUILTIN_DEFAULTS.get(preset_name)
        if defaults is None:
            if explicit:
                known = ", ".join(sorted(BUILTIN_DEFAULTS))
                raise ValueError(
                    f"unknown builtin '{explicit}'; available built-ins: {known}"
                )
            return data

        if explicit is None and "path" not in data:
            # A custom backend or a model that only repeats the preset backend
            # still benefits from the preset defaults.  An explicit ``path``
            # means the user is pointing at their own checkpoint.
            if data.get("backend") not in (None, defaults["backend"]):
                return data
        elif explicit is None:
            return data

        merged = dict(defaults)
        merged.update({key: value for key, value in data.items() if value is not None})
        merged["builtin"] = preset_name
        return merged

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

    @field_validator("draft_model", mode="before")
    @classmethod
    def expand_draft_model_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        return Path(str(v)).expanduser()

    @field_validator("mmproj", mode="before")
    @classmethod
    def expand_mmproj_path(cls, v: object) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return Path(str(v)).expanduser()

    @field_validator("target_device")
    @classmethod
    def normalize_target_device(cls, v: str) -> str:
        value = v.strip().upper()
        if not value:
            raise ValueError("target_device cannot be empty")
        return value

    @field_validator("device", "dtype")
    @classmethod
    def normalize_lowercase(cls, v: str) -> str:
        value = v.strip().lower()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("laya_model", "laya_default_model")
    @classmethod
    def normalize_optional_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        return value or None

    @field_validator("laya_preload_models")
    @classmethod
    def normalize_preload_models(cls, v: list[str]) -> list[str]:
        names = [name.strip() for name in v if name.strip()]
        return names or ["english", "multilingual"]

    @field_validator("revision")
    @classmethod
    def normalize_revision(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        if not value:
            raise ValueError("revision cannot be empty")
        if "@" in value:
            raise ValueError("revision must not contain '@'; configure it separately")
        return value

    @field_validator("kv_cache_precision", mode="before")
    @classmethod
    def validate_kv_cache_precision(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        val = str(v)
        if val != "u8":
            raise ValueError(f"kv_cache_precision must be 'u8' or None, got '{val}'")
        return val

    @field_validator("model_distribution_policy", mode="before")
    @classmethod
    def validate_model_distribution_policy(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        val = str(v).strip().upper()
        if val not in ("TENSOR_PARALLEL", "PIPELINE_PARALLEL"):
            raise ValueError(
                "model_distribution_policy must be 'TENSOR_PARALLEL', "
                f"'PIPELINE_PARALLEL', or None, got '{v}'"
            )
        return val

    @model_validator(mode="after")
    def validate_consistency(self) -> "ModelConfig":
        supported = _SUPPORTED_MODEL_TYPES[self.backend]
        if self.model_type not in supported:
            if (
                self.backend == BackendType.openvino
                and self.model_type == ModelType.speech
            ):
                raise ValueError(
                    "openvino no longer supports model_type=speech; use backend=tts instead"
                )
            allowed = _join([mt.value for mt in supported])
            raise ValueError(f"{self.backend.value} supports model_type={allowed}")

        if self.backend == BackendType.tts:
            if not self.tts_engine:
                raise ValueError(
                    "tts backend requires 'tts_engine' field to be set "
                    "(e.g. 'qwen', 'supertonic')"
                )
        elif self.tts_engine:
            raise ValueError("tts_engine is only valid for backend=tts")

        is_ovms_llm = (
            self.backend == BackendType.openvino and self.model_type == ModelType.llm
        )
        if not is_ovms_llm:
            for field in _OVMS_LLM_ONLY_FIELDS:
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"{field} is only supported for openvino backend with "
                        "model_type=llm"
                    )
        elif self.draft_model_vram_mb is not None and self.draft_model is None:
            raise ValueError("draft_model_vram_mb requires draft_model to be set")

        if self.backend != BackendType.llamacpp:
            if self.mmproj is not None:
                raise ValueError("mmproj is only supported for backend=llamacpp")

        if self.backend not in (BackendType.llamacpp, BackendType.neutronstar):
            if self.extra_args:
                raise ValueError(
                    "extra_args is only supported for backend=llamacpp and "
                    "backend=neutronstar"
                )

        if self.backend != BackendType.neutronstar and (
            self.ns_max_seq is not None or self.ns_env
        ):
            raise ValueError(
                "ns_max_seq and ns_env are only supported for backend=neutronstar"
            )

        if self.quantization != TransformersQuantization.none:
            if (
                self.backend != BackendType.transformers
                or self.model_type != ModelType.llm
            ):
                raise ValueError(
                    "transformers quantization is currently supported only for "
                    "model_type=llm"
                )
            if self.dtype not in ("auto", "float16", "bfloat16"):
                raise ValueError(
                    "transformers quantization requires dtype to be auto, "
                    "float16, or bfloat16"
                )

        if self.builtin:
            preset = BUILTIN_DEFAULTS.get(self.builtin)
            if preset is not None and self.backend.value != preset["backend"]:
                raise ValueError(
                    f"builtin '{self.builtin}' requires backend={preset['backend']}"
                )

        if self.backend == BackendType.laya:
            if self.device not in _LAYA_DEVICES:
                raise ValueError(
                    "laya device must be one of: " + ", ".join(sorted(_LAYA_DEVICES))
                )
            if self.device != "cpu" and self.vram_mb == 0:
                raise ValueError(
                    "laya on device='%s' requires vram_mb > 0 (or use device: cpu)"
                    % self.device
                )
        else:
            for field_name, default in _LAYA_ONLY_FIELDS.items():
                if getattr(self, field_name) != default:
                    raise ValueError(f"{field_name} is only supported for backend=laya")

        return self


class ServerConfig(BaseModel):
    """DynLLM proxy server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class BackendConfig(BaseModel):
    """Backend executables and the port range for subprocesses."""

    llamacpp_binary: str = "llama-server"
    ovms_binary: str = "ovms"
    transformers_binary: str = "transformers"
    neutronstar_binary: str = "ns-server"

    port_range_start: int = 9100
    port_range_end: int = 9200


class Settings(BaseModel):
    """Top-level application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    models_dir: Optional[Path] = None
    """Base directory for relative model paths."""

    total_vram_mb: int = Field(default=8192, gt=0)
    """Total GPU VRAM available in megabytes."""

    idle_timeout_seconds: int = Field(default=300, gt=0)
    """Seconds of inactivity before a model is automatically unloaded."""

    enabled_backends: list[BackendType] = Field(
        default_factory=lambda: [BackendType.llamacpp, BackendType.openvino]
    )
    """Active backends. Models whose backend is not listed are rejected."""

    models: list[ModelConfig] = Field(default_factory=list)

    preload_models: list[str] = Field(default_factory=list)
    """Names of models to load automatically on startup."""

    auto_install: bool = True
    """
    Install missing Python packages required by in-process built-in models
    (currently ``laya``) at startup.  Set to ``false`` for read-only or
    air-gapped deployments.
    """

    db_path: Path = Field(default=Path("dynllm_state.db"))
    """Path to the SQLite state database."""

    log_level: str = "info"

    @model_validator(mode="after")
    def resolve_model_paths(self) -> "Settings":
        if self.models_dir is not None:
            base = self.models_dir.expanduser().resolve()
            for model in self.models:
                if not model.path.is_absolute():
                    model.path = base / model.path
                if model.mmproj is not None and not model.mmproj.is_absolute():
                    model.mmproj = base / model.mmproj

        # Built-in models bring their backend with them: requiring the user to
        # repeat ``laya`` in enabled_backends defeats the point of a preset.
        for model in self.models:
            if model.builtin and model.backend not in self.enabled_backends:
                self.enabled_backends.append(model.backend)

        return self

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()

    def model_by_name(self, name: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.name == name), None)


def load_config(path: Optional[str | Path] = None) -> Settings:
    """Load and validate configuration from a YAML file.

    Resolution order: the ``path`` argument, ``DYNLLM_CONFIG``, then
    ``config.yaml`` in the current working directory.  A missing file yields an
    empty configuration (no models defined).
    """
    if path is None:
        path = os.environ.get("DYNLLM_CONFIG", "config.yaml")

    config_path = Path(path).expanduser()
    if not config_path.exists():
        return Settings()

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    return Settings.model_validate(raw)
