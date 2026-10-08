"""
OpenVINO Model Server (OVMS) backend.

Spawns one ``ovms`` subprocess per loaded OpenVINO model.  OVMS exposes a REST
API; inference is proxied through its OpenAI-compatible ``/v3/`` endpoints and
the KServe v2 API.

Readiness is detected via the KServe Model Readiness endpoint
(``GET /v2/models/<name>/ready``); audio and image-generation task models, which
do not register a KServe model, are probed through their own endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from pathlib import Path

import httpx

from dynllm.backends.base import SubprocessBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5

# Model types served through the classic OVMS config-file flow (one IR model).
_CONFIG_MODEL_TYPES = frozenset(
    {
        ModelType.embedding,
        ModelType.rerank,
        ModelType.detection,
        ModelType.segmentation,
        ModelType.ocr,
    }
)


def _probe_wav(duration_seconds: float = 1.0) -> bytes:
    """Return a valid mono PCM WAV of silence for audio readiness probing."""
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    num_samples = int(sample_rate * duration_seconds)
    data = b"\x00" * (num_samples * bytes_per_sample)
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    fmt_chunk = struct.pack(
        "<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits_per_sample
    )
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(data)),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            fmt_chunk,
            b"data",
            struct.pack("<I", len(data)),
            data,
        ]
    )


class OpenVINOBackend(SubprocessBackend):
    """Backend that manages OVMS subprocesses for OpenVINO IR models."""

    def __init__(self, binary: str = "ovms") -> None:
        super().__init__(
            binary=binary,
            install_hint=(
                "OVMS binary not found. Install OpenVINO Model Server or set "
                "backend.ovms_binary to an absolute executable path."
            ),
        )

    @property
    def backend_type(self) -> BackendType:
        return BackendType.openvino

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _validate(self, model: ModelConfig) -> None:
        if not Path(model.path).exists():
            raise RuntimeError(f"Model path not found: {model.path}")

    def _build_command(self, model: ModelConfig, port: int, workdir: Path) -> list[str]:
        model_path = Path(model.path)

        if model.model_type == ModelType.llm:
            return self._build_task_command(model, port, model_path, "text_generation")
        if model.model_type in _CONFIG_MODEL_TYPES:
            return self._build_config_command(model, port, model_path, workdir)
        if model.model_type == ModelType.transcription:
            return self._build_task_command(model, port, model_path, "speech2text")
        if model.model_type == ModelType.image_generation:
            return self._build_task_command(model, port, model_path, "image_generation")
        raise RuntimeError(
            f"Unsupported OpenVINO model_type '{model.model_type.value}' "
            f"for '{model.name}'"
        )

    def _build_config_command(
        self, model: ModelConfig, port: int, model_path: Path, workdir: Path
    ) -> list[str]:
        config = {
            "model_config_list": [
                {
                    "config": {
                        "name": model.name,
                        "base_path": str(model_path),
                        "target_device": model.target_device,
                        **({"shape": model.ovms_shape} if model.ovms_shape else {}),
                    }
                }
            ]
        }
        config_file = workdir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))
        return [
            self._binary,
            "--config_path",
            str(config_file),
            "--rest_port",
            str(port),
            "--port",
            "0",
        ]

    def _build_task_command(
        self, model: ModelConfig, port: int, model_path: Path, task: str
    ) -> list[str]:
        if not model_path.is_dir():
            raise RuntimeError(
                f"OpenVINO model '{model.name}' must point to a directory: {model_path}"
            )
        cmd = [
            self._binary,
            "--rest_port",
            str(port),
            "--port",
            "0",
            "--model_repository_path",
            str(model_path.parent),
            "--source_model",
            model_path.name,
            "--model_name",
            model.name,
            "--task",
            task,
            "--target_device",
            model.target_device,
        ]

        optional = [
            (model.tool_parser, "--tool_parser"),
            (model.reasoning_parser, "--reasoning_parser"),
            (model.kv_cache_precision, "--kv_cache_precision"),
            (model.cache_size, "--cache_size"),
            (model.max_num_seqs, "--max_num_seqs"),
            (model.max_num_batched_tokens, "--max_num_batched_tokens"),
            (model.model_distribution_policy, "--model_distribution_policy"),
        ]
        for value, flag in optional:
            if value is not None:
                cmd.extend([flag, str(value)])

        for value, flag in (
            (model.enable_tool_guided_generation, "--enable_tool_guided_generation"),
            (model.enable_prefix_caching, "--enable_prefix_caching"),
            (model.dynamic_split_fuse, "--dynamic_split_fuse"),
        ):
            if value is not None:
                cmd.extend([flag, str(value).lower()])

        if model.draft_model:
            cmd.extend(["--draft_source_model", str(model.draft_model)])

        return cmd

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    async def is_ready(self, port: int, model: ModelConfig, timeout: float) -> bool:
        ready_url = f"http://127.0.0.1:{port}/v2/models/{model.name}/ready"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(ready_url)
                    if resp.status_code == 200:
                        logger.info(
                            "OVMS on port %d: model '%s' is ready", port, model.name
                        )
                        return True
                    if resp.status_code == 404 and await self._task_model_ready(
                        client, port, model
                    ):
                        return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning(
            "OVMS on port %d: model '%s' did not become ready within %.0fs",
            port,
            model.name,
            timeout,
        )
        return False

    async def _task_model_ready(
        self, client: httpx.AsyncClient, port: int, model: ModelConfig
    ) -> bool:
        """Readiness probe for OVMS task models, which skip KServe registration."""
        if model.model_type == ModelType.transcription:
            return await self._transcription_ready(client, port, model.name)
        if model.model_type == ModelType.image_generation:
            path = "v3/images/generations"
        else:
            logger.debug(
                "OVMS port %d: no task readiness probe for model_type '%s'",
                port,
                model.model_type.value,
            )
            return False

        try:
            resp = await client.options(f"http://127.0.0.1:{port}/{path}")
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            return False
        if resp.status_code in (200, 204, 405):
            logger.info(
                "OVMS on port %d: %s model '%s' is ready",
                port,
                model.model_type.value,
                model.name,
            )
            return True
        return False

    async def _transcription_ready(
        self, client: httpx.AsyncClient, port: int, model_name: str
    ) -> bool:
        """Probe the transcription endpoint with a short silent WAV file.

        OVMS audio task models expose no KServe readiness endpoint, so a real
        request detects when the model is loaded.  ``200/202/400/422`` mean the
        endpoint accepts traffic; ``503/404`` mean it is still loading.
        """
        try:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v3/audio/transcriptions",
                files={"file": ("probe.wav", _probe_wav(), "audio/wav")},
                data={"model": model_name},
                timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=5.0),
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            return False
        return resp.status_code in (200, 202, 400, 422)
