"""
OpenVINO Model Server (OVMS) backend.

Spawns one ``ovms`` subprocess per loaded OpenVINO model directory.
OVMS exposes a REST API; inference is proxied via its OpenAI-compatible
``/v3/`` endpoints (introduced in OVMS 2023+).

Readiness is detected via the KServe Model Readiness endpoint:
  GET /v2/models/<model_name>/ready  →  200 when the model is loaded

OVMS is started with a single-model config generated on the fly so each
subprocess serves exactly one model, mirroring the llama-server approach.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Final

import httpx

from dynllm.backends.base import Backend
from dynllm.core.config import BackendType, ModelConfig, ModelType

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5
_AUDIO_READINESS_PATHS: Final[tuple[str, ...]] = ("v3/audio/transcriptions",)
_IMAGE_READINESS_PATHS: Final[tuple[str, ...]] = ("v3/images/generations",)


def _probe_wav(duration_seconds: float = 1.0) -> bytes:
    """Return a valid PCM WAV file of silence for readiness probing.

    The default 1-second clip is long enough to be accepted by ASR backends
    (including Whisper-based ones) while still being very small.
    """
    import struct

    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    num_samples = int(sample_rate * duration_seconds)
    data = b"\x00" * (num_samples * bytes_per_sample)
    data_size = len(data)
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    fmt_chunk = struct.pack(
        "<HHIIHH",
        1,  # format tag (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + data_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            fmt_chunk,
            b"data",
            struct.pack("<I", data_size),
            data,
        ]
    )


def _minimal_wav() -> bytes:
    """Return a tiny, valid WAV file (46 bytes, 16 kHz mono PCM, one sample).

    Kept for backward compatibility; prefer :func:`_probe_wav` for ASR probes.
    """
    return _probe_wav(duration_seconds=1.0 / 16000)


class OpenVINOBackend(Backend):
    """Backend that manages OVMS subprocesses for OpenVINO IR model directories."""

    def __init__(self, binary: str = "ovms") -> None:
        self._binary = self._resolve_binary(binary, "backend.ovms_binary")
        # Keep a reference to temp dirs so they are not GC'd while the server runs
        self._temp_dirs: dict[int, tempfile.TemporaryDirectory] = {}  # pid -> tmpdir

    @property
    def backend_type(self) -> BackendType:
        return BackendType.openvino

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    async def start(self, model: ModelConfig, port: int) -> int:
        """
        Launch OVMS for a single model.

        LLM models use the config-file flow already used by DynLLM. Audio
        models use OVMS task mode with a model repository path pointing at the
        parent directory and the configured model name mapped to the directory.
        """
        model_path = Path(model.path)
        if not model_path.exists():
            raise RuntimeError(f"Model path not found: {model_path}")

        self._binary = self._require_binary(
            self._binary,
            "OVMS binary not found. Install OpenVINO Model Server or set backend.ovms_binary to an absolute executable path.",
        )

        tmpdir = tempfile.TemporaryDirectory(prefix=f"dynllm_ovms_{model.name}_")
        try:
            cmd = self._build_command(model, port, model_path, Path(tmpdir.name))
        except Exception:
            tmpdir.cleanup()
            raise

        logger.info(
            "Starting OVMS for model '%s' on REST port %d: %s",
            model.name,
            port,
            " ".join(cmd),
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Brief wait to detect immediate crashes
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.5)
            _, stderr_bytes = await proc.communicate()
            err = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            tmpdir.cleanup()
            raise RuntimeError(
                f"OVMS exited immediately for model '{model.name}': {err}"
            )
        except asyncio.TimeoutError:
            pass

        # Store temp dir – it must remain alive as long as the subprocess runs
        self._temp_dirs[proc.pid] = tmpdir

        logger.debug("OVMS for '%s' started with PID %d", model.name, proc.pid)
        return proc.pid

    def _build_command(
        self, model: ModelConfig, port: int, model_path: Path, temp_dir: Path
    ) -> list[str]:
        if model.model_type == ModelType.llm:
            if not model_path.is_dir():
                raise RuntimeError(
                    f"OpenVINO LLM model '{model.name}' must point to a directory: {model_path}"
                )
            return self._build_task_command(model, port, model_path, "text_generation")
        if model.model_type in (
            ModelType.embedding,
            ModelType.rerank,
            ModelType.classification,
            ModelType.detection,
            ModelType.segmentation,
            ModelType.ocr,
        ):
            return self._build_llm_command(model, port, model_path, temp_dir)
        if model.model_type == ModelType.transcription:
            return self._build_audio_command(model, port, model_path, "speech2text")
        if model.model_type == ModelType.image_generation:
            return self._build_image_gen_command(model, port, model_path)
        raise RuntimeError(
            f"Unsupported OpenVINO model_type '{model.model_type.value}' for '{model.name}'"
        )

    def _build_llm_command(
        self, model: ModelConfig, port: int, model_path: Path, temp_dir: Path
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

        config_file = temp_dir / "config.json"
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

    def _build_audio_command(
        self, model: ModelConfig, port: int, model_path: Path, task: str
    ) -> list[str]:
        if not model_path.is_dir():
            raise RuntimeError(
                f"OpenVINO audio model '{model.name}' must point to a directory: {model_path}"
            )
        return self._build_task_command(model, port, model_path, task)

    def _build_image_gen_command(
        self, model: ModelConfig, port: int, model_path: Path
    ) -> list[str]:
        if not model_path.is_dir():
            raise RuntimeError(
                f"OpenVINO image generation model '{model.name}' must point to a directory: {model_path}"
            )
        return self._build_task_command(model, port, model_path, "image_generation")

    def _build_task_command(
        self, model: ModelConfig, port: int, model_path: Path, task: str
    ) -> list[str]:
        repository_path = model_path.parent
        source_model = model_path.name
        cmd = [
            self._binary,
            "--rest_port",
            str(port),
            "--port",
            "0",
            "--model_repository_path",
            str(repository_path),
            "--source_model",
            source_model,
            "--model_name",
            model.name,
            "--task",
            task,
            "--target_device",
            model.target_device,
        ]
        if model.tool_parser:
            cmd.extend(["--tool_parser", model.tool_parser])
        if model.reasoning_parser:
            cmd.extend(["--reasoning_parser", model.reasoning_parser])
        if model.enable_tool_guided_generation is not None:
            cmd.extend(
                [
                    "--enable_tool_guided_generation",
                    str(model.enable_tool_guided_generation).lower(),
                ]
            )

        # Speculative decoding
        if model.draft_model:
            cmd.extend(["--draft_source_model", str(model.draft_model)])

        # KV cache optimization
        if model.kv_cache_precision:
            cmd.extend(["--kv_cache_precision", model.kv_cache_precision])
        if model.cache_size is not None:
            cmd.extend(["--cache_size", str(model.cache_size)])
        if model.enable_prefix_caching is not None:
            cmd.extend(
                ["--enable_prefix_caching", str(model.enable_prefix_caching).lower()]
            )

        # Scheduling / Batching
        if model.max_num_seqs is not None:
            cmd.extend(["--max_num_seqs", str(model.max_num_seqs)])
        if model.max_num_batched_tokens is not None:
            cmd.extend(["--max_num_batched_tokens", str(model.max_num_batched_tokens)])
        if model.dynamic_split_fuse is not None:
            cmd.extend(["--dynamic_split_fuse", str(model.dynamic_split_fuse).lower()])

        # Multi-device distribution
        if model.model_distribution_policy:
            cmd.extend(["--model_distribution_policy", model.model_distribution_policy])

        return cmd

    async def stop(self, pid: int) -> None:
        """Terminate the OVMS process with *pid* and clean up its temp config."""
        logger.info("Stopping OVMS PID %d", pid)
        await self._terminate_pid(pid)

        # Clean up temp config directory
        tmpdir = self._temp_dirs.pop(pid, None)
        if tmpdir is not None:
            try:
                tmpdir.cleanup()
            except Exception as exc:
                logger.warning("Failed to clean up temp dir for PID %d: %s", pid, exc)

    async def is_ready(
        self,
        port: int,
        model_name: str = "",
        timeout: float = 120.0,
        model_type: str = "llm",
    ) -> bool:
        """
        Poll ``GET /v2/models/<model_name>/ready`` until OVMS confirms the
        model is ready to serve requests, or the timeout expires.

        This is the KServe Model Readiness endpoint that OVMS exposes:
          - 200  → model is loaded and ready
          - 503  → model is still loading
          - 404  → model unknown (should not happen with a correctly configured name)

        Using this endpoint avoids any inference probe entirely, which
        previously caused false 404 responses due to a wrong model name in
        the probe body.

        OVMS can take a while to load large IR models so we use a generous
        default timeout of 120 s.

        Audio models (transcription / speech) do not register a KServe model
        so their readiness must be detected via the /v3/audio/* endpoints.
        """
        if not model_name:
            logger.warning(
                "OVMS is_ready() called without model_name; falling back to /v1/config"
            )
            ready_url = f"http://127.0.0.1:{port}/v1/config"
        else:
            ready_url = f"http://127.0.0.1:{port}/v2/models/{model_name}/ready"

        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient(timeout=3.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(ready_url)
                    if resp.status_code == 200:
                        logger.info(
                            "OVMS on port %d: model '%s' is ready",
                            port,
                            model_name,
                        )
                        return True
                    if resp.status_code == 404 and model_name and model_type != "llm":
                        if (
                            model_type == "image_generation"
                            and await self._image_endpoints_present(client, port)
                        ):
                            logger.info(
                                "OVMS on port %d: image generation model '%s' is ready",
                                port,
                                model_name,
                            )
                            return True
                        if (
                            model_type == "transcription"
                            and await self._transcription_ready(
                                client, port, model_name
                            )
                        ):
                            logger.info(
                                "OVMS on port %d: transcription model '%s' is ready",
                                port,
                                model_name,
                            )
                            return True
                        if await self._audio_endpoints_present(client, port):
                            logger.info(
                                "OVMS on port %d: audio model '%s' is ready",
                                port,
                                model_name,
                            )
                            return True
                    logger.debug(
                        "OVMS port %d: readiness probe returned %d, retrying…",
                        port,
                        resp.status_code,
                    )
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_POLL_INTERVAL)

        logger.warning(
            "OVMS on port %d: model '%s' did not become ready within %.0fs",
            port,
            model_name,
            timeout,
        )
        return False

    async def _transcription_ready(
        self, client: httpx.AsyncClient, port: int, model_name: str
    ) -> bool:
        """Probe the transcription endpoint with a short silent WAV file.

        OVMS audio models do not expose a KServe readiness endpoint, so we use a
        real request to detect when the model is actually loaded. A 200/202/400/422
        response means the endpoint is accepting traffic; 503/404 means the model
        is still loading. We use a 1-second WAV clip because very short probes can
        be rejected by some ASR pipelines before the model is fully initialised.
        """
        wav = _probe_wav(duration_seconds=1.0)
        try:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v3/audio/transcriptions",
                files={"file": ("probe.wav", wav, "audio/wav")},
                data={"model": model_name},
                timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=5.0),
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            return False
        if resp.status_code in (503, 404):
            return False
        if resp.status_code in (200, 202, 400, 422):
            return True
        logger.debug(
            "Transcription readiness probe on port %d returned unexpected status %d",
            port,
            resp.status_code,
        )
        return False

    async def _audio_endpoints_present(
        self, client: httpx.AsyncClient, port: int
    ) -> bool:
        for path in _AUDIO_READINESS_PATHS:
            try:
                resp = await client.options(f"http://127.0.0.1:{port}/{path}")
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                return False
            if resp.status_code not in (200, 204, 405):
                return False
        return True

    async def _image_endpoints_present(
        self, client: httpx.AsyncClient, port: int
    ) -> bool:
        for path in _IMAGE_READINESS_PATHS:
            try:
                resp = await client.options(f"http://127.0.0.1:{port}/{path}")
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                return False
            if resp.status_code not in (200, 204, 405):
                return False
        return True
