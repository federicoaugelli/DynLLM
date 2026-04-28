"""
FastAPI router with OpenAI-compatible endpoints.

Endpoints:
  GET  /v1/models                    – list configured models
  POST /v1/chat/completions          – chat completions (streaming + non-streaming)
  POST /v1/completions               – text completions (streaming + non-streaming)

Management endpoints (non-standard):
  GET  /admin/models                 – detailed model state
  POST /admin/models/unload          – manually unload a model
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from dynllm.api.proxy import forward_request, forward_streaming_request
from dynllm.api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelObject,
    ModelStateResponse,
    ModelsResponse,
    SpeechRequest,
    UnloadRequest,
)
from dynllm.core.config import (
    BackendType,
    ModelConfig,
    ModelType,
    Settings,
    get_settings,
)
from dynllm.core.vram_manager import VRAMManager, decrement_active, increment_active
from dynllm.db.manager import StateManager

logger = logging.getLogger(__name__)
_MULTIPART_MODEL_RE = re.compile(
    rb'name="model"\r\n\r\n(?P<model>[^\r\n]+)',
    re.IGNORECASE,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------

# These are set by the application factory on startup.
_vram_manager: VRAMManager | None = None
_state_manager: StateManager | None = None


def set_managers(vram: VRAMManager, state: StateManager) -> None:
    global _vram_manager, _state_manager
    _vram_manager = vram
    _state_manager = state


def _get_vram() -> VRAMManager:
    if _vram_manager is None:
        raise RuntimeError("VRAMManager not initialised")
    return _vram_manager


def _get_state() -> StateManager:
    if _state_manager is None:
        raise RuntimeError("StateManager not initialised")
    return _state_manager


def _require_model(
    settings: Settings,
    model_name: str,
    *,
    expected_types: set[ModelType],
) -> ModelConfig:
    model_cfg = settings.model_by_name(model_name)
    if model_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not configured in DynLLM.",
        )
    if model_cfg.model_type not in expected_types:
        allowed = ", ".join(sorted(model_type.value for model_type in expected_types))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is configured as '{model_cfg.model_type.value}' "
                f"and cannot serve this endpoint. Expected: {allowed}."
            ),
        )
    return model_cfg


async def _ensure_loaded(model_cfg: ModelConfig, vram: VRAMManager) -> int:
    try:
        return await vram.ensure_loaded(model_cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _backend_request_model(model_cfg: ModelConfig) -> str:
    if model_cfg.backend == BackendType.transformers:
        return str(model_cfg.path)
    return model_cfg.name


def _rewrite_backend_model(raw_body: bytes, model_cfg: ModelConfig, content_type: str) -> bytes:
    backend_model = _backend_request_model(model_cfg)
    if backend_model == model_cfg.name:
        return raw_body

    if "application/json" in content_type:
        payload = json.loads(raw_body.decode("utf-8"))
        payload["model"] = backend_model
        return json.dumps(payload).encode("utf-8")

    if "multipart/form-data" in content_type:
        return _MULTIPART_MODEL_RE.sub(
            lambda match: match.group(0).replace(
                match.group("model"), backend_model.encode("utf-8")
            ),
            raw_body,
            count=1,
        )

    return raw_body


async def _proxy_model_request(
    request: Request,
    *,
    model_cfg: ModelConfig,
    state: StateManager,
    vram: VRAMManager,
    path: str,
    stream: bool = False,
) -> Response:
    port = await _ensure_loaded(model_cfg, vram)
    await state.touch(model_cfg.name)
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "").lower()
    proxied_body = _rewrite_backend_model(raw_body, model_cfg, content_type)

    await increment_active(model_cfg.name)
    try:
        if stream:
            return await forward_streaming_request(request, port, path, proxied_body)
        return await forward_request(request, port, path, proxied_body)
    finally:
        await decrement_active(model_cfg.name)


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    """Return the list of models defined in the config."""
    objects = [
        ModelObject(
            id=m.name,
            created=int(time.time()),
            owned_by="dynllm",
        )
        for m in settings.models
    ]
    return ModelsResponse(data=objects)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(_get_vram),
    state: StateManager = Depends(_get_state),
) -> Response:
    """
    Proxy a chat completion request to the appropriate backend.

    Loads the model on-demand if it is not already in VRAM.
    Supports both streaming (SSE) and non-streaming responses.
    """
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.llm})
    # OVMS uses /v3/ for its OpenAI-compatible endpoints; llama-server uses /v1/
    api_version = "v3" if model_cfg.backend == BackendType.openvino else "v1"
    path = f"{api_version}/chat/completions"
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=path,
        stream=body.stream,
    )


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


@router.post("/v1/completions")
async def completions(
    request: Request,
    body: CompletionRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(_get_vram),
    state: StateManager = Depends(_get_state),
) -> Response:
    """
    Proxy a text completion request to the appropriate backend.
    """
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.llm})
    # OVMS uses /v3/ for its OpenAI-compatible endpoints; llama-server uses /v1/
    api_version = "v3" if model_cfg.backend == BackendType.openvino else "v1"
    path = f"{api_version}/completions"
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=path,
        stream=body.stream,
    )


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions and translations
# ---------------------------------------------------------------------------


async def _audio_form_model_name(request: Request) -> str:
    raw_body = await request.body()
    match = _MULTIPART_MODEL_RE.search(raw_body)
    if match is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required multipart field 'model'.",
        )
    try:
        model_name = match.group("model").decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid multipart model field."
        ) from exc
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Missing required multipart field 'model'.",
        )
    return model_name


async def _audio_form_request(
    request: Request,
    *,
    settings: Settings,
    vram: VRAMManager,
    state: StateManager,
    path: str,
) -> Response:
    model_name = await _audio_form_model_name(request)
    model_cfg = _require_model(
        settings,
        model_name,
        expected_types={ModelType.transcription},
    )
    if model_cfg.backend not in (BackendType.openvino, BackendType.transformers):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not use a backend that supports "
                "audio transcription endpoints."
            ),
        )
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=path,
    )


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(_get_vram),
    state: StateManager = Depends(_get_state),
) -> Response:
    return await _audio_form_request(
        request,
        settings=settings,
        vram=vram,
        state=state,
        path="v3/audio/transcriptions",
    )


@router.post("/v1/audio/translations")
async def audio_translations(
    request: Request,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(_get_vram),
    state: StateManager = Depends(_get_state),
) -> Response:
    return await _audio_form_request(
        request,
        settings=settings,
        vram=vram,
        state=state,
        path="v3/audio/translations",
    )


# ---------------------------------------------------------------------------
# /v1/audio/speech
# ---------------------------------------------------------------------------


@router.post("/v1/audio/speech")
async def audio_speech(
    request: Request,
    body: SpeechRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(_get_vram),
    state: StateManager = Depends(_get_state),
) -> Response:
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.speech})
    if model_cfg.backend != BackendType.openvino:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{body.model}' does not use the OpenVINO backend.",
        )
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path="v3/audio/speech",
    )


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@router.get("/admin/models", response_model=list[ModelStateResponse])
async def admin_list_models(
    state: StateManager = Depends(_get_state),
) -> list[ModelStateResponse]:
    """Return detailed state for every known model."""
    rows = await state.get_all()
    return [
        ModelStateResponse(
            name=r.name,
            status=r.status.value,
            backend=r.backend,
            vram_mb=r.vram_mb,
            port=r.port,
            pid=r.pid,
            loaded_at=r.loaded_at.isoformat() if r.loaded_at else None,
            last_used_at=r.last_used_at.isoformat() if r.last_used_at else None,
        )
        for r in rows
    ]


@router.post("/admin/models/unload")
async def admin_unload_model(
    body: UnloadRequest,
    vram: VRAMManager = Depends(_get_vram),
) -> JSONResponse:
    """Manually unload a model from VRAM."""
    try:
        await vram.unload(body.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"status": "ok", "model": body.model})
