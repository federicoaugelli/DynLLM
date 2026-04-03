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
    UnloadRequest,
)
from dynllm.core.config import BackendType, Settings, get_settings
from dynllm.core.vram_manager import VRAMManager, decrement_active, increment_active
from dynllm.db.manager import StateManager

logger = logging.getLogger(__name__)

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
    model_cfg = settings.model_by_name(body.model)
    if model_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' is not configured in DynLLM.",
        )

    try:
        port = await vram.ensure_loaded(model_cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Update last_used_at
    await state.touch(body.model)

    raw_body = await request.body()
    # OVMS uses /v3/ for its OpenAI-compatible endpoints; llama-server uses /v1/
    api_version = "v3" if model_cfg.backend == BackendType.openvino else "v1"
    path = f"{api_version}/chat/completions"

    await increment_active(body.model)
    try:
        if body.stream:
            return await forward_streaming_request(request, port, path, raw_body)
        else:
            return await forward_request(request, port, path, raw_body)
    finally:
        await decrement_active(body.model)


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
    model_cfg = settings.model_by_name(body.model)
    if model_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' is not configured in DynLLM.",
        )

    try:
        port = await vram.ensure_loaded(model_cfg)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    await state.touch(body.model)

    raw_body = await request.body()
    # OVMS uses /v3/ for its OpenAI-compatible endpoints; llama-server uses /v1/
    api_version = "v3" if model_cfg.backend == BackendType.openvino else "v1"
    path = f"{api_version}/completions"

    await increment_active(body.model)
    try:
        if body.stream:
            return await forward_streaming_request(request, port, path, raw_body)
        else:
            return await forward_request(request, port, path, raw_body)
    finally:
        await decrement_active(body.model)


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
