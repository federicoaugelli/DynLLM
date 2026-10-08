"""
FastAPI router with OpenAI-compatible endpoints.

Inference endpoints:
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/audio/transcriptions
  POST /v1/audio/translations
  POST /v1/audio/speech
  POST /v1/images/generations
  POST /v1/embeddings
  POST /v1/rerank
  GET  /v2/models/{name}            – KServe metadata
  GET  /v2/models/{name}/ready      – KServe readiness
  POST /v2/models/{name}/infer      – KServe inference
  POST /beta/litellm_basic_guardrail_api

Management endpoints (non-standard):
  GET  /admin/models
  POST /admin/models/unload
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from dynllm.api.proxy import forward_request, forward_streaming_request
from dynllm.api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    GuardrailRequest,
    GuardrailResponse,
    ImageGenerationRequest,
    ModelObject,
    ModelStateResponse,
    ModelsResponse,
    RerankRequest,
    SpeechRequest,
    UnloadRequest,
)
from dynllm.backends.privacy_filter import PrivacyFilterBackend
from dynllm.backends.tts import TTSBackend
from dynllm.core.config import BackendType, ModelConfig, ModelType, Settings
from dynllm.core.vram_manager import VRAMManager
from dynllm.db.manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter()

_MULTIPART_MODEL_RE = re.compile(
    rb'name="model"\r\n\r\n(?P<model>[^\r\n]+)', re.IGNORECASE
)
_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vram(request: Request) -> VRAMManager:
    return request.app.state.vram_manager


def get_state(request: Request) -> StateManager:
    return request.app.state.state_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_model(
    settings: Settings, model_name: str, *, expected_types: set[ModelType]
) -> ModelConfig:
    model_cfg = settings.model_by_name(model_name)
    if model_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not configured in DynLLM.",
        )
    if model_cfg.model_type not in expected_types:
        allowed = ", ".join(sorted(mt.value for mt in expected_types))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is configured as "
                f"'{model_cfg.model_type.value}' and cannot serve this endpoint. "
                f"Expected: {allowed}."
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


def _rewrite_backend_model(
    raw_body: bytes, model_cfg: ModelConfig, content_type: str
) -> bytes:
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
    body: bytes | None = None,
) -> Response:
    port = await _ensure_loaded(model_cfg, vram)
    await state.touch(model_cfg.name)
    raw_body = body if body is not None else await request.body()
    proxied_body = _rewrite_backend_model(
        raw_body, model_cfg, request.headers.get("content-type", "").lower()
    )

    await vram.increment_active(model_cfg.name)
    try:
        if stream:
            return await forward_streaming_request(request, port, path, proxied_body)
        return await forward_request(request, port, path, proxied_body)
    finally:
        await vram.decrement_active(model_cfg.name)


def _api_version(model_cfg: ModelConfig) -> str:
    """OVMS serves its OpenAI-compatible endpoints under /v3/."""
    return "v3" if model_cfg.backend == BackendType.openvino else "v1"


# ---------------------------------------------------------------------------
# Model listing and text completions
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(settings: Settings = Depends(get_settings)) -> ModelsResponse:
    return ModelsResponse(
        data=[
            ModelObject(id=m.name, created=int(time.time()), owned_by="dynllm")
            for m in settings.models
        ]
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.llm})
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=f"{_api_version(model_cfg)}/chat/completions",
        stream=body.stream,
    )


@router.post("/v1/completions")
async def completions(
    request: Request,
    body: CompletionRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.llm})
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=f"{_api_version(model_cfg)}/completions",
        stream=body.stream,
    )


# ---------------------------------------------------------------------------
# Audio transcription / translation
# ---------------------------------------------------------------------------


def _multipart_model_name(raw_body: bytes) -> str:
    match = _MULTIPART_MODEL_RE.search(raw_body)
    if match is None:
        raise HTTPException(
            status_code=400, detail="Missing required multipart field 'model'."
        )
    try:
        model_name = match.group("model").decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid multipart model field."
        ) from exc
    if not model_name:
        raise HTTPException(
            status_code=400, detail="Missing required multipart field 'model'."
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
    raw_body = await request.body()
    model_name = _multipart_model_name(raw_body)
    model_cfg = _require_model(
        settings, model_name, expected_types={ModelType.transcription}
    )
    if model_cfg.backend not in (BackendType.openvino, BackendType.transformers):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not use a backend that supports "
                "audio transcription endpoints."
            ),
        )

    # On a cold start the backend may still be finalising its model, so retry
    # transient-looking failures instead of surfacing a 400/404/503.
    already_loaded = await vram.get_port(model_cfg.name) is not None
    max_attempts = 1 if already_loaded else 10

    response: Response | None = None
    for attempt in range(max_attempts):
        response = await _proxy_model_request(
            request,
            body=raw_body,
            model_cfg=model_cfg,
            state=state,
            vram=vram,
            path=path,
        )
        if response.status_code not in (400, 404, 502, 503):
            return response
        if attempt < max_attempts - 1:
            delay = min(2.0**attempt, 15.0)
            logger.debug(
                "Audio request for '%s' returned %d on cold-start attempt %d; "
                "retrying in %.1fs...",
                model_cfg.name,
                response.status_code,
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)

    assert response is not None
    return response


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
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
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    return await _audio_form_request(
        request,
        settings=settings,
        vram=vram,
        state=state,
        path="v3/audio/translations",
    )


# ---------------------------------------------------------------------------
# Audio speech
# ---------------------------------------------------------------------------


@router.post("/v1/audio/speech")
async def audio_speech(
    request: Request,
    body: SpeechRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(settings, body.model, expected_types={ModelType.speech})

    if model_cfg.backend == BackendType.tts:
        await _ensure_loaded(model_cfg, vram)
        await state.touch(model_cfg.name)
        backend = vram.get_instance(model_cfg.name)
        if not isinstance(backend, TTSBackend):
            raise HTTPException(status_code=503, detail="TTS backend not available")
        await vram.increment_active(model_cfg.name)
        try:
            audio = await backend.synthesize(
                text=body.input,
                voice=body.voice,
                response_format=body.response_format or "wav",
                speed=body.speed or 1.0,
            )
        finally:
            await vram.decrement_active(model_cfg.name)
        media_type = _MEDIA_TYPES.get(body.response_format or "wav", "audio/wav")
        return Response(content=audio, media_type=media_type)

    if model_cfg.backend == BackendType.openvino:
        return await _proxy_model_request(
            request,
            model_cfg=model_cfg,
            state=state,
            vram=vram,
            path="v3/audio/speech",
        )

    if model_cfg.backend == BackendType.transformers:
        return await _proxy_model_request(
            request,
            model_cfg=model_cfg,
            state=state,
            vram=vram,
            path="v1/audio/speech",
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Model '{body.model}' does not use a backend that supports audio "
            "speech endpoints."
        ),
    )


# ---------------------------------------------------------------------------
# Image generation, embeddings, rerank
# ---------------------------------------------------------------------------


@router.post("/v1/images/generations")
async def image_generations(
    request: Request,
    body: ImageGenerationRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(
        settings, body.model, expected_types={ModelType.image_generation}
    )
    if model_cfg.backend != BackendType.openvino:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{body.model}' does not use the OpenVINO backend. "
                "Image generation is only supported via OVMS."
            ),
        )
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path="v3/images/generations",
    )


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(
        settings, body.model, expected_types={ModelType.embedding, ModelType.llm}
    )
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path=f"{_api_version(model_cfg)}/embeddings",
    )


@router.post("/v1/rerank")
async def rerank(
    request: Request,
    body: RerankRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    model_cfg = _require_model(
        settings, body.model, expected_types={ModelType.rerank, ModelType.llm}
    )
    return await _proxy_model_request(
        request,
        model_cfg=model_cfg,
        state=state,
        vram=vram,
        path="v1/rerank",
    )


# ---------------------------------------------------------------------------
# KServe v2 passthrough (OpenVINO only)
# ---------------------------------------------------------------------------

_KSERVE_MODEL_TYPES = {
    ModelType.llm,
    ModelType.embedding,
    ModelType.rerank,
    ModelType.classification,
    ModelType.detection,
    ModelType.segmentation,
    ModelType.ocr,
}


async def _kserve_proxy(
    request: Request,
    name: str,
    kserve_path: str,
    settings: Settings,
    vram: VRAMManager,
    state: StateManager,
) -> Response:
    model_cfg = _require_model(settings, name, expected_types=_KSERVE_MODEL_TYPES)
    if model_cfg.backend != BackendType.openvino:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{name}' is backed by '{model_cfg.backend.value}', which "
                "does not expose a KServe API. Only OpenVINO models support "
                "KServe endpoints."
            ),
        )
    port = await _ensure_loaded(model_cfg, vram)
    await state.touch(name)
    raw_body = await request.body()
    await vram.increment_active(name)
    try:
        return await forward_request(request, port, kserve_path, raw_body)
    finally:
        await vram.decrement_active(name)


@router.get("/v2/models/{name}")
async def kserve_model_metadata(
    request: Request,
    name: str,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    return await _kserve_proxy(
        request, name, f"v2/models/{name}", settings, vram, state
    )


@router.get("/v2/models/{name}/ready")
async def kserve_model_ready(
    request: Request,
    name: str,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    return await _kserve_proxy(
        request, name, f"v2/models/{name}/ready", settings, vram, state
    )


@router.post("/v2/models/{name}/infer")
async def kserve_model_infer(
    request: Request,
    name: str,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> Response:
    return await _kserve_proxy(
        request, name, f"v2/models/{name}/infer", settings, vram, state
    )


# ---------------------------------------------------------------------------
# litellm Generic Guardrail API
# ---------------------------------------------------------------------------


@router.post("/beta/litellm_basic_guardrail_api")
async def litellm_guardrail(
    body: GuardrailRequest,
    settings: Settings = Depends(get_settings),
    vram: VRAMManager = Depends(get_vram),
    state: StateManager = Depends(get_state),
) -> GuardrailResponse:
    """PII masking for litellm's ``generic_guardrail_api`` guardrail."""
    if body.input_type != "request" or not body.texts:
        return GuardrailResponse(action="NONE")

    params = body.additional_provider_specific_params or {}
    model_name = params.get("model")
    if model_name:
        model_cfg = settings.model_by_name(model_name)
    else:
        model_cfg = next(
            (m for m in settings.models if m.model_type == ModelType.classification),
            None,
        )

    if model_cfg is None or model_cfg.backend != BackendType.privacy_filter:
        return GuardrailResponse(action="NONE")

    await _ensure_loaded(model_cfg, vram)
    await state.touch(model_cfg.name)
    backend = vram.get_instance(model_cfg.name)
    if not isinstance(backend, PrivacyFilterBackend):
        return GuardrailResponse(action="NONE")

    masked_texts: list[str] = []
    any_changed = False
    for text in body.texts:
        await vram.increment_active(model_cfg.name)
        try:
            result = await backend.filter_text(
                text,
                mask_strategy=params.get("mask_strategy", "replace"),
                categories=params.get("categories"),
            )
        finally:
            await vram.decrement_active(model_cfg.name)
        masked_texts.append(result["masked_text"])
        any_changed = any_changed or result["masked_text"] != text

    if not any_changed:
        return GuardrailResponse(action="NONE")
    return GuardrailResponse(action="GUARDRAIL_INTERVENED", texts=masked_texts)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@router.get("/admin/models", response_model=list[ModelStateResponse])
async def admin_list_models(
    state: StateManager = Depends(get_state),
) -> list[ModelStateResponse]:
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
    body: UnloadRequest, vram: VRAMManager = Depends(get_vram)
) -> JSONResponse:
    try:
        await vram.unload(body.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"status": "ok", "model": body.model})
