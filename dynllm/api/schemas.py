"""
OpenAI-compatible request/response Pydantic schemas.

Only the fields that DynLLM actually uses/proxies are declared.
Unknown extra fields are forwarded transparently to the backend.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "dynllm"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[Any], None] = None
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Text completions (/v1/completions)
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list[str]]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    seed: Optional[int] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Audio endpoints
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    instructions: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Image generation (/v1/images/generations)
# ---------------------------------------------------------------------------


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    response_format: Optional[str] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Embeddings (/v1/embeddings)
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str], list[list[int]]]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Reranking (/v1/rerank)
# ---------------------------------------------------------------------------

# Cohere-compatible rerank request (also supported by llama.cpp & OVMS)


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    max_chunks_per_doc: Optional[int] = None
    return_documents: Optional[bool] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Admin / management (non-OpenAI endpoints)
# ---------------------------------------------------------------------------


class ModelStateResponse(BaseModel):
    name: str
    status: str
    backend: str
    vram_mb: int
    port: Optional[int]
    pid: Optional[int]
    loaded_at: Optional[str]
    last_used_at: Optional[str]


class UnloadRequest(BaseModel):
    model: str


# ---------------------------------------------------------------------------
# Privacy filter (/v1/guardrails/privacy-filter)
# ---------------------------------------------------------------------------


class PrivacyFilterSpan(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class PrivacyFilterRequest(BaseModel):
    model: str
    text: str
    mask_strategy: str = "replace"
    categories: Optional[list[str]] = None


class PrivacyFilterResponse(BaseModel):
    masked_text: str
    spans: list[PrivacyFilterSpan]
