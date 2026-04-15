"""
OpenAI-compatible request/response Pydantic schemas.

Only the fields that DynLLM actually uses/proxies are declared.
Unknown extra fields are forwarded transparently to the backend.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


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
