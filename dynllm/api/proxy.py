"""
HTTP proxy helpers.

Forward a request to a backend subprocess and stream (or buffer) the response
back to the caller.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 120.0
_HOP_BY_HOP = ("content-length", "transfer-encoding", "content-encoding")


def _backend_url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}/{path.lstrip('/')}"


def _forward_headers(request: Request) -> dict[str, str]:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }


def _response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}


async def forward_request(
    request: Request,
    port: int,
    path: str,
    body: bytes,
) -> Response:
    """Forward *request* to a backend and return its buffered response."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=30.0, pool=5.0
        )
    ) as client:
        resp = await client.request(
            request.method,
            _backend_url(port, path),
            headers=_forward_headers(request),
            content=body,
        )

    headers = _response_headers(resp)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=headers,
        media_type=headers.get("content-type"),
    )


async def forward_streaming_request(
    request: Request,
    port: int,
    path: str,
    body: bytes,
) -> StreamingResponse:
    """Forward a streaming (SSE) request and pipe the chunks back."""
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=None, write=30.0, pool=5.0)
    )

    async def generator() -> AsyncIterator[bytes]:
        try:
            async with client.stream(
                request.method,
                _backend_url(port, path),
                headers=_forward_headers(request),
                content=body,
            ) as backend_resp:
                async for chunk in backend_resp.aiter_bytes():
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
