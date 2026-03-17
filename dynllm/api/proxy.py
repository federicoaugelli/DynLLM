"""
HTTP proxy helpers.

Forwards requests to the appropriate backend subprocess and streams
(or buffers) the response back to the caller.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Timeout for non-streaming requests.
# For streaming the connect timeout is the same but read timeout is unlimited.
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 120.0


def _backend_url(port: int, path: str) -> str:
    """Build the URL to a backend subprocess."""
    path = path.lstrip("/")
    return f"http://127.0.0.1:{port}/{path}"


async def forward_request(
    request: Request,
    port: int,
    path: str,
    body: bytes,
) -> Response:
    """
    Forward *request* to a backend on *port* at *path*.

    Handles both streaming (SSE) and regular JSON responses.
    """
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    url = _backend_url(port, path)
    method = request.method

    async def _stream_generator(
        client: httpx.AsyncClient,
    ) -> AsyncIterator[bytes]:
        async with client.stream(
            method,
            url,
            headers=headers,
            content=body,
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT, read=None, write=30.0, pool=5.0
            ),
        ) as backend_resp:
            async for chunk in backend_resp.aiter_bytes():
                yield chunk

    # Peek at Content-Type to decide streaming vs buffered
    # We decide based on the *request* body's stream field rather than
    # re-parsing – the caller already knows.
    # Use a short initial probe to get response headers.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=30.0, pool=5.0)
    ) as client:
        resp = await client.request(
            method,
            url,
            headers=headers,
            content=body,
        )

    response_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower()
        not in (
            "transfer-encoding",
            "content-encoding",
            "content-length",
        )
    }

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )


async def forward_streaming_request(
    request: Request,
    port: int,
    path: str,
    body: bytes,
) -> StreamingResponse:
    """
    Forward a streaming (SSE) request and pipe the chunks back.
    """
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    url = _backend_url(port, path)
    method = request.method

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_CONNECT_TIMEOUT,
            read=None,  # no read timeout for streaming
            write=30.0,
            pool=5.0,
        )
    )

    async def generator() -> AsyncIterator[bytes]:
        try:
            async with client.stream(
                method, url, headers=headers, content=body
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
