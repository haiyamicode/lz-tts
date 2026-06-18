"""Request body decompression middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import gzip
from typing import Any

import brotli
import zstandard as zstd
from starlette.types import Message, Receive, Scope, Send


class RequestDecompressionMiddleware:
    """Decode compressed HTTP request bodies before application routing."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        encoding = _content_encoding(scope)
        if encoding is None or encoding == "identity":
            await self.app(scope, receive, send)
            return
        if encoding not in {"gzip", "br", "zstd"}:
            await _send_error(send, 415, f"Unsupported Content-Encoding: {encoding}")
            return

        body = await _read_body(receive)
        try:
            decoded = _decompress(body, encoding)
        except Exception as exc:
            await _send_error(send, 400, f"Invalid {encoding} request body: {exc}")
            return

        sent = False

        async def decoded_receive() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": decoded, "more_body": False}

        await self.app(_decoded_scope(scope, decoded), decoded_receive, send)


def _content_encoding(scope: Scope) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-encoding":
            return value.decode("latin-1").strip().lower()
    return None


async def _read_body(receive: Receive) -> bytes:
    parts: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b""
        if message["type"] != "http.request":
            continue
        parts.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(parts)


def _decompress(body: bytes, encoding: str) -> bytes:
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "br":
        return brotli.decompress(body)
    if encoding == "zstd":
        return zstd.ZstdDecompressor().decompress(body)
    raise ValueError(encoding)


def _decoded_scope(scope: Scope, body: bytes) -> Scope:
    new_scope: dict[str, Any] = dict(scope)
    headers = [
        (key, value)
        for key, value in scope.get("headers", [])
        if key.lower() not in {b"content-encoding", b"content-length"}
    ]
    headers.append((b"content-length", str(len(body)).encode("ascii")))
    new_scope["headers"] = headers
    return new_scope


async def _send_error(send: Send, status: int, detail: str) -> None:
    body = detail.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
