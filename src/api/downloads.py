"""Fetching what a request points at: a URL the caller gave us, with a size it may not pass.

Two ways, and which one a caller takes is not a preference: to a file when what follows reads a
file (ffmpeg, a model that takes a path), into memory only when the bytes themselves are the input
to the next step. Either way the body is read in chunks and the limit is enforced as it arrives, so
a URL that answers with a hundred gigabytes costs us one chunk, not a hundred gigabytes.

The caller names the URL, so it may name us: a loopback admin port, the LAN, a cloud metadata
address. Only public addresses are fetched, on every redirect hop: the host is resolved and refused
before the request is sent, and a direct connection connects to the address it vetted, so a name
that resolves public for the check and private for the connect (DNS rebinding) gets nowhere. Through
an HTTP proxy the proxy connects, so the hop check is what stands there. ``LZAI_ALLOW_PRIVATE_URLS=1``
lifts it for local tests against a server on this machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
from pathlib import Path

import httpcore
import httpx

#: A voice reference is one clean take of someone speaking; nothing that long needs more.
MAX_REFERENCE_BYTES = 30 * 1024 * 1024
#: An image to look at or audio to read: generous for either, and a bound where there was none.
MAX_MEDIA_BYTES = 100 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0

CHUNK_BYTES = 1 << 20


class DownloadError(ValueError):
    """The URL the caller gave could not be fetched, or is larger than this fetch allows."""


def public_address(address: str) -> bool:
    """Whether an IP is one anyone on the internet could reach: not loopback, private, link-local,
    shared (CGNAT, our tailnet), reserved, documentation or multicast space."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _private_allowed() -> bool:
    return os.environ.get("LZAI_ALLOW_PRIVATE_URLS", "").strip().lower() in {"1", "true", "yes"}


async def _public_addresses(host: str, port: int) -> list[str]:
    """``host``'s addresses, all of them public, or a refusal naming the host."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise httpcore.ConnectError(f"{host} does not resolve: {exc}") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(public_address(address) for address in addresses):
        # One private answer is enough: which one a connect picks is not ours to bet on.
        raise httpcore.ConnectError(f"{host} is not a public address")
    return addresses


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """Connects only to public addresses, and to the very address it checked."""

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last_error: Exception | None = None
        for address in await _public_addresses(host, port):
            try:
                # TLS still verifies and sends SNI for the URL's host: httpcore passes it separately.
                return await self._inner.connect_tcp(
                    address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except httpcore.ConnectError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("unix sockets are not fetched")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


async def _refuse_private_hop(request: httpx.Request) -> None:
    # Runs in the client, outside the transport that maps httpcore errors to httpx ones.
    host = request.url.host
    try:
        await _public_addresses(host, request.url.port or (443 if request.url.scheme == "https" else 80))
    except httpcore.ConnectError as exc:
        raise httpx.ConnectError(str(exc), request=request) from None


def _client(timeout_seconds: float) -> httpx.AsyncClient:
    """A client that follows redirects but only ever reaches public addresses (module docstring)."""
    if _private_allowed():
        return httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds)
    transport = httpx.AsyncHTTPTransport()
    pool = getattr(transport, "_pool", None)
    if not isinstance(getattr(pool, "_network_backend", None), httpcore.AsyncNetworkBackend):
        # An httpx that moved its pool must fail every fetch here, not fetch unguarded.
        raise RuntimeError("httpx transport has no httpcore network backend to guard")
    pool._network_backend = _PublicOnlyBackend(pool._network_backend)
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_seconds,
        transport=transport,
        event_hooks={"request": [_refuse_private_hop]},
    )


async def fetch_to_file(
    url: str,
    dest: Path,
    max_bytes: int,
    *,
    url_name: str = "the url",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Streams the URL into ``dest``. Nothing is held: the file is what the next step reads.

    ``dest`` is left only when it holds the whole body — half a file in a cache is worse than none.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    too_large = False
    try:
        with dest.open("wb") as handle:
            async with _client(timeout_seconds) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(CHUNK_BYTES):
                        if written + len(chunk) > max_bytes:
                            too_large = True
                            break
                        written += len(chunk)
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        _discard(dest)
        # Hanging up on a body still arriving is the transport's complaint, not the caller's.
        if too_large:
            raise _too_large(url_name, max_bytes) from None
        raise DownloadError(f"{url_name} could not be fetched: {exc}") from exc
    except BaseException:
        _discard(dest)
        raise
    if too_large:
        _discard(dest)
        raise _too_large(url_name, max_bytes)
    if not written:
        _discard(dest)
        raise DownloadError(f"{url_name} returned an empty response")
    return dest


async def fetch_bytes(
    url: str,
    max_bytes: int,
    *,
    url_name: str = "the url",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> bytes:
    """The URL's bytes, for the step that reads bytes — a model, a decoder, a classifier."""
    parts: list[bytes] = []
    held = 0
    too_large = False
    try:
        async with _client(timeout_seconds) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(CHUNK_BYTES):
                    if held + len(chunk) > max_bytes:
                        too_large = True
                        break
                    held += len(chunk)
                    parts.append(chunk)
    except httpx.HTTPError as exc:
        # Hanging up on a body still arriving is the transport's complaint, not the caller's.
        if too_large:
            raise _too_large(url_name, max_bytes) from None
        raise DownloadError(f"{url_name} could not be fetched: {exc}") from exc
    if too_large:
        raise _too_large(url_name, max_bytes)
    if not held:
        raise DownloadError(f"{url_name} returned an empty response")
    return b"".join(parts)


def _too_large(url_name: str, max_bytes: int) -> DownloadError:
    return DownloadError(f"{url_name} is larger than {max_bytes // (1024 * 1024)} MB")


def _discard(dest: Path) -> None:
    with contextlib.suppress(OSError):
        dest.unlink()
