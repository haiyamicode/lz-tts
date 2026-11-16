"""Fetching what a request points at: a URL the caller gave us, with a size it may not pass.

Two ways, and which one a caller takes is not a preference: to a file when what follows reads a
file (ffmpeg, a model that takes a path), into memory only when the bytes themselves are the input
to the next step. Either way the body is read in chunks and the limit is enforced as it arrives, so
a URL that answers with a hundred gigabytes costs us one chunk, not a hundred gigabytes.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import httpx

#: A voice reference is one clean take of someone speaking; nothing that long needs more.
MAX_REFERENCE_BYTES = 30 * 1024 * 1024
#: An image to look at or audio to read: generous for either, and a bound where there was none.
MAX_MEDIA_BYTES = 100 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0

CHUNK_BYTES = 1 << 20


class DownloadError(ValueError):
    """The URL the caller gave could not be fetched, or is larger than this fetch allows."""


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
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds) as client:
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
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds) as client:
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
