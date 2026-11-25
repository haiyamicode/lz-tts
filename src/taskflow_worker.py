"""Headless Taskflow worker for durable LZAI synthesis jobs."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .api.server import (
    InferenceOperationError,
    InferenceResult,
    LzTtsInferenceSession,
    _env_bool,
    _media_scratch_dir,
    create_app,
    deferred_engines,
    get_health_status,
    residency_snapshot,
    set_status,
)
from .api.take_stream import Mp3Segment, Mp3Segmenter, mp3_duration_seconds
from .api.worker_common import ChildWorkerDied
from .process_guard import hard_exit

_LOGGER = logging.getLogger(__name__)
TASK_TYPES = (
    "tts-synthesis",
    "voice-design",
    "voice-enhance",
    "nudity-detect",
    "face-detect",
    "language-detect",
    "transcribe",
    "force-align",
    "generate-music",
    "videogen",
    "videogen-chain-decode",
    "lipsync",
    "imagegen",
)
# Media runtime task types: advertised only when the worker's server config ``media`` enables them.
_MEDIA_TASK_TYPES = {
    "generate-music": "music",
    "videogen": "videogen",
    "videogen-chain-decode": "videogen",
    "lipsync": "lipsync",
    "imagegen": "imagegen",
}
# Task types whose work runs long enough that the person waiting needs to see it moving. The rest
# answer in seconds, where a report would land after the result it was about.
_REPORTED_TASK_TYPES = frozenset({*_MEDIA_TASK_TYPES, "transcribe"})

# Task types whose payload carries no operation field: the task type implies
# exactly one operation. Everything else (tts-synthesis) reads payload["operation"].
_TYPE_OPERATIONS = {
    "voice-design": "voice-design",
    "voice-enhance": "voice-enhance",
    "nudity-detect": "detect-nudity",
    "face-detect": "detect-face",
    "language-detect": "detect-language",
    "transcribe": "transcribe",
    "force-align": "force-align",
    "generate-music": "generate-music",
    "videogen": "videogen",
    "videogen-chain-decode": "videogen-chain-decode",
    "lipsync": "lipsync",
    "imagegen": "imagegen",
}


# Engines a task type may run on. A type is pulled only once none of them is still waiting
# for its model sets to download (LZAI_REQUIRED_MODELS); the other types need core data only.
_TYPE_ENGINES = {
    "tts-synthesis": ("pipertts", "voxcpm", "seed_vc"),
    "voice-design": ("voxcpm",),
    "transcribe": ("transcribe",),
    **{task_type: ("media",) for task_type in _MEDIA_TASK_TYPES},
}


# Leases a worker runs at once, per group of task types (Taskflow join ``capabilities.taskGroups``).
# Only VoxCPM work batches (nano-vLLM batches across the leases in flight), so its two types share
# the box concurrency (TASKFLOW_WORKER_CONCURRENCY). The media types share one GPU lock in the
# media runtime, so one at a time between them. Every other type is a single call that gains
# nothing from company: one at a time each.
_BATCHED_TASK_TYPES = ("tts-synthesis", "voice-design")


@dataclass(frozen=True)
class TaskGroup:
    task_types: tuple[str, ...]
    concurrency: int


def task_groups(task_types: tuple[str, ...], concurrency: int) -> tuple[TaskGroup, ...]:
    """The groups of the served ``task_types``; types outside the batched and media groups run alone."""
    shared = (
        TaskGroup(tuple(t for t in _BATCHED_TASK_TYPES if t in task_types), concurrency),
        TaskGroup(tuple(t for t in _MEDIA_TASK_TYPES if t in task_types), 1),
    )
    grouped = {t for group in shared for t in group.task_types}
    alone = tuple(TaskGroup((t,), 1) for t in task_types if t not in grouped)
    return tuple(group for group in (*shared, *alone) if group.task_types)


def open_task_types(
    groups: tuple[TaskGroup, ...], in_flight: Mapping[str, int], task_types: tuple[str, ...]
) -> tuple[tuple[str, ...], int]:
    """The ``task_types`` whose group has a free slot beside the leases ``in_flight`` (by type),
    and how many leases those free slots take together."""
    open_types: set[str] = set()
    room = 0
    for group in groups:
        free = group.concurrency - sum(in_flight.get(t, 0) for t in group.task_types)
        wanted = [t for t in group.task_types if t in task_types]
        if free > 0 and wanted:
            open_types.update(wanted)
            room += free
    return tuple(t for t in task_types if t in open_types), room


def ready_task_types(task_types: tuple[str, ...], deferred: frozenset[str]) -> tuple[str, ...]:
    """The task types whose engines are all loaded (none of them in ``deferred``)."""
    return tuple(t for t in task_types if not deferred.intersection(_TYPE_ENGINES.get(t, ())))


# load.mode on-demand: the swappable engine each task type mainly runs on. tts-synthesis may
# route to Sparrow, but counts as VoxCPM work here; force-align is the aligner; voice-enhance
# (ffmpeg) and the detect-* types need none (CPU).
_TYPE_SWAP_ENGINE = {
    "tts-synthesis": "voxcpm",
    "voice-design": "voxcpm",
    "force-align": "aligner",
    "transcribe": "transcribe",
    **{task_type: "media" for task_type in _MEDIA_TASK_TYPES},
}


def on_demand_task_types(
    task_types: tuple[str, ...],
    resident: frozenset[str],
    waiting_for: str | None,
    sticky: bool,
    pinned: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """What an on-demand worker pulls.

    While a job waits for memory to load ``waiting_for``, only that engine's types (and the
    types needing no swappable engine): anything else would run on an engine about to be
    unloaded. While ``sticky``, the loaded engines' types, so a mixed queue does not swap
    models on every job; the caller drops stickiness after an empty pull or a while after
    the last swap, so the other types still get pulled. ``pinned`` engines never unload:
    their types are always pulled, and they alone make nothing sticky.
    """
    if waiting_for is not None:
        allowed = {waiting_for} | pinned
    elif sticky and resident - pinned:
        allowed = set(resident) | pinned
    else:
        return task_types
    return tuple(t for t in task_types if t not in _TYPE_SWAP_ENGINE or _TYPE_SWAP_ENGINE[t] in allowed)


def served_task_types(config: Any) -> tuple[str, ...]:
    """TASK_TYPES minus those whose engine this worker's config (or LZAI_MODELS) turns off."""
    off = {task_type for task_type, feature in _MEDIA_TASK_TYPES.items() if not getattr(config.media, feature)}
    if not config.transcribe.enabled:
        off.add("transcribe")
    if not config.engines.voxcpm:
        off.add("voice-design")
        if not config.engines.pipertts:
            off.add("tts-synthesis")
    return tuple(task_type for task_type in TASK_TYPES if task_type not in off)
_REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)
# Join, heartbeat, pull, settle and leave are small requests (pulls long-poll
# for at most a second), answered by the server in milliseconds. On the long
# home -> API path a single connection sometimes stalls while fresh ones go
# straight through, so a control request gives up well inside the server's
# session timeout (3 heartbeat intervals, 45 s) and is retried on a new
# connection, instead of letting one stuck request expire the session.
_CONTROL_TIMEOUT = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0)
# Retry delay after a failed heartbeat or pull; the request that failed took
# its connection down with it, so the retry goes out on a fresh one.
_CONTROL_RETRY_DELAY = 1.0
_PULL_RETRY_MAX_DELAY = 5.0
# The server expires a session after this many heartbeat intervals without one.
_SESSION_HEARTBEATS = 3
_CALLBACK_MAX_ATTEMPTS = 3
_RECONNECT_BASE_DELAY = 2.0
# Length of the segments a streamed take is uploaded in.
_SEGMENT_SECONDS = 1.0
# The least time between two progress reports of one job.
_PROGRESS_INTERVAL_SECONDS = 1.0
# Where each stage sits in the whole of a job: the render is the long one.
_PROGRESS_STAGE_SPANS = {
    "preparing": (0.0, 0.05),
    "rendering": (0.05, 0.95),
    "finishing": (0.95, 1.0),
}
_RECONNECT_MAX_DELAY = 60.0


# Taskflow error codes. A lease that a cancelled, expired or superseded run no
# longer owns is a different thing from a worker session that was fenced or
# expired, and the worker has to keep them apart: a released lease is a routine
# outcome, while a dead session needs a rejoin.
_LEASE_GONE_CODES = frozenset(
    {"INVALID_LEASE", "LEASE_LOST", "RUN_CANCELLED", "RUN_TERMINAL"}
)
_SESSION_GONE_CODES = frozenset(
    {"INVALID_WORKER", "INVALID_WORKER_SESSION", "WORKER_NOT_AVAILABLE"}
)


def _error_code(response: httpx.Response) -> str | None:
    """Machine-readable code from a Taskflow error body (``{"error":{"code":...}}``)."""
    try:
        code = response.json()["error"]["code"]
    except Exception:
        return None
    return code if isinstance(code, str) else None


class ProtocolError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        code: str | None = None,
        *,
        scope: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.scope = scope

    @property
    def loses_lease(self) -> bool:
        """The lease is gone: its run was cancelled, expired or superseded."""
        if self.code is not None:
            return self.code in _LEASE_GONE_CODES
        # Lease endpoints authenticate with the lease token alone, and the
        # artifact route turns every authorization failure into a bare 401, so a
        # code-less client error there can only mean the lease was released.
        return (
            self.scope == "lease"
            and self.status_code is not None
            and 400 <= self.status_code < 500
        )

    @property
    def loses_session(self) -> bool:
        """The worker session is invalid: fenced by another join, expired or disabled."""
        if self.code is not None:
            return self.code in _SESSION_GONE_CODES
        return self.scope == "session" and self.status_code in {401, 403, 404}


@dataclass
class TaskflowWorker:
    base_url: str
    worker_token: str
    worker_id: str
    concurrency: int = 8
    # Batch cap for negative-priority (background) work, so a long low-priority
    # batch never keeps a normal job waiting behind it.
    low_priority_batch_size: int = 3
    persistent: bool = False
    task_types: tuple[str, ...] = TASK_TYPES
    # What pulls ask for: the joined ``task_types`` minus those whose engines are still
    # waiting for their model sets (Taskflow narrows a pull to a subset of the join's types).
    pull_task_types: tuple[str, ...] | None = None
    # On-demand mode: the last pull was narrowed to the loaded engines' types (stickiness),
    # the last such pull came back empty, and when the loaded set last changed.
    pull_sticky: bool = False
    sticky_missed: bool = False
    resident: frozenset[str] = frozenset()
    resident_since: float = 0.0
    pool_metadata: dict[str, object] = field(default_factory=dict)
    sticky_seconds: float = 300.0
    session_id: str | None = None
    session_token: str | None = None
    heartbeat_interval: float = 15.0
    active_lease_ids: set[str] = field(default_factory=set)
    # lease id -> "cancelled" | "lost", as reported by the server heartbeat.
    dropped_lease_ids: dict[str, str] = field(default_factory=dict)
    # lease id -> set when the heartbeat sees it released: a media render running for it stops.
    lease_cancel_events: dict[str, asyncio.Event] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

    @property
    def task_groups(self) -> tuple[TaskGroup, ...]:
        return task_groups(self.task_types, self.concurrency)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def reset_client(self) -> None:
        """Replace the HTTP client, discarding its whole connection pool.

        httpcore can strand connections in the pool for good: through an HTTP
        proxy, a tunnel whose CONNECT succeeded but whose TLS handshake timed
        out stays neither closed, idle nor expired, so it is never evicted.
        Once those fill the pool, every request fails with PoolTimeout even
        after the network recovers. A fresh client is the only way out.
        """
        old_client, self._client = self._client, httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(old_client.aclose(), timeout=10)

    async def close(self) -> None:
        try:
            if self.session_id and self.session_token:
                await self._request(
                    "DELETE",
                    f"/workers/{self.worker_id}",
                    token=self.session_token,
                    headers=self._session_headers(),
                )
        except Exception:
            _LOGGER.exception("Failed to leave Taskflow during shutdown")
        finally:
            self.session_id = None
            self.session_token = None
            await self._client.aclose()

    async def _request(
        self, method: str, path: str, *, token: str | None = None, **kwargs: Any
    ) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        if token:
            headers = {"Authorization": f"Bearer {token}", **headers}
        kwargs.setdefault("timeout", _CONTROL_TIMEOUT)
        response = await self._client.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            **kwargs,
        )
        if response.is_error:
            raise ProtocolError(
                f"Taskflow {method} {path} failed ({response.status_code}): {response.text[:500]}",
                response.status_code,
                _error_code(response),
                scope="session" if path.startswith("/workers/") else None,
            )
        return response

    async def join(self, metadata: dict[str, Any]) -> None:
        response = await self._request(
            "POST",
            "/workers/join",
            token=self.worker_token,
            json={
                "workerId": self.worker_id,
                "ephemeral": not self.persistent,
                "metadata": metadata,
                "capabilities": {
                    "taskTypes": list(self.task_types),
                    "concurrency": self.concurrency,
                    # A Taskflow server without per-group limits ignores this.
                    "taskGroups": [
                        {"taskTypes": list(group.task_types), "concurrency": group.concurrency}
                        for group in self.task_groups
                    ],
                },
            },
        )
        body = response.json()
        self.session_id = body["worker"]["sessionId"]
        self.session_token = body["sessionToken"]
        self.heartbeat_interval = max(1.0, body["heartbeatIntervalMs"] / 1000)

    def _session_headers(self) -> dict[str, str]:
        if not self.session_id:
            raise ProtocolError("Worker has not joined Taskflow")
        return {"Taskflow-Worker-Session": self.session_id}

    async def heartbeat(self) -> None:
        if not self.session_token:
            raise ProtocolError("Worker has not joined Taskflow")
        response = await self._request(
            "POST",
            f"/workers/{self.worker_id}/heartbeat",
            token=self.session_token,
            headers=self._session_headers(),
            json={"activeLeaseIds": sorted(self.active_lease_ids)},
        )
        body = response.json()
        cancelled = body.get("cancelledLeaseIds", [])
        lost = body.get("lostLeaseIds", [])
        if cancelled or lost:
            # Report it where it happens; otherwise the only trace of a cancelled
            # run is a confusing 401 on the artifact upload minutes later, after
            # the worker has already generated the audio.
            _LOGGER.info(
                "Taskflow released in-flight leases cancelled=%s lost=%s",
                sorted(cancelled),
                sorted(lost),
            )
        for lease_id in cancelled:
            self.dropped_lease_ids[lease_id] = "cancelled"
        for lease_id in lost:
            self.dropped_lease_ids.setdefault(lease_id, "lost")
        for lease_id in (*cancelled, *lost):
            self.active_lease_ids.discard(lease_id)
            event = self.lease_cancel_events.get(lease_id)
            if event is not None:
                event.set()

    async def heartbeat_loop(self, connection_lost: asyncio.Event) -> None:
        """Beat every interval, retrying a failed beat at once on a new connection.

        A failed request says nothing about the session: only the server saying
        so (or no beat getting through for as long as the server keeps a
        session) sends the serve loop to rejoin. Beating goes on after that
        too: heartbeats also extend the leases of the batches still in flight
        while the serve loop drains them.
        """
        last_ok = time.monotonic()
        delay = self.heartbeat_interval
        while True:
            await asyncio.sleep(delay)
            started = time.monotonic()
            try:
                await self.heartbeat()
            except ProtocolError as error:
                if not error.loses_session:
                    delay = self._heartbeat_failed(error, last_ok, connection_lost)
                    continue
                _LOGGER.error("Taskflow ended the worker session: %s", error)
                connection_lost.set()
                delay = self.heartbeat_interval
            except Exception as error:
                delay = self._heartbeat_failed(error, last_ok, connection_lost)
            else:
                last_ok = time.monotonic()
                delay = max(0.0, self.heartbeat_interval - (last_ok - started))

    def _heartbeat_failed(
        self, error: BaseException, last_ok: float, connection_lost: asyncio.Event
    ) -> float:
        silent = time.monotonic() - last_ok
        if silent >= self.heartbeat_interval * _SESSION_HEARTBEATS:
            _LOGGER.error(
                "Taskflow worker heartbeat failing for %.0fs; reconnecting: %r", silent, error
            )
            connection_lost.set()
            return self.heartbeat_interval
        _LOGGER.warning(
            "Taskflow worker heartbeat failed after %.0fs without one; retrying: %r", silent, error
        )
        return _CONTROL_RETRY_DELAY

    async def pull(
        self,
        *,
        max_tasks: int | None = None,
        wait_ms: int = 1000,
        min_priority: int | None = None,
        priority_bands: list[dict[str, int]] | None = None,
        task_types: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.session_token:
            raise ProtocolError("Worker has not joined Taskflow")
        if task_types is None:
            task_types = self.task_types if self.pull_task_types is None else self.pull_task_types
        body: dict[str, Any] = {
            "maxTasks": max_tasks or self.concurrency,
            "waitMs": wait_ms,
            "taskTypes": list(task_types),
        }
        # A Taskflow server older than these pull filters ignores them.
        if min_priority is not None:
            body["minPriority"] = min_priority
        if priority_bands:
            body["priorityBands"] = priority_bands
        try:
            response = await self._request(
                "POST",
                f"/workers/{self.worker_id}/pull",
                token=self.session_token,
                headers=self._session_headers(),
                json=body,
            )
        except ProtocolError as error:
            # A server restart can leave the previous long-poll marker alive briefly in Redis.
            # Keep the valid worker session and heartbeat alive until that marker expires.
            if error.status_code == 409 and "PULL_IN_PROGRESS" in str(error):
                await asyncio.sleep(1)
                return []
            raise
        return response.json()["leases"]

    async def settle(self, settlements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Complete or fail leases in one request; their results come back in order.

        Each item carries its own lease token, and a lease the server already
        released is reported in its result instead of failing the request. So a
        transient failure is safe to retry: a lease the lost attempt did settle
        just comes back released.
        """
        attempt = 1
        while True:
            try:
                response = await self._request(
                    "POST", "/leases/settle", json={"leases": settlements}
                )
                return response.json()["results"]
            except (httpx.TransportError, ProtocolError) as error:
                transient = isinstance(error, httpx.TransportError) or (
                    (error.status_code or 0) >= 500 and error.status_code != 501
                )
                if not transient or attempt >= _CALLBACK_MAX_ATTEMPTS:
                    raise
                _LOGGER.warning(
                    "Taskflow lease settlement failed attempt=%d leases=%d: %s",
                    attempt,
                    len(settlements),
                    error,
                )
            await asyncio.sleep(1.0 * attempt)
            attempt += 1

    @property
    def artifact_base_url(self) -> str:
        return self.base_url.removesuffix("/taskflow/v1") + "/tts-artifacts/v1"

    async def deliver_media(self, lease: dict[str, Any], result: dict[str, Any]) -> None:
        """Hand a media job's result to the API before the lease settles, so the job is
        delivered (credits charged, the next workflow step queued) by the same run that rendered
        it. ``result`` is exactly the completion the lease then settles with; the API takes it
        idempotently, so a retry after a lost response is harmless.

        204 is delivered and 404 (an API without the route, a deploy window) is taken as
        delivered. Any other client error means the lease is no longer this worker's. A server or
        network error is retried with backoff and then raised, which fails the lease as
        retryable."""
        url = (
            self.base_url.removesuffix("/taskflow/v1")
            + f"/media-jobs/v1/leases/{lease['id']}/deliver"
        )
        headers = {"Authorization": f"Bearer {lease['leaseToken']}"}
        last_error = ""
        for attempt in range(1, _CALLBACK_MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(
                    url, json={"result": result}, headers=headers, timeout=_CONTROL_TIMEOUT
                )
            except httpx.HTTPError as error:
                last_error = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 404:
                    _LOGGER.warning(
                        "Media delivery route missing (404) lease=%s; settling", lease["id"]
                    )
                    return
                if not response.is_error:
                    _LOGGER.info("Media job delivered lease=%s", lease["id"])
                    return
                if response.status_code < 500:
                    raise ProtocolError(
                        f"Media delivery failed ({response.status_code}): {response.text[:500]}",
                        response.status_code,
                        _error_code(response),
                        scope="lease",
                    )
                last_error = f"{response.status_code}: {response.text[:500]}"
            if attempt < _CALLBACK_MAX_ATTEMPTS:
                await asyncio.sleep(2.0 * attempt)
        raise RuntimeError(f"Media delivery failed lease={lease['id']}: {last_error}")

    async def report_progress(self, lease: dict[str, Any], progress: dict[str, Any]) -> None:
        """Tell the API where the job of ``lease`` has got to, for it to pass to the person
        waiting. It goes on the run itself, over the Taskflow protocol this worker already speaks:
        the lease is the authority, and the API knows from the run which job it is for.

        A report that fails is dropped: the progress is only shown, and a job must never fail over
        one."""
        try:
            await self._request(
                "POST",
                f"/leases/{lease['id']}/progress",
                token=lease["leaseToken"],
                json=progress,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("Progress report dropped lease=%s: %s", lease["id"], error)

    async def upload_segment(
        self,
        lease: dict[str, Any],
        artifact_id: str,
        index: int,
        segment: Mp3Segment,
        *,
        last: bool,
    ) -> None:
        """Upload one segment of a streamed take; the last one completes the artifact."""
        params = {"duration": f"{segment.seconds:.6f}"}
        if last:
            params["last"] = "1"
        response = await self._client.put(
            f"{self.artifact_base_url}/leases/{lease['id']}/{artifact_id}/segments/{index}",
            params=params,
            content=segment.data,
            headers={
                "Authorization": f"Bearer {lease['leaseToken']}",
                "Content-Type": "application/octet-stream",
            },
            timeout=_CONTROL_TIMEOUT,
        )
        if response.is_error:
            raise ProtocolError(
                f"Segment upload failed ({response.status_code}): {response.text[:500]}",
                response.status_code,
                _error_code(response),
                scope="lease",
            )

    async def upload(self, lease: dict[str, Any], artifact_id: str, content: bytes | Path) -> None:
        """Send an artifact: audio made in memory as the bytes it is, a file as the file it is.

        A render is a video of up to gigabytes, and nothing here has any use for its bytes, so
        they are read a chunk at a time and sent as they come. The length is the file's, so the
        upload is an ordinary one the API sizes up front, not a chunked stream.
        """
        artifact_base_url = self.artifact_base_url
        started_at = time.monotonic()
        size = len(content) if isinstance(content, bytes) else content.stat().st_size
        _LOGGER.info(
            "Artifact upload started lease=%s artifact=%s bytes=%d",
            lease["id"],
            artifact_id,
            size,
        )
        response = await self._client.put(
            f"{artifact_base_url}/leases/{lease['id']}/{artifact_id}",
            content=content if isinstance(content, bytes) else _file_chunks(content),
            headers={
                "Authorization": f"Bearer {lease['leaseToken']}",
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
        )
        if response.is_error:
            raise ProtocolError(
                f"Artifact upload failed ({response.status_code}): {response.text[:500]}",
                response.status_code,
                _error_code(response),
                scope="lease",
            )
        _LOGGER.info(
            "Artifact upload completed lease=%s artifact=%s bytes=%d wall_seconds=%.3f",
            lease["id"],
            artifact_id,
            size,
            time.monotonic() - started_at,
        )

    async def download(self, lease: dict[str, Any], artifact_id: str, destination: Path) -> int:
        """Read an artifact an earlier run of the lease's workflow uploaded (a videogen chain
        part's latent) into ``destination``, through the lease. Returns its size."""
        url = f"{self.artifact_base_url}/leases/{lease['id']}/{artifact_id}"
        headers = {"Authorization": f"Bearer {lease['leaseToken']}"}
        async with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code == 404:
                raise InferenceOperationError(400, f"artifact {artifact_id} does not exist")
            if response.is_error:
                await response.aread()
                if response.status_code >= 500:  # retried like any failed job
                    raise RuntimeError(
                        f"Artifact download failed ({response.status_code}): {response.text[:500]}"
                    )
                raise ProtocolError(  # the lease is gone
                    f"Artifact download failed ({response.status_code}): {response.text[:500]}",
                    response.status_code,
                    _error_code(response),
                    scope="lease",
                )
            size = 0
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes(1 << 20):
                    handle.write(chunk)
                    size += len(chunk)
        _LOGGER.info(
            "Artifact download completed lease=%s artifact=%s bytes=%d", lease["id"], artifact_id, size
        )
        return size


@dataclass(frozen=True)
class _LeaseWork:
    lease: dict[str, Any]
    operation: str
    request: dict[str, Any]
    task_type: str
    started_at: float
    # Inputs the worker read from the lease for the job (a chain's latents), deleted after it.
    staged_dir: Path | None = None


async def _stage_lease_inputs(taskflow: TaskflowWorker, work: _LeaseWork) -> _LeaseWork:
    """A videogen chain part or chain decode names its latents by artifact id, a staged lipsync's chunk /
    assemble task its earlier stages' files; the media runtime reads files, so they are downloaded
    through the lease and the request names their paths."""
    from .api.media import VideogenChain  # pylint: disable=import-outside-toplevel

    request = work.request
    if work.task_type == "videogen" and request.get("chain") is not None:
        chain = VideogenChain.model_validate(request["chain"])
        if chain.latent_artifact_id != work.lease["payload"].get("artifactId"):
            raise InferenceOperationError(400, "chain.latent_artifact_id is not the payload's artifactId")
        wanted = {
            "continue_latent_path": chain.previous_latent_artifact_id,
            "chain_reference_latent_path": chain.anchor_latent_artifact_id,
        }
        request = {key: value for key, value in request.items() if key != "chain"}
        request["latent_only"] = True
    elif work.task_type == "videogen-chain-decode" and request.get("latent_artifact_ids"):
        ids = _artifact_id_list(request["latent_artifact_ids"], "latent_artifact_ids")
        wanted = {f"latent{index}": artifact_id for index, artifact_id in enumerate(ids)}
        request = {key: value for key, value in request.items() if key != "latent_artifact_ids"}
    elif work.task_type == "lipsync" and request.get("stage") in ("chunk", "assemble"):
        # A staged lipsync's chunk / assemble task: the plan's state, the previous chunk's boundary
        # frame, the chunks' renders and the job's audio, all earlier stages' artifacts.
        wanted = {
            "state_path": request.get("state_artifact_id"),
            "boundary_path": request.get("boundary_artifact_id"),
            "audio_path": request.get("audio_artifact_id"),
        }
        if request.get("chunk_artifact_ids") is not None:
            ids = _artifact_id_list(request["chunk_artifact_ids"], "chunk_artifact_ids")
            wanted.update({f"chunk{index}": artifact_id for index, artifact_id in enumerate(ids)})
        request = {key: value for key, value in request.items()
                   if key not in ("state_artifact_id", "boundary_artifact_id", "audio_artifact_id", "chunk_artifact_ids")}
    else:
        return work
    wanted = {name: artifact_id for name, artifact_id in wanted.items() if artifact_id}
    if not wanted:
        return dataclasses.replace(work, request=request)
    staged_dir = _media_scratch_dir(f"{work.task_type}-inputs")
    try:
        paths = {}
        for name, artifact_id in wanted.items():
            suffix = _STAGED_SUFFIXES.get(name, ".mp4" if work.task_type == "lipsync" else ".safetensors")
            paths[name] = staged_dir / f"{name.removesuffix('_path')}{suffix}"
            await taskflow.download(work.lease, artifact_id, paths[name])
    except BaseException:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    if work.task_type == "videogen-chain-decode":
        request["latent_paths"] = [str(path) for path in paths.values()]
    elif work.task_type == "lipsync":
        chunks = [str(path) for name, path in paths.items() if name.startswith("chunk")]
        request.update({name: str(path) for name, path in paths.items() if name.endswith("_path")})
        if chunks:
            request["chunk_paths"] = chunks
    else:
        request.update({name: str(path) for name, path in paths.items()})
    return dataclasses.replace(work, request=request, staged_dir=staged_dir)


# A staged input's file suffix (the media runtime probes the content; the suffix only names it).
_STAGED_SUFFIXES = {
    "state_path": ".tar",
    "boundary_path": ".png",
    "audio_path": ".audio",
}


def _artifact_id_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise InferenceOperationError(400, f"{field} must be a list of artifact ids")
    return value


@dataclass
class _LeaseResult:
    """A finished lease's outcome, uploaded and waiting on its ack."""

    work: _LeaseWork
    completion: dict[str, Any] | None = None
    error: BaseException | None = None
    retry: bool = False
    ack: dict[str, Any] | None = None


class _SegmentUploads:
    """Uploads a take in ~1 s segments as its MP3 is encoded, so it can play live.

    Segments go up one at a time, in order. A take that wasn't encoded live is
    segmented once it is complete. If a segment upload fails, the take is
    uploaded whole instead: it can't play live, but it is still kept.
    """

    def __init__(self, taskflow: TaskflowWorker, lease: dict[str, Any], artifact_id: str):
        self._taskflow = taskflow
        self._lease = lease
        self._artifact_id = artifact_id
        self._segmenter = Mp3Segmenter(_SEGMENT_SECONDS)
        self._received = 0
        self._count = 0
        self._chain: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    def write(self, mp3: bytes) -> None:
        self._received += len(mp3)
        for segment in self._segmenter.feed(mp3):
            self._send(segment, last=False)

    def _send(self, segment: Mp3Segment, *, last: bool) -> None:
        self._chain = asyncio.create_task(
            self._upload_after(self._chain, self._count, segment, last)
        )
        self._count += 1

    async def _upload_after(
        self,
        previous: asyncio.Task[None] | None,
        index: int,
        segment: Mp3Segment,
        last: bool,
    ) -> None:
        if previous is not None:
            await previous
        if self._error is not None:
            return
        try:
            await self._taskflow.upload_segment(
                self._lease, self._artifact_id, index, segment, last=last
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._error = error

    async def finish(self, audio: bytes) -> None:
        """Send the rest of ``audio`` (the whole take) and complete the artifact."""
        for segment in self._segmenter.feed(audio[self._received :]):
            self._send(segment, last=False)
        last = self._segmenter.finish()
        if last is not None:
            self._send(last, last=True)
        if self._chain is not None:
            await self._chain
        if self._error is None:
            return
        if isinstance(self._error, ProtocolError) and self._error.loses_lease:
            raise self._error
        _LOGGER.warning(
            "Segment upload failed; uploading the take whole lease=%s artifact=%s: %s",
            self._lease["id"],
            self._artifact_id,
            self._error,
        )
        await self._taskflow.upload(self._lease, self._artifact_id, audio)

    def cancel(self) -> None:
        if self._chain is not None:
            self._chain.cancel()


class _ProgressReports:
    """Tells the API where a long job has got to, so the person waiting on it sees it moving.

    Reports go one at a time, at most one a second, each of them the latest: progress overtaken
    while the one before it was in flight was never worth sending. Nothing waits on a report, and
    one that fails is dropped.
    """

    def __init__(self, taskflow: TaskflowWorker, lease: dict[str, Any]):
        self._taskflow = taskflow
        self._lease = lease
        self._next: dict[str, Any] | None = None
        self._sending = False

    def report(self, stage: str, done: int | None = None, total: int | None = None) -> None:
        counts = done is not None and total is not None
        # The run keeps both: how far along the whole job is, and where it is in its own terms.
        self._next = {
            "progress": _progress_fraction(stage, done, total),
            "state": {"stage": stage, **({"done": done, "total": total} if counts else {})},
        }
        if not self._sending:
            self._sending = True
            asyncio.create_task(self._send())

    async def _send(self) -> None:
        try:
            while self._next is not None:
                progress, self._next = self._next, None
                started = time.monotonic()
                await self._taskflow.report_progress(self._lease, progress)
                await asyncio.sleep(
                    max(0.0, _PROGRESS_INTERVAL_SECONDS - (time.monotonic() - started))
                )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Nobody awaits this task, so what it raises would only be logged as an error of its
            # own at the end of the job. The progress is shown and nothing more; it is dropped.
            _LOGGER.debug("Progress reporting stopped lease=%s: %s", self._lease["id"], error)
        finally:
            self._sending = False


def _progress_fraction(stage: str, done: int | None, total: int | None) -> float:
    """How far the whole job has got, 0 to 1, from how far one of its stages has."""
    start, end = _PROGRESS_STAGE_SPANS.get(stage, (0.0, 1.0))
    if not total or done is None:
        return start
    return start + (end - start) * min(1.0, done / total)


def _progress_reports(taskflow: TaskflowWorker, work: _LeaseWork) -> _ProgressReports | None:
    """Reports for a lease whose work runs long enough to be worth watching."""
    if work.task_type not in _REPORTED_TASK_TYPES:
        return None
    return _ProgressReports(taskflow, work.lease)


def _segment_uploads(taskflow: TaskflowWorker, work: _LeaseWork) -> _SegmentUploads | None:
    """Uploads for a lease whose API takes its artifact in segments."""
    payload = work.lease["payload"]
    artifact_id = payload.get("artifactId")
    if payload.get("segments") is not True or not isinstance(artifact_id, str) or not artifact_id:
        return None
    return _SegmentUploads(taskflow, work.lease, artifact_id)


def _callback_token(lease: dict[str, Any]) -> str | None:
    """Extract the opaque completion token from a lease payload, if any."""
    payload = lease.get("payload")
    if not isinstance(payload, dict):
        return None
    callback = payload.get("callback")
    if not isinstance(callback, dict):
        return None
    token = callback.get("token")
    return token if isinstance(token, str) and token else None


async def _file_chunks(path: Path, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
    """The file, a megabyte at a time, read off the event loop."""
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while chunk := await asyncio.to_thread(handle.read, chunk_size):
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)


def _artifact_duration(content: bytes | Path, content_type: str | None) -> float | None:
    """Length of a finished artifact in seconds; None when it can't be measured.

    An MP3 is counted frame by frame. A take is a bare LAME stream, so ffprobe
    has only the first frame's bitrate to go on and reads a VBR take 10-20%
    short — and a short duration is what the exporter trims each block to, which
    cuts the end off the line. Other containers state their length in the header,
    where ffprobe is exact.
    """
    if isinstance(content, bytes) and (content_type or "").startswith("audio/mpeg"):
        seconds = mp3_duration_seconds(content)
        if seconds is not None:
            return seconds
    return _measure_audio_duration(content)


def _measure_audio_duration(content: bytes | Path) -> float | None:
    """Measure audio duration in seconds via ffprobe; None when unavailable.

    A render is already a file, and ffprobe reads it where it is. Audio held in memory is written
    out first, as that is the only way ffprobe can be shown it.
    """
    if isinstance(content, Path):
        return _probe_duration(str(content))
    if not content:
        return 0.0
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
            handle.write(content)
            path = handle.name
    except OSError:
        _LOGGER.exception("Failed to buffer audio for duration measurement")
        return None
    try:
        return _probe_duration(path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _probe_duration(path: str) -> float | None:
    """What ffprobe makes of the file's length, in seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            _LOGGER.warning(
                "ffprobe duration measurement failed: %s",
                (result.stderr or result.stdout).strip()[:300],
            )
            return None
        return float(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        _LOGGER.warning("Audio duration measurement unavailable: %s", error)
        return None


@dataclass
class _AckFailure:
    error: str
    retry: bool


class SynthesisAckClient:
    """Acks synthesis outcomes to the API, as many as are ready in one request.

    The ack is what settles a block (attaching its audio, or recording a
    terminal failure) and runs post-synthesis work such as charging, so a run
    is only completed once its ack was accepted. The API applies acks
    idempotently, so resending items from a partially applied request is safe.

    Leases finish one by one, and the API commits each request's acks for a
    project as one project op. So an ack goes out at once when none is in
    flight; the ones that finish meanwhile go together in the next request.
    """

    def __init__(self, taskflow: TaskflowWorker, endpoint: str) -> None:
        # Shares the worker's HTTP client, which a reconnect may replace.
        self._taskflow = taskflow
        self._endpoint = endpoint
        self._waiting: list[tuple[list[dict[str, Any]], asyncio.Future[dict[str, _AckFailure]]]] = []
        self._sender: asyncio.Task[None] | None = None

    async def send(self, items: list[dict[str, Any]]) -> dict[str, _AckFailure]:
        """Ack ``items``; returns the ones the API did not accept, by runId."""
        result: asyncio.Future[dict[str, _AckFailure]] = asyncio.get_running_loop().create_future()
        self._waiting.append((items, result))
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_waiting())
        return await result

    async def _send_waiting(self) -> None:
        while self._waiting:
            batch, self._waiting = self._waiting, []
            try:
                failures = await self._send_items([item for items, _ in batch for item in items])
            except BaseException as error:  # pylint: disable=broad-exception-caught
                for _, result in batch:
                    if not result.done():
                        result.set_exception(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
                continue
            for items, result in batch:
                if not result.done():
                    result.set_result(
                        {item["runId"]: failures[item["runId"]] for item in items if item["runId"] in failures}
                    )

    async def _send_items(self, items: list[dict[str, Any]]) -> dict[str, _AckFailure]:
        failures: dict[str, _AckFailure] = {}
        pending = items
        for attempt in range(_CALLBACK_MAX_ATTEMPTS):
            if attempt:
                await asyncio.sleep(1.0 * attempt)
            try:
                response = await self._taskflow.client.post(
                    self._endpoint, json={"items": pending}
                )
                if response.is_error:
                    raise ProtocolError(
                        f"Synthesis ack batch failed ({response.status_code}): "
                        f"{response.text[:500]}",
                        response.status_code,
                    )
                body = response.json()
            except (httpx.HTTPError, ProtocolError, ValueError) as error:
                _LOGGER.warning(
                    "Synthesis ack request failed attempt=%d items=%d: %s",
                    attempt + 1,
                    len(pending),
                    error,
                )
                for item in pending:
                    failures[item["runId"]] = _AckFailure(str(error), retry=True)
                continue
            results = {result.get("runId"): result for result in body.get("results", [])}
            rejected = {rejection.get("index"): rejection for rejection in body.get("rejected", [])}
            retry_items: list[dict[str, Any]] = []
            for index, item in enumerate(pending):
                run_id = item["runId"]
                result = results.get(run_id) or {}
                if index in rejected:
                    # The API can't verify this callback token; resending won't help.
                    failures[run_id] = _AckFailure(
                        f"Synthesis ack rejected: {rejected[index].get('error')}", retry=False
                    )
                elif result.get("ok") is True:
                    failures.pop(run_id, None)
                else:
                    failures[run_id] = _AckFailure(
                        f"Synthesis ack failed: {result.get('error', 'no result')}", retry=True
                    )
                    retry_items.append(item)
            _LOGGER.info(
                "Synthesis acks sent items=%d accepted=%d",
                len(pending),
                sum(1 for item in pending if item["runId"] not in failures),
            )
            pending = retry_items
            if not pending:
                break
        return failures


def _prepare_lease(taskflow: TaskflowWorker, lease: dict[str, Any]) -> _LeaseWork:
    lease_id = lease["id"]
    run_id = lease.get("runId")
    task_type = lease.get("type", "unknown")
    payload = lease["payload"]
    operation = _TYPE_OPERATIONS.get(task_type) or payload["operation"]
    request = payload["request"]
    taskflow.active_lease_ids.add(lease_id)
    _LOGGER.info(
        "LZAI task started type=%s lease=%s run=%s operation=%s artifact=%s "
        "model=%s voice=%s locale=%s",
        task_type,
        lease_id,
        run_id,
        operation,
        payload.get("artifactId"),
        request.get("model"),
        request.get("voice_id") or request.get("id"),
        request.get("language") or request.get("locale"),
    )
    return _LeaseWork(
        lease=lease,
        operation=operation,
        request=request,
        task_type=task_type,
        started_at=time.monotonic(),
    )


def _lease_released(taskflow: TaskflowWorker, work: _LeaseWork) -> bool:
    lease_id = work.lease["id"]
    if lease_id in taskflow.active_lease_ids:
        return False
    # The heartbeat already watched the server release this lease, which
    # normally means the client cancelled or superseded the run.
    _LOGGER.info(
        "Discarding completed inference for released lease=%s run=%s reason=%s",
        lease_id,
        work.lease.get("runId"),
        taskflow.dropped_lease_ids.get(lease_id, "unknown"),
    )
    return True


def _completed_settlement(lease: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "leaseId": lease["id"],
        "token": lease["leaseToken"],
        "outcome": "completed",
        "result": result,
    }


def _failed_settlement(
    lease: dict[str, Any], error: BaseException, *, retry: bool
) -> dict[str, Any]:
    return {
        "leaseId": lease["id"],
        "token": lease["leaseToken"],
        "outcome": "failed",
        "retry": retry,
        "error": {"message": str(error), "data": {"type": type(error).__name__}},
    }


def _log_dropped_result(work: _LeaseWork, error: object) -> None:
    # Uploading or completing a result the server no longer wants is a
    # routine race with a run that got cancelled while we were generating
    # it. The lease is gone, not the session.
    _LOGGER.warning(
        "Dropped result for lease the server released lease=%s run=%s operation=%s: %s",
        work.lease["id"],
        work.lease.get("runId"),
        work.operation,
        error,
    )


async def _prepare_result(
    taskflow: TaskflowWorker,
    work: _LeaseWork,
    outcome: Any,
    segments: _SegmentUploads | None = None,
) -> _LeaseResult | None:
    """Upload a lease's artifact and build its ack and Taskflow report.

    Returns None when the server released the lease, leaving nothing to report.
    """
    try:
        return await _upload_result(taskflow, work, outcome, segments)
    finally:
        # A result's files are the worker's to delete, however the lease ended.
        if isinstance(outcome, InferenceResult):
            outcome.release()


def _media_job(lease: dict[str, Any]) -> bool:
    """A lease the API runs as a media job: its result is delivered to the API before it settles."""
    metadata = lease.get("metadata")
    return isinstance(metadata, dict) and bool(metadata.get("mediaJobId"))


async def _upload_result(
    taskflow: TaskflowWorker,
    work: _LeaseWork,
    outcome: Any,
    segments: _SegmentUploads | None,
) -> _LeaseResult | None:
    lease = work.lease
    lease_id = lease["id"]
    run_id = lease.get("runId")
    if segments is not None and (
        isinstance(outcome, BaseException) or _lease_released(taskflow, work)
    ):
        segments.cancel()
    if _lease_released(taskflow, work):
        return None
    try:
        if isinstance(outcome, BaseException):
            raise outcome
        result = outcome
        if result.kind == "json":
            completion = {"kind": "json", "data": result.data}
            if _media_job(lease):
                await taskflow.deliver_media(lease, completion)
            return _LeaseResult(work, completion=completion)

        artifact_id = lease["payload"].get("artifactId")
        if not artifact_id:
            raise ValueError(
                f"TTS operation {work.operation} returned audio without an artifactId"
            )
        size = result.size
        if not size:
            raise RuntimeError("LZAI returned an empty audio artifact")
        # A result's further files (a continuable clip's lastFrame) each go to the artifact the
        # API named for it; one it didn't name would be lost, so that is the API's bug to fix.
        file_ids = lease["payload"].get("artifactIds") or {}
        files = result.files or {}
        missing = sorted(name for name in files if not isinstance(file_ids.get(name), str))
        if missing:
            raise ValueError(
                f"TTS operation {work.operation} returned files {missing} without artifactIds"
            )
        for name, file in files.items():
            await taskflow.upload(lease, file_ids[name], file.source)
        if segments is not None:
            # A streamed take, whose segments went up as they were encoded: it is in memory.
            await segments.finish(result.audio or b"")
        else:
            await taskflow.upload(lease, artifact_id, result.source)
        duration = await asyncio.to_thread(_artifact_duration, result.source, result.content_type)
        completion: dict[str, Any] = {
            **(result.data or {}),
            "kind": "artifact",
            "artifactId": artifact_id,
            "bytes": size,
            "contentType": result.content_type or "application/octet-stream",
            **(
                {
                    "files": {
                        name: {
                            "artifactId": file_ids[name],
                            "bytes": file.size,
                            "contentType": file.content_type,
                        }
                        for name, file in files.items()
                    }
                }
                if files
                else {}
            ),
        }
        if duration is not None:
            completion["duration"] = duration
        if _media_job(lease):
            await taskflow.deliver_media(lease, completion)
        token = _callback_token(lease)
        ack = (
            {
                "token": token,
                "runId": run_id,
                "artifactId": artifact_id,
                "ok": True,
                **({"duration": duration} if duration is not None else {}),
            }
            if token
            else None
        )
        return _LeaseResult(work, completion=completion, ack=ack)
    except InferenceOperationError as error:
        retry = error.status_code >= 500 or error.status_code in {408, 429}
        _LOGGER.error(
            "LZAI task rejected type=%s lease=%s run=%s operation=%s status=%d retry=%s "
            "detail=%s",
            work.task_type,
            lease_id,
            run_id,
            work.operation,
            error.status_code,
            retry,
            str(error.detail)[:1000],
        )
        token = _callback_token(lease)
        payload_artifact_id = lease.get("payload", {}).get("artifactId")
        ack = None
        if token and not retry and isinstance(payload_artifact_id, str) and payload_artifact_id:
            # Only terminal failures ack; retriable failures get a fresh attempt
            # (and eventually a terminal ack or a run.failed event).
            ack = {
                "token": token,
                "runId": run_id,
                "artifactId": payload_artifact_id,
                "ok": False,
                "error": str(error)[:2000],
            }
        return _LeaseResult(work, error=error, retry=retry, ack=ack)
    except ProtocolError as error:
        if not error.loses_lease:
            raise
        _log_dropped_result(work, error)
        return None
    except Exception as error:
        _LOGGER.exception(
            "LZAI task failed type=%s lease=%s run=%s operation=%s",
            work.task_type,
            lease_id,
            run_id,
            work.operation,
        )
        return _LeaseResult(work, error=error, retry=True)


async def _settle_results(
    taskflow: TaskflowWorker,
    results: list[_LeaseResult],
    ack_failures: dict[str, _AckFailure],
) -> None:
    """Complete or fail a batch's leases in one request, once the API settled their acks."""
    settlements: list[dict[str, Any]] = []
    for result in results:
        work = result.work
        lease = work.lease
        ack_failure = ack_failures.get(result.ack["runId"]) if result.ack is not None else None
        if result.error is not None:
            # A terminal failure's ack is best effort: Taskflow's run.failed
            # event settles the block as well.
            settlements.append(_failed_settlement(lease, result.error, retry=result.retry))
        elif ack_failure is not None:
            _LOGGER.error(
                "LZAI task not acked type=%s lease=%s run=%s operation=%s retry=%s: %s",
                work.task_type,
                lease["id"],
                lease.get("runId"),
                work.operation,
                ack_failure.retry,
                ack_failure.error,
            )
            settlements.append(
                _failed_settlement(
                    lease, RuntimeError(ack_failure.error), retry=ack_failure.retry
                )
            )
        else:
            settlements.append(_completed_settlement(lease, result.completion or {}))
    if not settlements:
        return
    try:
        outcomes = await taskflow.settle(settlements)
    except Exception:
        _LOGGER.exception(
            "Failed to settle LZAI leases with Taskflow leases=%s",
            [settlement["leaseId"] for settlement in settlements],
        )
        raise

    for result, settlement, outcome in zip(results, settlements, outcomes, strict=True):
        work = result.work
        lease = work.lease
        error = outcome.get("error")
        if isinstance(error, dict):
            if error.get("code") in _LEASE_GONE_CODES:
                # The run was cancelled, expired or superseded while we were
                # working on it: the server already reclaimed the work.
                _log_dropped_result(work, f"{error.get('code')}: {error.get('message')}")
            else:
                _LOGGER.error(
                    "Taskflow rejected lease settlement lease=%s run=%s outcome=%s: %s",
                    lease["id"],
                    lease.get("runId"),
                    settlement["outcome"],
                    error,
                )
            continue
        if settlement["outcome"] != "completed":
            continue
        completion = result.completion or {}
        if completion.get("kind") == "json":
            _LOGGER.info(
                "LZAI task completed type=%s lease=%s run=%s operation=%s kind=json "
                "wall_seconds=%.3f",
                work.task_type,
                lease["id"],
                lease.get("runId"),
                work.operation,
                time.monotonic() - work.started_at,
            )
        else:
            _LOGGER.info(
                "LZAI task completed type=%s lease=%s run=%s operation=%s kind=artifact "
                "bytes=%d wall_seconds=%.3f",
                work.task_type,
                lease["id"],
                lease.get("runId"),
                work.operation,
                completion.get("bytes", 0),
                time.monotonic() - work.started_at,
            )


async def _process_leases(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    leases: list[dict[str, Any]],
    acks: SynthesisAckClient | None = None,
) -> None:
    works: list[_LeaseWork] = []
    invalid: list[tuple[dict[str, Any], BaseException]] = []
    for lease in leases:
        try:
            works.append(_prepare_lease(taskflow, lease))
        except Exception as error:
            _LOGGER.exception(
                "Invalid LZAI lease payload lease=%s run=%s",
                lease.get("id"),
                lease.get("runId"),
            )
            invalid.append((lease, error))
    if invalid:
        try:
            await taskflow.settle(
                [_failed_settlement(lease, error, retry=False) for lease, error in invalid]
            )
        except Exception:
            _LOGGER.exception(
                "Failed to reject invalid LZAI leases leases=%s",
                [lease.get("id") for lease, _ in invalid],
            )

    if not works:
        return
    staged = await asyncio.gather(
        *(_stage_lease_inputs(taskflow, work) for work in works), return_exceptions=True
    )
    # A lease whose inputs could not be read fails like its job would, without running.
    staging_errors = {i: item for i, item in enumerate(staged) if isinstance(item, BaseException)}
    works = [item if isinstance(item, _LeaseWork) else work for item, work in zip(staged, works, strict=True)]
    run = [i for i in range(len(works)) if i not in staging_errors]
    uploads = [_segment_uploads(taskflow, work) for work in works]
    reports = [_progress_reports(taskflow, work) for work in works]
    cancel_events = [asyncio.Event() for _ in works]
    for work, event in zip(works, cancel_events, strict=True):
        taskflow.lease_cancel_events[work.lease["id"]] = event
        if work.lease["id"] not in taskflow.active_lease_ids:
            event.set()  # released before it got here
    try:
        ran = (
            await inference.execute_many(
                [(works[i].operation, works[i].request) for i in run],
                mp3_sinks=[uploads[i].write if uploads[i] is not None else None for i in run],
                progress_sinks=[reports[i].report if reports[i] is not None else None for i in run],
                cancel_events=[cancel_events[i] for i in run],
            )
            if run
            else []
        )
        if len(ran) != len(run):
            raise RuntimeError(
                "Inference batch returned a different number of outcomes"
            )
        outcomes: list[Any] = [staging_errors.get(i) for i in range(len(works))]
        for i, outcome in zip(run, ran, strict=True):
            outcomes[i] = outcome
    except ChildWorkerDied:
        # A backend subprocess died. Don't waste retries on these leases; let
        # the worker's serve loop drop the Taskflow session and surface
        # ``error`` on /health so the orchestrator pulls traffic.
        for upload in uploads:
            if upload is not None:
                upload.cancel()
        raise
    except Exception as error:
        _LOGGER.exception("LZAI task batch planning or execution failed")
        outcomes = [staging_errors.get(i, error) for i in range(len(works))]
    try:
        prepared = await asyncio.gather(
            *(
                _prepare_result(taskflow, work, outcome, upload)
                for work, outcome, upload in zip(works, outcomes, uploads, strict=True)
            ),
            return_exceptions=True,
        )
        results = [
            result
            for result in prepared
            if isinstance(result, _LeaseResult) and not _lease_released(taskflow, result.work)
        ]
        # One ack request for the whole batch, then one settle request: a run
        # must not finish until the API has settled its block.
        ack_items = [result.ack for result in results if result.ack is not None]
        ack_failures = await acks.send(ack_items) if ack_items and acks is not None else {}
        await _settle_results(taskflow, results, ack_failures)
    finally:
        for work in works:
            taskflow.active_lease_ids.discard(work.lease["id"])
            if work.staged_dir is not None:
                shutil.rmtree(work.staged_dir, ignore_errors=True)
            taskflow.dropped_lease_ids.pop(work.lease["id"], None)
            taskflow.lease_cancel_events.pop(work.lease["id"], None)
    for result in prepared:
        if isinstance(result, BaseException):
            # Let every lease finish cleaning up before the serve loop drops the
            # session and reconnects (only inference-runtime failures get here).
            raise result


async def _process_lease(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    lease: dict[str, Any],
    acks: SynthesisAckClient | None = None,
) -> None:
    await _process_leases(taskflow, inference, [lease], acks)


def _update_pull_task_types(taskflow: TaskflowWorker) -> None:
    ready = ready_task_types(taskflow.task_types, deferred_engines())
    note = ""
    waiting = [t for t in taskflow.task_types if t not in ready]
    if waiting:
        note = f" (waiting for model sets: {waiting})"
    snapshot = residency_snapshot()
    taskflow.pull_sticky = False
    if snapshot is not None:
        now = time.monotonic()
        if snapshot.resident != taskflow.resident:
            taskflow.resident = snapshot.resident
            taskflow.resident_since = now
        sticky = not taskflow.sticky_missed and now - taskflow.resident_since < taskflow.sticky_seconds
        taskflow.sticky_missed = False  # one full pull per empty sticky pull
        narrowed = on_demand_task_types(ready, snapshot.resident, snapshot.waiting_for, sticky, snapshot.pinned)
        if narrowed != ready:
            taskflow.pull_sticky = snapshot.waiting_for is None
            note += (
                f" (loading {snapshot.waiting_for})"
                if snapshot.waiting_for
                else f" (loaded: {sorted(snapshot.resident)})"
            )
        ready = narrowed
    if ready != taskflow.pull_task_types:
        _LOGGER.info("Pulling task types %s%s", list(ready), note)
        taskflow.pull_task_types = ready


async def _pull_leases(
    taskflow: TaskflowWorker, in_flight: Mapping[str, int], background: int
) -> list[dict[str, Any]] | None:
    """Pull what fits beside the leases in flight (counted by task type, ``background`` of
    them negative-priority); None when nothing fits, so nothing was asked for.

    Only the types whose group (``task_groups``) has a free slot are asked for, at most as
    many as those slots and the box concurrency take. Taskflow's highest-priority ready task
    picks the band for the whole pull. Negative priority (background work) only runs when
    nothing >= 0 is ready, and never more than ``low_priority_batch_size`` at once.
    """
    ready = taskflow.task_types if taskflow.pull_task_types is None else taskflow.pull_task_types
    task_types, room = open_task_types(taskflow.task_groups, in_flight, ready)
    free = min(taskflow.concurrency - sum(in_flight.values()), room)
    if free <= 0:
        return None
    background_room = taskflow.low_priority_batch_size - background
    if background_room <= 0:
        return await taskflow.pull(max_tasks=free, min_priority=0, task_types=task_types)
    return await taskflow.pull(
        max_tasks=free,
        priority_bands=[{"below": 0, "maxTasks": min(free, background_room)}],
        task_types=task_types,
    )


def _transient_failure(error: BaseException) -> bool:
    """A failed request worth retrying in the same session: the network, or the
    API restarting (502/503/504), as opposed to an answer about the session."""
    if isinstance(error, httpx.TransportError):
        return True
    return (
        isinstance(error, ProtocolError)
        and not error.loses_session
        and error.status_code is not None
        and error.status_code >= 500
    )


class _DisconnectWatchdog:
    """Hard-exit the worker once it has not been joined to Taskflow for too long.

    The serve loop reconnects on its own, but a failure it does not know how to
    recover from (a wedged HTTP client, a stuck event loop) would otherwise
    leave the worker reconnecting forever. A pull worker can afford a restart:
    after ``LZAI_MAX_DISCONNECTED_SECONDS`` (default 600, 0 disables) without
    an ``ok`` health status, kill the process tree and let pm2 start it clean.
    Runs on a thread so a blocked event loop cannot keep it from firing.
    """

    def __init__(self) -> None:
        self.limit = float(os.environ.get("LZAI_MAX_DISCONNECTED_SECONDS", "600"))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="taskflow-disconnect-watchdog", daemon=True
        )

    def start(self) -> None:
        if self.limit > 0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last_ok = time.monotonic()
        while not self._stop.wait(min(5.0, self.limit / 4)):
            now = time.monotonic()
            if get_health_status()["status"] == "ok":
                last_ok = now
            elif now - last_ok > self.limit:
                hard_exit(
                    f"not joined to Taskflow for {now - last_ok:.0f}s "
                    f"(LZAI_MAX_DISCONNECTED_SECONDS={self.limit:g})"
                )


async def _serve_taskflow(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    synthesis_capabilities: dict[str, Any],
    acks: SynthesisAckClient | None = None,
) -> None:
    set_status("starting")
    watchdog = _DisconnectWatchdog()
    watchdog.start()
    joined_once = False
    failures = 0
    try:
        while True:
            heartbeat_task: asyncio.Task[None] | None = None
            reason = "taskflow connection lost"
            reconnect = True
            # Leases in flight -> (task type, 1 if negative-priority else 0).
            running: dict[asyncio.Task[None], tuple[str, int]] = {}
            pull_failures = 0
            try:
                if taskflow.session_token:
                    await taskflow.heartbeat()
                else:
                    await taskflow.join({"service": "lzai", "synthesis": synthesis_capabilities, **taskflow.pool_metadata})
                connection_lost = asyncio.Event()
                heartbeat_task = asyncio.create_task(taskflow.heartbeat_loop(connection_lost))
                set_status("ok")
                joined_once = True
                failures = 0
                _LOGGER.info("LZAI Taskflow worker ready id=%s", taskflow.worker_id)
                # Each lease runs as its own task while the next pull goes
                # out: nano-vLLM batches across them, and a lease acks (its
                # block turns Ready) the moment its own take is done instead
                # of waiting for the longest line it was pulled with.
                while not connection_lost.is_set():
                    for task in [task for task in running if task.done()]:
                        del running[task]
                        error = task.exception()
                        if error is not None:
                            # _process_leases only re-raises when the inference
                            # runtime itself is broken (per-lease errors are
                            # handled internally). Drop to ``error`` so the
                            # orchestrator pulls traffic before reconnect storms.
                            reason = f"inference runtime failed: {error}"
                            set_status("error", reason=reason)
                            raise error
                    in_flight = Counter(task_type for task_type, _ in running.values())
                    background = sum(count for _, count in running.values())
                    _update_pull_task_types(taskflow)
                    try:
                        leases = await _pull_leases(taskflow, in_flight, background)
                    except Exception as error:
                        if not _transient_failure(error):
                            raise
                        # The session is still good (the heartbeat says when it
                        # is not): pull again on a fresh connection instead of
                        # draining the batches in flight and rejoining.
                        pull_failures += 1
                        _LOGGER.warning(
                            "Taskflow pull failed; retrying in the same session (attempt %d): %r",
                            pull_failures,
                            error,
                        )
                        await asyncio.sleep(
                            min(_PULL_RETRY_MAX_DELAY, _CONTROL_RETRY_DELAY * 2 ** min(pull_failures - 1, 3))
                        )
                        continue
                    pull_failures = 0
                    if leases is None:
                        # Every group with work to pull is full: wait for a lease to finish.
                        if running:
                            await asyncio.wait(
                                running, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                            )
                        else:
                            await asyncio.sleep(1.0)
                        continue
                    if taskflow.pull_sticky:
                        # Nothing ready for the loaded engines: let the next pull take any type.
                        taskflow.sticky_missed = not leases
                    for lease in leases:
                        task = asyncio.create_task(
                            _process_lease(taskflow, inference, lease, acks)
                        )
                        running[task] = (lease.get("type", "unknown"), int(lease.get("priority", 0) < 0))
            except ProtocolError as error:
                if error.loses_lease:
                    # A lease-level rejection (cancelled run, expired lease) says
                    # nothing about the session, so keep it and keep pulling.
                    # Clearing it here would fence this session and release the
                    # leases of the worker's other in-flight tasks, making them run
                    # a second time.
                    _LOGGER.warning("Taskflow released a lease; keeping the session: %s", error)
                    reconnect = False
                else:
                    if error.loses_session:
                        taskflow.session_id = None
                        taskflow.session_token = None
                    _LOGGER.exception(
                        "Taskflow connection lost; reconnecting without reloading models"
                    )
            except Exception:
                _LOGGER.exception("Taskflow connection lost; reconnecting without reloading models")
            finally:
                # Let the leases in flight finish (and settle) while the
                # heartbeat still renews their leases.
                if running:
                    await asyncio.gather(*running, return_exceptions=True)
                # Once joined, any disconnection is a failure worth surfacing;
                # ``starting`` only means models are up but never joined yet.
                if joined_once:
                    set_status("error", reason=reason)
                else:
                    set_status("starting")
                if heartbeat_task:
                    heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat_task
            if not reconnect:
                await asyncio.sleep(_RECONNECT_BASE_DELAY)
                continue
            # Nothing else uses the client now (the batches are done and the
            # heartbeat stopped), so drop its pool before trying again.
            await taskflow.reset_client()
            delay = min(_RECONNECT_MAX_DELAY, _RECONNECT_BASE_DELAY * 2 ** min(failures, 5))
            failures += 1
            await asyncio.sleep(delay)
    finally:
        watchdog.stop()


def _start_http_server(inference: LzTtsInferenceSession) -> tuple[Any, asyncio.Task]:
    """Serve the in-process HTTP adapter (health probe + dev /task/sync).

    The HTTP server is always on. ``/health`` is unauthenticated and reflects
    whether the worker has joined the Taskflow cluster (``ok``) or is still
    starting up / reconnecting (``starting``). ``/task/sync`` is gated by the
    standard API key middleware and reuses the worker's inference session.
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app = create_app(session=inference)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    return server, asyncio.create_task(server.serve())


def pool_metadata() -> dict[str, object]:
    """Join metadata maydeploy sizes the rented fleet from (/admin/pipeline-stats).

    ``TASKFLOW_WORKER_POOL``: "owned" on our own always-on boxes, "spot" on rented workers;
    omitted when unset. ``TASKFLOW_WORKER_CAPACITY``: 3090-GPU-seconds of work cleared per
    wall second (one RTX 3090 = 1, default 1).
    """
    raw = os.environ.get("TASKFLOW_WORKER_CAPACITY", "").strip() or "1"
    try:
        capacity = float(raw)
    except ValueError:
        raise ValueError(f"TASKFLOW_WORKER_CAPACITY must be a number, got {raw!r}") from None
    if not math.isfinite(capacity) or capacity < 0:
        raise ValueError(f"TASKFLOW_WORKER_CAPACITY must be a non-negative number, got {raw!r}")
    pool = os.environ.get("TASKFLOW_WORKER_POOL", "").strip()
    return {**({"pool": pool} if pool else {}), "capacity": capacity}


async def run_worker() -> None:
    load_dotenv()
    worker_token = os.environ.get("TASKFLOW_WORKER_TOKEN", "").strip()
    if not worker_token:
        raise RuntimeError("TASKFLOW_WORKER_TOKEN is required to authenticate with Lazybird Taskflow")
    lazybird_url = os.environ.get("LZB_API", "http://localhost:4001").rstrip("/")
    taskflow = TaskflowWorker(
        base_url=f"{lazybird_url}/internal/taskflow/v1",
        worker_token=worker_token,
        worker_id=os.environ.get("TASKFLOW_WORKER_ID", f"lzai-{socket.gethostname()}"),
        concurrency=max(1, int(os.environ.get("TASKFLOW_WORKER_CONCURRENCY", "8"))),
        low_priority_batch_size=max(1, int(os.environ.get("TASKFLOW_LOW_PRIORITY_BATCH_SIZE", "3"))),
        persistent=_env_bool("TASKFLOW_WORKER_PERSISTENT", False),
        sticky_seconds=float(os.environ.get("LZAI_SWAP_STICKY_SECONDS", "300")),
    )
    taskflow.pool_metadata = pool_metadata()
    inference = LzTtsInferenceSession()
    taskflow.task_types = served_task_types(inference.config)
    try:
        await inference.start()
    except BaseException:  # pylint: disable=broad-exception-caught
        # A partial startup may have spawned children that already hold GPU
        # memory. Kill the whole tree immediately rather than trusting loop
        # teardown (which can itself wedge on a stuck child).
        _LOGGER.exception("Failed to start the inference runtime; killing the process tree")
        hard_exit("inference runtime startup failed")
    http_server, http_task = _start_http_server(inference)
    # uvicorn installs its own handlers when the server task starts; ours run
    # afterwards so SIGINT/SIGTERM end the whole worker instead of leaving the
    # lease loop running until pm2 SIGKILLs us (which would orphan the children).
    await asyncio.sleep(0)
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    # Backstop: if graceful shutdown wedges (a child stuck in an uninterruptible
    # CUDA call, a thread that will not join), don't wait for pm2 to SIGKILL us
    # at kill_timeout. Tear the whole tree down ourselves so the GPU is free
    # before the supervisor restarts the worker.
    shutdown_grace = max(1.0, float(os.environ.get("LZAI_SHUTDOWN_GRACE_SECONDS", "20")))
    shutdown_watchdog = threading.Timer(
        shutdown_grace,
        hard_exit,
        args=("graceful shutdown exceeded its grace period",),
    )
    shutdown_watchdog.daemon = True
    shutdown_requested = False

    def _request_shutdown() -> None:
        nonlocal shutdown_requested
        # SIGINT and SIGTERM can both arrive; only arm the watchdog once.
        if shutdown_requested:
            return
        shutdown_requested = True
        shutdown_watchdog.start()
        main_task.cancel()

    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(shutdown_signal, _request_shutdown)
    acks = SynthesisAckClient(
        taskflow,
        lazybird_url.rstrip("/") + "/internal/synthesis-events/v1/batch",
    )
    try:
        synthesis_capabilities = inference.synthesis_capabilities()
        await _serve_taskflow(taskflow, inference, synthesis_capabilities, acks)
    except asyncio.CancelledError:
        _LOGGER.info("LZAI worker shutdown requested")
    finally:
        shutdown_watchdog.cancel()
        http_server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await http_task
        set_status("starting")
        await taskflow.close()
        await inference.close()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(run_worker())
    except BaseException:  # pylint: disable=broad-exception-caught
        # Anything that escapes the worker loop means we cannot guarantee the
        # backend children are still healthy. Never leave them holding GPU
        # memory: kill the tree and let the supervisor start us clean.
        _LOGGER.exception("LZAI worker failed; killing the whole process tree")
        hard_exit("unhandled worker error")
