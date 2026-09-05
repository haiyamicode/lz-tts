"""Headless Taskflow worker for durable LZ-TTS synthesis jobs."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from .api.server import (
    InferenceOperationError,
    LzTtsInferenceSession,
    create_app,
    get_health_status,
    set_status,
)
from .api.worker_common import ChildWorkerDied

_LOGGER = logging.getLogger(__name__)
TASK_TYPES = ("tts-synthesis", "voice-enhance")
_REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)
_CALLBACK_FLUSH_INTERVAL_SECONDS = 2.0
_CALLBACK_MAX_ATTEMPTS = 3


class ProtocolError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TaskflowWorker:
    base_url: str
    worker_token: str
    worker_id: str
    concurrency: int = 8
    session_id: str | None = None
    session_token: str | None = None
    heartbeat_interval: float = 15.0
    active_lease_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

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

    async def _request(self, method: str, path: str, *, token: str, **kwargs: Any) -> httpx.Response:
        response = await self._client.request(
            method,
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})},
            **kwargs,
        )
        if response.is_error:
            raise ProtocolError(
                f"Taskflow {method} {path} failed ({response.status_code}): {response.text[:500]}",
                response.status_code,
            )
        return response

    async def join(self, metadata: dict[str, Any]) -> None:
        response = await self._request(
            "POST",
            "/workers/join",
            token=self.worker_token,
            json={
                "workerId": self.worker_id,
                "ephemeral": False,
                "metadata": metadata,
                "capabilities": {"taskTypes": list(TASK_TYPES), "concurrency": self.concurrency},
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
        for lease_id in (*body.get("cancelledLeaseIds", []), *body.get("lostLeaseIds", [])):
            self.active_lease_ids.discard(lease_id)

    async def heartbeat_loop(self, connection_lost: asyncio.Event) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.heartbeat()
            except Exception:
                _LOGGER.exception("Taskflow worker heartbeat failed")
                connection_lost.set()
                return

    async def pull(
        self,
        *,
        max_tasks: int | None = None,
        wait_ms: int = 1000,
    ) -> list[dict[str, Any]]:
        if not self.session_token:
            raise ProtocolError("Worker has not joined Taskflow")
        try:
            response = await self._request(
                "POST",
                f"/workers/{self.worker_id}/pull",
                token=self.session_token,
                headers=self._session_headers(),
                json={
                    "maxTasks": max_tasks or self.concurrency,
                    "waitMs": wait_ms,
                    "taskTypes": list(TASK_TYPES),
                },
            )
        except ProtocolError as error:
            # A server restart can leave the previous long-poll marker alive briefly in Redis.
            # Keep the valid worker session and heartbeat alive until that marker expires.
            if error.status_code == 409 and "PULL_IN_PROGRESS" in str(error):
                await asyncio.sleep(1)
                return []
            raise
        return response.json()["leases"]

    async def complete(self, lease: dict[str, Any], result: dict[str, Any]) -> None:
        await self._request(
            "POST",
            f"/leases/{lease['id']}/complete",
            token=lease["leaseToken"],
            json={"result": result},
        )

    async def fail(self, lease: dict[str, Any], error: BaseException, *, retry: bool) -> None:
        await self._request(
            "POST",
            f"/leases/{lease['id']}/fail",
            token=lease["leaseToken"],
            json={
                "retry": retry,
                "error": {"message": str(error), "data": {"type": type(error).__name__}},
            },
        )

    async def upload(self, lease: dict[str, Any], artifact_id: str, audio: bytes) -> None:
        artifact_base_url = self.base_url.removesuffix("/taskflow/v1") + "/tts-artifacts/v1"
        started_at = time.monotonic()
        _LOGGER.info(
            "Artifact upload started lease=%s artifact=%s bytes=%d",
            lease["id"],
            artifact_id,
            len(audio),
        )
        response = await self._client.put(
            f"{artifact_base_url}/leases/{lease['id']}/{artifact_id}",
            content=audio,
            headers={
                "Authorization": f"Bearer {lease['leaseToken']}",
                "Content-Type": "application/octet-stream",
            },
        )
        if response.is_error:
            raise ProtocolError(
                f"Artifact upload failed ({response.status_code}): {response.text[:500]}",
                response.status_code,
            )
        _LOGGER.info(
            "Artifact upload completed lease=%s artifact=%s bytes=%d wall_seconds=%.3f",
            lease["id"],
            artifact_id,
            len(audio),
            time.monotonic() - started_at,
        )


@dataclass(frozen=True)
class _LeaseWork:
    lease: dict[str, Any]
    operation: str
    request: dict[str, Any]
    task_type: str
    started_at: float


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


def _token_project_id(token: str) -> str | None:
    """Best-effort decode of the public projectId segment of a callback token.

    Used purely to partition ack batches per project; verification happens
    server-side.
    """
    body = token.split(".", 1)[0]
    try:
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, json.JSONDecodeError):
        return None
    project_id = claims.get("projectId") if isinstance(claims, dict) else None
    return project_id if isinstance(project_id, str) and project_id else None


def _measure_audio_duration(audio: bytes) -> float | None:
    """Measure audio duration in seconds via ffprobe; None when unavailable."""
    if not audio:
        return 0.0
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
            handle.write(audio)
            path = handle.name
    except OSError:
        _LOGGER.exception("Failed to buffer audio for duration measurement")
        return None
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
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


class SynthesisAckBatcher:
    """Batches synthesis completion acks and flushes them per project.

    Acks are partitioned by projectId (one request per project per flush) so a
    project's acks are applied strictly in order and never conflict with each
    other. The API applies acks idempotently, so duplicate batches are safe.
    A permanently dropped batch leaves the block pending; re-saving the
    project re-submits it via the API's synthesis sync.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        *,
        flush_interval: float = _CALLBACK_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._flush_interval = flush_interval
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._flush_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.flush()

    def enqueue(self, item: dict[str, Any]) -> None:
        self._queue.put_nowait(item)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.flush()

    async def flush(self) -> None:
        async with self._flush_lock:
            items: list[dict[str, Any]] = []
            while not self._queue.empty():
                items.append(self._queue.get_nowait())
            if not items:
                return
            groups: dict[str, list[dict[str, Any]]] = {}
            for item in items:
                token = item.get("token", "")
                key = _token_project_id(token) if isinstance(token, str) else None
                groups.setdefault(key or f"run:{item.get('runId')}", []).append(item)
            results = await asyncio.gather(
                *(self._post_group(key, group) for key, group in groups.items()),
                return_exceptions=True,
            )
            for key, outcome in zip(groups, results, strict=True):
                if isinstance(outcome, BaseException):
                    _LOGGER.error(
                        "Dropping %d synthesis acks for %s after retries: %s",
                        len(groups[key]),
                        key,
                        outcome,
                    )

    async def _post_group(self, key: str, items: list[dict[str, Any]]) -> None:
        for attempt in range(_CALLBACK_MAX_ATTEMPTS):
            try:
                response = await self._client.post(self._endpoint, json={"items": items})
                if response.is_error:
                    raise ProtocolError(
                        f"Synthesis ack batch failed ({response.status_code}): "
                        f"{response.text[:500]}",
                        response.status_code,
                    )
                _LOGGER.info(
                    "Synthesis ack batch sent key=%s items=%d",
                    key,
                    len(items),
                )
                return
            except Exception:
                if attempt == _CALLBACK_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(1.0 * (attempt + 1))


def _prepare_lease(taskflow: TaskflowWorker, lease: dict[str, Any]) -> _LeaseWork:
    lease_id = lease["id"]
    run_id = lease.get("runId")
    task_type = lease.get("type", "unknown")
    payload = lease["payload"]
    operation = "voice-enhance" if task_type == "voice-enhance" else payload["operation"]
    request = payload["request"]
    taskflow.active_lease_ids.add(lease_id)
    _LOGGER.info(
        "LZ-TTS task started type=%s lease=%s run=%s operation=%s artifact=%s "
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


async def _finish_lease(
    taskflow: TaskflowWorker,
    work: _LeaseWork,
    outcome: Any,
    acks: SynthesisAckBatcher | None = None,
) -> None:
    lease = work.lease
    lease_id = lease["id"]
    run_id = lease.get("runId")
    try:
        if lease_id not in taskflow.active_lease_ids:
            _LOGGER.warning(
                "Discarding completed inference for inactive lease=%s run=%s",
                lease_id,
                run_id,
            )
            return
        if isinstance(outcome, BaseException):
            raise outcome
        result = outcome
        if result.kind == "json":
            await taskflow.complete(lease, {"kind": "json", "data": result.data})
            _LOGGER.info(
                "LZ-TTS task completed type=%s lease=%s run=%s operation=%s kind=json "
                "wall_seconds=%.3f",
                work.task_type,
                lease_id,
                run_id,
                work.operation,
                time.monotonic() - work.started_at,
            )
            return

        artifact_id = lease["payload"].get("artifactId")
        if not artifact_id:
            raise ValueError(
                f"TTS operation {work.operation} returned audio without an artifactId"
            )
        audio = result.audio or b""
        if not audio:
            raise RuntimeError("LZ-TTS returned an empty audio artifact")
        await taskflow.upload(lease, artifact_id, audio)
        duration = await asyncio.to_thread(_measure_audio_duration, audio)
        completion: dict[str, Any] = {
            "kind": "artifact",
            "artifactId": artifact_id,
            "bytes": len(audio),
            "contentType": result.content_type or "application/octet-stream",
        }
        if duration is not None:
            completion["duration"] = duration
        await taskflow.complete(lease, completion)
        token = _callback_token(lease)
        if token and acks is not None:
            acks.enqueue(
                {
                    "token": token,
                    "runId": run_id,
                    "artifactId": artifact_id,
                    "ok": True,
                    **({"duration": duration} if duration is not None else {}),
                }
            )
        _LOGGER.info(
            "LZ-TTS task completed type=%s lease=%s run=%s operation=%s kind=artifact bytes=%d "
            "wall_seconds=%.3f",
            work.task_type,
            lease_id,
            run_id,
            work.operation,
            len(audio),
            time.monotonic() - work.started_at,
        )
    except InferenceOperationError as error:
        retry = error.status_code >= 500 or error.status_code in {408, 429}
        _LOGGER.error(
            "LZ-TTS task rejected type=%s lease=%s run=%s operation=%s status=%d retry=%s "
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
        if (
            token
            and not retry
            and acks is not None
            and isinstance(payload_artifact_id, str)
            and payload_artifact_id
        ):
            # Only terminal failures ack immediately; retriable failures get a
            # fresh attempt (and eventually a terminal ack or a run.failed event).
            acks.enqueue(
                {
                    "token": token,
                    "runId": run_id,
                    "artifactId": payload_artifact_id,
                    "ok": False,
                    "error": str(error)[:2000],
                }
            )
        try:
            await taskflow.fail(lease, error, retry=retry)
        except Exception:
            _LOGGER.exception(
                "Failed to report rejected LZ-TTS task to Taskflow type=%s lease=%s run=%s",
                work.task_type,
                lease_id,
                run_id,
            )
            raise
    except Exception as error:
        _LOGGER.exception(
            "LZ-TTS task failed type=%s lease=%s run=%s operation=%s",
            work.task_type,
            lease_id,
            run_id,
            work.operation,
        )
        try:
            await taskflow.fail(lease, error, retry=True)
        except Exception:
            _LOGGER.exception(
                "Failed to report failed LZ-TTS task to Taskflow type=%s lease=%s run=%s",
                work.task_type,
                lease_id,
                run_id,
            )
            raise
    finally:
        taskflow.active_lease_ids.discard(lease_id)


async def _process_leases(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    leases: list[dict[str, Any]],
    acks: SynthesisAckBatcher | None = None,
) -> None:
    works: list[_LeaseWork] = []
    for lease in leases:
        try:
            works.append(_prepare_lease(taskflow, lease))
        except Exception as error:
            _LOGGER.exception(
                "Invalid LZ-TTS lease payload lease=%s run=%s",
                lease.get("id"),
                lease.get("runId"),
            )
            try:
                await taskflow.fail(lease, error, retry=False)
            except Exception:
                _LOGGER.exception(
                    "Failed to reject invalid LZ-TTS lease lease=%s run=%s",
                    lease.get("id"),
                    lease.get("runId"),
                )

    if not works:
        return
    try:
        outcomes = await inference.execute_many(
            [(work.operation, work.request) for work in works]
        )
        if len(outcomes) != len(works):
            raise RuntimeError(
                "Inference batch returned a different number of outcomes"
            )
    except ChildWorkerDied:
        # A backend subprocess died. Don't waste retries on these leases; let
        # the worker's serve loop drop the Taskflow session and surface
        # ``error`` on /health so the orchestrator pulls traffic.
        raise
    except Exception as error:
        _LOGGER.exception("LZ-TTS task batch planning or execution failed")
        outcomes = [error] * len(works)
    await asyncio.gather(
        *(
            _finish_lease(taskflow, work, outcome, acks)
            for work, outcome in zip(works, outcomes, strict=True)
        )
    )


async def _process_lease(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    lease: dict[str, Any],
    acks: SynthesisAckBatcher | None = None,
) -> None:
    await _process_leases(taskflow, inference, [lease], acks)


async def _pull_task_batch(taskflow: TaskflowWorker) -> list[dict[str, Any]]:
    leases = await taskflow.pull()
    if not leases or len(leases) >= taskflow.concurrency:
        return leases

    # Keep isolated request latency low. Once a burst is visible, give its
    # remaining submissions time to reach Taskflow and fill the model batch.
    burst = len(leases) > 1
    deadline = time.monotonic() + (1.0 if burst else 0.1)
    while len(leases) < taskflow.concurrency:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break
        more = await taskflow.pull(
            max_tasks=taskflow.concurrency - len(leases),
            wait_ms=max(1, min(100, round(remaining_seconds * 1000))),
        )
        if more:
            leases.extend(more)
            if not burst:
                burst = True
                deadline = time.monotonic() + 1.0
        elif not burst:
            break
    return leases


async def _serve_taskflow(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    synthesis_capabilities: dict[str, Any],
    acks: SynthesisAckBatcher | None = None,
) -> None:
    set_status("starting")
    while True:
        heartbeat_task: asyncio.Task[None] | None = None
        was_ok = get_health_status()["status"] == "ok"
        try:
            if taskflow.session_token:
                await taskflow.heartbeat()
            else:
                await taskflow.join({"service": "lz-tts", "synthesis": synthesis_capabilities})
            connection_lost = asyncio.Event()
            heartbeat_task = asyncio.create_task(taskflow.heartbeat_loop(connection_lost))
            set_status("ok")
            _LOGGER.info("LZ-TTS Taskflow worker ready id=%s", taskflow.worker_id)
            while not connection_lost.is_set():
                leases = await _pull_task_batch(taskflow)
                if leases:
                    try:
                        await _process_leases(taskflow, inference, leases, acks)
                    except Exception as error:
                        # _process_leases only re-raises when the inference
                        # runtime itself is broken (per-lease errors are
                        # handled internally). Drop to ``error`` so the
                        # orchestrator pulls traffic before reconnect storms.
                        set_status("error", reason=f"inference runtime failed: {error}")
                        raise
        except ProtocolError as error:
            if error.status_code in {401, 403, 404}:
                taskflow.session_id = None
                taskflow.session_token = None
            _LOGGER.exception("Taskflow connection lost; reconnecting without reloading models")
        except Exception:
            _LOGGER.exception("Taskflow connection lost; reconnecting without reloading models")
        finally:
            if was_ok:
                set_status("error", reason="taskflow connection lost")
            else:
                set_status("starting")
            if heartbeat_task:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
        await asyncio.sleep(2)


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


async def run_worker() -> None:
    load_dotenv()
    worker_token = os.environ.get("TASKFLOW_WORKER_TOKEN", "").strip()
    if not worker_token:
        raise RuntimeError("TASKFLOW_WORKER_TOKEN is required to authenticate with Lazybird Taskflow")
    lazybird_url = os.environ.get("LZB_API", "http://localhost:4001").rstrip("/")
    taskflow = TaskflowWorker(
        base_url=f"{lazybird_url}/internal/taskflow/v1",
        worker_token=worker_token,
        worker_id=os.environ.get("TASKFLOW_WORKER_ID", f"lz-tts-{socket.gethostname()}"),
        concurrency=max(1, int(os.environ.get("TASKFLOW_WORKER_CONCURRENCY", "8"))),
    )
    inference = LzTtsInferenceSession()
    await inference.start()
    http_server, http_task = _start_http_server(inference)
    acks = SynthesisAckBatcher(
        taskflow._client,
        lazybird_url.rstrip("/") + "/internal/synthesis-events/v1/batch",
    )
    acks.start()
    try:
        synthesis_capabilities = inference.synthesis_capabilities()
        await _serve_taskflow(taskflow, inference, synthesis_capabilities, acks)
    finally:
        http_server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await http_task
        set_status("starting")
        await acks.stop()
        await taskflow.close()
        await inference.close()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_worker())
