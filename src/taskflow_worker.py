"""Headless Taskflow worker for durable LZ-TTS synthesis jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from .api.server import InferenceOperationError, LzTtsInferenceSession

_LOGGER = logging.getLogger(__name__)
TASK_TYPES = ("tts-synthesis", "voice-enhance")
_REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)


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
        await taskflow.complete(
            lease,
            {
                "kind": "artifact",
                "artifactId": artifact_id,
                "bytes": len(audio),
                "contentType": result.content_type or "application/octet-stream",
            },
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
    except Exception as error:
        _LOGGER.exception("LZ-TTS task batch planning or execution failed")
        outcomes = [error] * len(works)
    await asyncio.gather(
        *(
            _finish_lease(taskflow, work, outcome)
            for work, outcome in zip(works, outcomes, strict=True)
        )
    )


async def _process_lease(
    taskflow: TaskflowWorker,
    inference: LzTtsInferenceSession,
    lease: dict[str, Any],
) -> None:
    await _process_leases(taskflow, inference, [lease])


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
) -> None:
    while True:
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            if taskflow.session_token:
                await taskflow.heartbeat()
            else:
                await taskflow.join({"service": "lz-tts", "synthesis": synthesis_capabilities})
            connection_lost = asyncio.Event()
            heartbeat_task = asyncio.create_task(taskflow.heartbeat_loop(connection_lost))
            _LOGGER.info("LZ-TTS Taskflow worker ready id=%s", taskflow.worker_id)
            while not connection_lost.is_set():
                leases = await _pull_task_batch(taskflow)
                if leases:
                    await _process_leases(taskflow, inference, leases)
        except ProtocolError as error:
            if error.status_code in {401, 403, 404}:
                taskflow.session_id = None
                taskflow.session_token = None
            _LOGGER.exception("Taskflow connection lost; reconnecting without reloading models")
        except Exception:
            _LOGGER.exception("Taskflow connection lost; reconnecting without reloading models")
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
        await asyncio.sleep(2)


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
    try:
        synthesis_capabilities = inference.synthesis_capabilities()
        await _serve_taskflow(taskflow, inference, synthesis_capabilities)
    finally:
        await taskflow.close()
        await inference.close()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_worker())
