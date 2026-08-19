"""Headless Taskflow worker for durable LZ-TTS synthesis jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from .api.server import create_app

_LOGGER = logging.getLogger(__name__)
TASK_TYPE = "tts-synthesis"
OPERATION_PATHS = {
    "synthesize": "/synthesize",
    "voice-convert": "/vc",
    "enhance": "/enhance",
    "find-voice": "/find-voice",
}


class ProtocolError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TaskflowWorker:
    base_url: str
    join_token: str
    worker_id: str
    concurrency: int = 1
    session_id: str | None = None
    session_token: str | None = None
    heartbeat_interval: float = 15.0
    active_lease_ids: set[str] = field(default_factory=set)
    active_lease_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=65.0))

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
            token=self.join_token,
            json={
                "workerId": self.worker_id,
                "ephemeral": False,
                "metadata": metadata,
                "capabilities": {"taskTypes": [TASK_TYPE], "concurrency": self.concurrency},
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
            task = self.active_lease_tasks.get(lease_id)
            if task and not task.done():
                task.cancel()

    async def heartbeat_loop(self, connection_lost: asyncio.Event) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.heartbeat()
            except Exception:
                _LOGGER.exception("Taskflow worker heartbeat failed")
                connection_lost.set()
                return

    async def pull(self) -> list[dict[str, Any]]:
        if not self.session_token:
            raise ProtocolError("Worker has not joined Taskflow")
        try:
            response = await self._request(
                "POST",
                f"/workers/{self.worker_id}/pull",
                token=self.session_token,
                headers=self._session_headers(),
                json={"maxTasks": self.concurrency, "waitMs": 1000, "taskTypes": [TASK_TYPE]},
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


async def _process_lease(
    taskflow: TaskflowWorker,
    inference: httpx.AsyncClient,
    lease: dict[str, Any],
) -> None:
    lease_id = lease["id"]
    taskflow.active_lease_ids.add(lease_id)
    current_task = asyncio.current_task()
    if current_task is not None:
        taskflow.active_lease_tasks[lease_id] = current_task
    try:
        payload = lease["payload"]
        operation = payload["operation"]
        path = OPERATION_PATHS.get(operation)
        if path is None:
            raise ValueError(f"Unsupported TTS operation: {operation}")
        response = await inference.post(path, json=payload["request"])
        if response.is_error:
            raise httpx.HTTPStatusError(
                f"LZ-TTS synthesis failed ({response.status_code}): {response.text[:1000]}",
                request=response.request,
                response=response,
            )

        content_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        if content_type == "application/json":
            await taskflow.complete(lease, {"kind": "json", "data": response.json()})
            return

        artifact_id = payload.get("artifactId")
        if not artifact_id:
            raise ValueError(f"TTS operation {operation} returned audio without an artifactId")
        audio = response.content
        if not audio:
            raise RuntimeError("LZ-TTS returned an empty audio artifact")
        await taskflow.upload(lease, artifact_id, audio)
        await taskflow.complete(
            lease,
            {
                "kind": "artifact",
                "artifactId": artifact_id,
                "bytes": len(audio),
                "contentType": content_type,
            },
        )
    except asyncio.CancelledError:
        _LOGGER.info("Cancelled superseded TTS task lease=%s run=%s", lease_id, lease.get("runId"))
    except httpx.HTTPStatusError as error:
        retry = error.response.status_code >= 500 or error.response.status_code in {408, 429}
        await taskflow.fail(lease, error, retry=retry)
    except Exception as error:
        _LOGGER.exception("TTS task failed lease=%s run=%s", lease_id, lease.get("runId"))
        with contextlib.suppress(Exception):
            await taskflow.fail(lease, error, retry=True)
    finally:
        taskflow.active_lease_ids.discard(lease_id)
        taskflow.active_lease_tasks.pop(lease_id, None)


async def _load_synthesis_capabilities(inference: httpx.AsyncClient) -> dict[str, Any]:
    while True:
        try:
            response = await inference.get("/synthesize/capabilities")
            response.raise_for_status()
            return response.json()
        except Exception:
            _LOGGER.exception("LZ-TTS is not ready; retrying synthesis capabilities")
            await asyncio.sleep(5)


async def _serve_taskflow(
    taskflow: TaskflowWorker,
    inference: httpx.AsyncClient,
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
                leases = await taskflow.pull()
                if leases:
                    await asyncio.gather(
                        *(_process_lease(taskflow, inference, lease) for lease in leases)
                    )
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
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_KEY is required to join Lazybird Taskflow")
    lazybird_url = os.environ.get("LZB_API", "http://localhost:4001").rstrip("/")
    taskflow = TaskflowWorker(
        base_url=f"{lazybird_url}/internal/taskflow/v1",
        join_token=api_key,
        worker_id=os.environ.get("TASKFLOW_WORKER_ID", f"lz-tts-{socket.gethostname()}"),
        concurrency=max(1, int(os.environ.get("TASKFLOW_WORKER_CONCURRENCY", "1"))),
    )
    app = create_app()
    await app.router.startup()
    transport = httpx.ASGITransport(app=app)
    inference = httpx.AsyncClient(
        transport=transport,
        base_url="http://lz-tts.internal",
        headers={"X-Api-Key": api_key},
        timeout=None,
    )
    try:
        synthesis_capabilities = await _load_synthesis_capabilities(inference)
        await _serve_taskflow(taskflow, inference, synthesis_capabilities)
    finally:
        await inference.aclose()
        await taskflow.close()
        await app.router.shutdown()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run_worker())
