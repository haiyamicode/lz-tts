"""The channel the API spawns the media runtime with: requests in, replies, state and progress out."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from .api.media import MediaInputError, MediaRuntime, MediaRuntimeError

RUNTIME_DIR = Path(__file__).resolve().parent / "media_runtime"

#: A stand-in for the media runtime: the real channel, with operations that need no GPU.
FAKE_RUNTIME = """
import argparse, os, sys, time
sys.path.insert(0, {runtime_dir!r})
import channel
import progress

parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--features")
args = parser.parse_args()

channel.connect()
{state}


def echo(body):
    return {{"said": body.get("say"), "features": args.features}}


def refuse(body):
    raise channel.RequestError("say is not a sentence")


def crash(body):
    raise ValueError("the render broke")


def report(body):
    with progress.job(body["jobId"]):
        progress.report("rendering", 1, 4)
    return {{"reported": True}}


def wait(body):
    time.sleep(float(body.get("seconds", 60)))
    return {{}}


def leave(body):
    os._exit(1)


channel.serve({{"echo": echo, "refuse": refuse, "crash": crash, "report": report,
              "wait": wait, "leave": leave}})
"""


def _runtime(tmp_path: Path, state: str = 'channel.send({"state": {"status": "ok", "reason": "ready"}})') -> MediaRuntime:
    script = FAKE_RUNTIME.format(runtime_dir=str(RUNTIME_DIR), state=state)
    (tmp_path / "server.py").write_text(textwrap.dedent(script))
    return MediaRuntime(
        python=sys.executable,
        runtime_dir=str(tmp_path),
        gpu=None,
        ready_timeout_seconds=30.0,
        features=["videogen"],
    )


def test_a_request_is_answered_over_the_channel(tmp_path):
    runtime = _runtime(tmp_path)

    async def run():
        try:
            return await runtime.request("echo", {"say": "hello"}, 30.0)
        finally:
            await asyncio.to_thread(runtime.stop)

    result = asyncio.run(run())
    assert result == {"said": "hello", "features": "videogen"}


def test_a_message_longer_than_the_pipe_arrives_whole(tmp_path):
    """A lipsync request carries its transcript and answers with its metrics: both outgrow the pipe."""
    runtime = _runtime(tmp_path)
    long_say = "x" * (4 << 20)

    async def run():
        try:
            return await runtime.request("echo", {"say": long_say}, 30.0)
        finally:
            await asyncio.to_thread(runtime.stop)

    assert asyncio.run(run())["said"] == long_say


def test_requests_do_not_queue_behind_each_other(tmp_path):
    """The child answers on a thread per request: a quick question overtakes a long render."""
    runtime = _runtime(tmp_path)

    async def run():
        try:
            slow = asyncio.create_task(runtime.request("wait", {"seconds": 5}, 30.0))
            await asyncio.sleep(0.5)
            answered = await asyncio.wait_for(runtime.request("echo", {"say": "quick"}, 10.0), 10.0)
            slow.cancel()
            return answered
        finally:
            await asyncio.to_thread(runtime.stop)

    assert asyncio.run(run())["said"] == "quick"


def test_a_bad_request_is_the_callers_fault(tmp_path):
    runtime = _runtime(tmp_path)

    async def run():
        try:
            with pytest.raises(MediaInputError, match="not a sentence"):
                await runtime.request("refuse", {}, 30.0)
            with pytest.raises(MediaRuntimeError, match="the render broke"):
                await runtime.request("crash", {}, 30.0)
            with pytest.raises(MediaInputError, match="unknown operation"):
                await runtime.request("nonsense", {}, 30.0)
        finally:
            await asyncio.to_thread(runtime.stop)

    asyncio.run(run())


def test_progress_reaches_the_job_it_belongs_to(tmp_path, monkeypatch):
    from .api import media

    reports: list[tuple] = []
    monkeypatch.setattr(media, "report_progress", lambda *args: reports.append(args))
    runtime = _runtime(tmp_path)

    async def run():
        try:
            await runtime.request("report", {"jobId": "videogen-1"}, 30.0)
            # The reply and the report are separate messages; let the last one land.
            await asyncio.sleep(0.2)
        finally:
            await asyncio.to_thread(runtime.stop)

    asyncio.run(run())
    assert ("videogen-1", "rendering", 1, 4) in reports


def test_a_runtime_that_exits_fails_what_it_was_asked(tmp_path):
    runtime = _runtime(tmp_path)

    async def run():
        try:
            with pytest.raises(MediaRuntimeError, match="exited"):
                await runtime.request("leave", {}, 30.0)
        finally:
            await asyncio.to_thread(runtime.stop)

    asyncio.run(run())


def test_a_runtime_that_cannot_load_is_not_asked(tmp_path):
    runtime = _runtime(
        tmp_path,
        state='channel.send({"state": {"status": "error", "reason": "model load failed: no weights"}})',
    )

    async def run():
        try:
            with pytest.raises(MediaRuntimeError, match="no weights"):
                await runtime.request("echo", {"say": "hello"}, 30.0)
        finally:
            await asyncio.to_thread(runtime.stop)

    asyncio.run(run())
