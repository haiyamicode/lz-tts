"""Where the job being rendered has got to, told to the process that launched this runtime.

A render holds the GPU for minutes, and a long lipsync for an hour or more; until its POST answers,
nothing at all reaches the person waiting for it. Each chunk finished is written as one line to the
pipe the parent opened before it spawned us (``LZAI_PROGRESS_FD``), and the parent hands it to the
lease's own sink and on to LazyBird. A giant job is then visibly moving rather than apparently
stuck.

The pipe is the parent's own: no port, no service, nothing that can be down while the render runs.
Reports go from a thread of their own, at most one a second, latest wins: a render never waits on
one, and one that fails is dropped — the progress is only shown, and losing it costs nothing.

One job renders at a time (the runtime's GPU lock), so the job being reported is this module's.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from typing import Any, Literal

#: The stages a job is reported in; they are LazyBird's own ``MediaJobStage``.
Stage = Literal["preparing", "rendering", "finishing"]

#: The least time between two reports of one job, seconds.
INTERVAL_SECONDS = 1.0
#: The inherited write end of the parent's progress pipe, as the parent named it.
PROGRESS_FD_VARIABLE = "LZAI_PROGRESS_FD"

_LOCK = threading.Lock()
_CURRENT: "_Job | None" = None
#: Set once the pipe has gone (the parent closed it, or it was never there).
_closed_pipe = False


class _Job:
    """The job being rendered, and the report waiting to go up for it.

    Only the latest report is kept: one that is overtaken while the one before it is still in
    flight was never worth sending, and a render must not queue up behind its own progress.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        #: The job counts chunks of its own, so the sampler's steps are not its progress.
        self.counts_chunks = False
        self._pending: tuple[Stage, int | None, int | None] | None = None
        self._closed = False
        self._wake = threading.Condition()
        self._thread = threading.Thread(target=self._send_loop, name=f"progress-{job_id}", daemon=True)
        self._thread.start()

    def report(self, stage: Stage, done: int | None, total: int | None) -> None:
        with self._wake:
            self._pending = (stage, done, total)
            self._wake.notify()

    def close(self) -> None:
        """Stop reporting once the pending report has gone; the render waits for neither."""
        with self._wake:
            self._closed = True
            self._wake.notify()

    def _send_loop(self) -> None:
        while True:
            with self._wake:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                pending, self._pending = self._pending, None
                if pending is None:
                    return
            _send(self.job_id, *pending)
            with self._wake:
                self._wake.wait(INTERVAL_SECONDS)


def _send(job_id: str, stage: Stage, done: int | None, total: int | None) -> None:
    """Writes one report up the pipe, as a line of JSON. One write, so lines never interleave."""
    global _closed_pipe  # pylint: disable=global-statement
    fd = os.environ.get(PROGRESS_FD_VARIABLE)
    if not fd or _closed_pipe:
        return
    body: dict[str, Any] = {"jobId": job_id, "stage": stage}
    if done is not None and total is not None:
        body["done"], body["total"] = int(done), int(total)
    line = f"{json.dumps(body)}\n".encode()
    try:
        os.write(int(fd), line)
    except Exception:  # pylint: disable=broad-exception-caught
        # The parent is gone or no longer listening; the render carries on without reporting.
        _closed_pipe = True


@contextlib.contextmanager
def job(job_id: str | None) -> Iterator[None]:
    """Reports what happens inside as ``job_id``'s progress. Without an id, nothing is reported."""
    global _CURRENT  # pylint: disable=global-statement
    if not job_id:
        yield
        return
    started = _Job(job_id)
    with _LOCK:
        _CURRENT = started
    try:
        yield
    finally:
        with _LOCK:
            if _CURRENT is started:
                _CURRENT = None
        started.close()


def report(stage: Stage, done: int | None = None, total: int | None = None) -> None:
    """Says where the job is: its stage, and the chunks it has finished of the chunks there are."""
    with _LOCK:
        current = _CURRENT
    if current is None:
        return
    if total is not None:
        current.counts_chunks = True
    current.report(stage, done, total)


def step(value: int, maximum: int) -> None:
    """One sampler step, as the job's progress — for a job that is not cut into chunks of its own.

    A lipsync counts its chunks, and each one's steps start again from one; its chunks are the
    progress, and these are dropped.
    """
    with _LOCK:
        current = _CURRENT
    if current is None or current.counts_chunks:
        return
    current.report("rendering", value, maximum)
