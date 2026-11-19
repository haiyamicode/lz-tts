"""Where the job being rendered has got to, told to the process that launched this runtime.

A render holds the GPU for minutes, and a long lipsync for an hour or more; until its reply goes
back, nothing at all reaches the person waiting for it. Each sampler step and each chunk finished
is sent up the channel the parent spawned us with (``channel.py``) as a message answering nothing,
and the parent hands it to the lease's own sink and on to LazyBird. A giant job is then visibly
moving rather than apparently stuck.

Reports go from a thread of their own, at most one a second, latest wins: a render never waits on
one, and one that fails is dropped — the progress is only shown, and losing it costs nothing.

One job renders at a time (the runtime's GPU lock), so the job being reported is this module's.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any, Literal

import channel

#: The stages a job is reported in; they are LazyBird's own ``MediaJobStage``.
Stage = Literal["preparing", "rendering", "finishing"]

#: The least time between two reports of one job, seconds.
INTERVAL_SECONDS = 1.0

#: A chunk's worth of a job, in the units its rendering is reported in, so a sampler step inside
#: one is a whole number of them (LazyBird shows done / total as a percent).
CHUNK_UNITS = 100
#: How far into its chunk the render's last sampler step lands; the decode and the checks are the
#: rest. One 256 s lipsync chunk on the dev 3090 (2026-11-19): setup 48 s, 3 steps 103 s, decode
#: 71 s, checks 34 s, so the steps end 59% of the way in.
CHUNK_STEPS_SHARE = 0.6

_LOCK = threading.Lock()
_CURRENT: "_Job | None" = None


class _Job:
    """The job being rendered, and the report waiting to go up for it.

    Only the latest report is kept: one that is overtaken while the one before it is still in
    flight was never worth sending, and a render must not queue up behind its own progress.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        #: The job counts chunks of its own: a sampler step moves it only inside `chunk_steps`.
        self.counts_chunks = False
        #: The chunk being rendered (its index, the chunks there are), and the units reported so far.
        self.chunk: tuple[int, int] | None = None
        self.units = 0
        #: Inside `chunk_steps`: the chunk's own render is sampling.
        self.chunk_sampling = False
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
    """Sends one report to the parent, as a message of the channel that carries the job itself."""
    body: dict[str, Any] = {"jobId": job_id, "stage": stage}
    if done is not None and total is not None:
        body["done"], body["total"] = int(done), int(total)
    channel.send({"progress": body})


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
    if stage == "rendering" and done is not None and total:
        # Chunks are reported in units, so the steps inside one (`chunk_steps`) can move it too.
        current.chunk = (int(done), int(total))
        current.units = int(done) * CHUNK_UNITS
        done, total = current.units, int(total) * CHUNK_UNITS
    current.report(stage, done, total)


@contextlib.contextmanager
def chunk_steps() -> Iterator[None]:
    """The sampler steps inside are the current chunk's own render, and move the job along it.

    Only the chunk's render counts: a boundary restore's preview and img2img sample too, on their
    own schedules, and are not the chunk's progress. A retake samples again from the first step;
    the job never goes back, so it holds where the first take got to.
    """
    with _LOCK:
        current = _CURRENT
    if current is None:
        yield
        return
    current.chunk_sampling = True
    try:
        yield
    finally:
        current.chunk_sampling = False


def step(value: int, maximum: int) -> None:
    """One sampler step, as the job's progress.

    A job that is not cut into chunks reports its steps as they are. A job that counts its chunks
    (a lipsync, whose steps start again from one in every chunk) moves inside the current chunk
    with its render's steps only (`chunk_steps`), up to `CHUNK_STEPS_SHARE` of it.
    """
    with _LOCK:
        current = _CURRENT
    if current is None or not maximum:
        return
    if not current.counts_chunks:
        current.report("rendering", value, maximum)
        return
    if not current.chunk_sampling or current.chunk is None:
        return
    index, chunks = current.chunk
    units = index * CHUNK_UNITS + round(CHUNK_UNITS * CHUNK_STEPS_SHARE * min(1.0, value / maximum))
    if units <= current.units:
        return
    current.units = units
    current.report("rendering", units, chunks * CHUNK_UNITS)
