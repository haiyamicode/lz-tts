"""Stopping a job its caller no longer wants: a cancelled lease must free the GPU, not render on for an hour.

The parent sends ``cancel {"job_id"}`` on the channel (it is answered on a thread of its own, so it never
waits behind the render). The job is marked, and the render stops at its next check:
- every ComfyUI graph (``minimax_h3_runtime._execute_graph``) checks before it starts, and a graph already
  running is interrupted through ComfyUI's own flag, which every model op and sampler step reads. ComfyUI
  catches that interrupt itself and only reports the prompt failed, so the graph runner turns a failure of
  a cancelled job into :class:`Cancelled`;
- YuE2 checks :func:`is_cancelled` on every token and ODE step;
- the stages around them (chunks, retakes, ffmpeg, RIFE/SPAN) call :func:`check` between steps.

:class:`Cancelled` is a ``BaseException`` so the ``except Exception`` around validators and helpers lets it
through; the server answers it as a ``cancelled`` error. A job cancelled before it started (still waiting
for the GPU) stops as soon as it enters :func:`job`.

One job renders at a time (the runtime's GPU lock), so the job checked is this module's.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator

#: How many cancelled job ids are remembered for jobs that have not started yet (or never will).
REMEMBERED = 256

_LOCK = threading.Lock()
_ASKED: "OrderedDict[str, None]" = OrderedDict()
_CURRENT: str | None = None


class Cancelled(BaseException):
    """The job was cancelled by its caller."""


def _comfy_interrupt(value: bool) -> None:
    """Set ComfyUI's interrupt flag, if ComfyUI is loaded in this process."""
    model_management = sys.modules.get("comfy.model_management")
    if model_management is not None:
        model_management.interrupt_current_processing(value)


def ask(job_id: str) -> bool:
    """Cancel ``job_id``: the running job stops at its next check, a later one as it starts.

    Returns whether it was the job running now.
    """
    with _LOCK:
        _ASKED[job_id] = None
        _ASKED.move_to_end(job_id)
        while len(_ASKED) > REMEMBERED:
            _ASKED.popitem(last=False)
        running = _CURRENT == job_id
        if running:
            _comfy_interrupt(True)
    return running


@contextlib.contextmanager
def job(job_id: str | None) -> Iterator[None]:
    """Runs the job ``job_id``: :func:`check` inside it raises once it is cancelled (at once if it already is)."""
    global _CURRENT  # pylint: disable=global-statement
    if not job_id:
        yield
        return
    with _LOCK:
        _CURRENT = job_id
    try:
        check()
        yield
    finally:
        with _LOCK:
            _CURRENT = None
            if _ASKED.pop(job_id, False) is None:
                # The interrupt may not have been consumed; it must not stop the next job's first graph.
                _comfy_interrupt(False)


def is_cancelled() -> bool:
    """Whether the job running now has been cancelled."""
    with _LOCK:
        return _CURRENT is not None and _CURRENT in _ASKED


def check() -> None:
    """Raises :class:`Cancelled` when the job running now has been cancelled."""
    if is_cancelled():
        raise Cancelled(f"job {_CURRENT} was cancelled")
