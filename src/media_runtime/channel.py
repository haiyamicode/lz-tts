"""The runtime's end of the channel it was spawned with: requests in, replies and progress out.

This runtime is a child process rather than a worker thread for one reason — YuE2 and ComfyUI/H3
need their own venv (py3.11, torch 2.14, cu130), and multiprocessing cannot cross venvs. So the
API spawns it and talks to it over two pipes it opened first: one line of JSON per message, the
read end of one and the write end of the other inherited by name (``LZAI_REQUEST_FD`` /
``LZAI_REPLY_FD``). There is no socket, no port and nothing to listen on; the channel exists only
between this process and its parent, and ends when either does.

A request is ``{"id", "op", "body"}``. Its reply is ``{"id", "result"}``, or ``{"id", "error":
{"kind", "message"}}`` — ``kind`` is ``input`` for a request that was wrong (the caller's fault,
an HTTP 400 in the API) and ``runtime`` for anything else. Two messages carry no id and answer
nothing: ``{"state": {...}}``, sent once the models are loaded or have failed to load, and
``{"progress": {...}}`` as a job goes (``progress.py``).

Requests are handled one per thread. The renders serialize on the GPU lock of their own accord,
and a question about this runtime's state must not queue behind an hour of lipsync.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

#: The inherited pipe ends, as the parent named them (``src/api/media.py``).
REQUEST_FD_VARIABLE = "LZAI_REQUEST_FD"
REPLY_FD_VARIABLE = "LZAI_REPLY_FD"


class RequestError(Exception):
    """A request this runtime will not run: ``kind`` says whose fault it is."""

    def __init__(self, message: str, kind: str = "input"):
        super().__init__(message)
        self.kind = kind


#: Runs one request's body and returns what goes back as its result.
Operation = Callable[[dict[str, Any]], dict[str, Any]]

_write_lock = threading.Lock()
_reply_fd: int | None = None
_request_fd: int | None = None


def connect() -> None:
    """Takes the two inherited pipe ends. Called before anything that reports, so a model load
    that finishes early still has somewhere to say so."""
    global _reply_fd, _request_fd  # pylint: disable=global-statement
    request_fd = os.environ.get(REQUEST_FD_VARIABLE)
    reply_fd = os.environ.get(REPLY_FD_VARIABLE)
    if not request_fd or not reply_fd:
        raise RuntimeError(
            f"{REQUEST_FD_VARIABLE} and {REPLY_FD_VARIABLE} are the channel the API spawns this runtime with"
        )
    _request_fd, _reply_fd = int(request_fd), int(reply_fd)


def send(message: dict[str, Any]) -> None:
    """Writes one message to the parent, under the lock so two threads' lines never interleave.

    A write to a pipe is whole only up to ``PIPE_BUF``, and a lipsync reply carries its metrics and
    its transcript, so the rest of a long line is written in the loop rather than dropped.
    """
    if _reply_fd is None:
        return
    line = memoryview(f"{json.dumps(message)}\n".encode())
    with _write_lock:
        try:
            while line:
                line = line[os.write(_reply_fd, line) :]
        except OSError:
            # The parent is gone; it takes this process with it, and nothing is owed to it.
            pass


def _reply(request_id: Any, operations: dict[str, Operation], op: Any, body: Any) -> None:
    try:
        operation = operations.get(op) if isinstance(op, str) else None
        if operation is None:
            raise RequestError(f"unknown operation: {op!r}")
        if not isinstance(body, dict):
            raise RequestError("a request body must be an object")
        send({"id": request_id, "result": operation(body)})
    except RequestError as error:
        send({"id": request_id, "error": {"kind": error.kind, "message": str(error)}})
    except BaseException as error:  # pylint: disable=broad-exception-caught
        # Every request is answered, whatever went wrong in it — including a SystemExit raised
        # inside a model, which would otherwise end the thread with the caller still waiting.
        send({"id": request_id, "error": {"kind": "runtime", "message": f"{type(error).__name__}: {error}"}})


def serve(operations: dict[str, Operation]) -> None:
    """Runs until the parent closes the channel: every request it sends becomes one reply.

    Returns when the request pipe ends, which is the parent saying there is no more work — or
    having died, in which case this process is on its way out too.
    """
    if _request_fd is None:
        connect()
    with os.fdopen(_request_fd, "r", buffering=1) as requests:
        for line in requests:
            try:
                request = json.loads(line)
                request_id, op, body = request["id"], request["op"], request.get("body") or {}
            except (ValueError, KeyError, TypeError):
                continue
            threading.Thread(
                target=_reply,
                args=(request_id, operations, op, body),
                name=f"request-{request_id}",
                daemon=True,
            ).start()
