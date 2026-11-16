"""What a caller points us at is fetched in chunks, to a file or to bytes, and never past its size."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .downloads import DownloadError, fetch_bytes, fetch_to_file

SAMPLE = b"sample audio" * 1000


@pytest.fixture(name="origin")
def _origin():
    """A server that answers /sample with the clip, /huge with far more, and /empty with nothing."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (BaseHTTPRequestHandler's name)
            body = {"/sample": SAMPLE, "/empty": b""}.get(self.path)
            self.send_response(200)
            if body is None:
                # Chunked and endless: a cap that only reads Content-Length would never see this.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # The client hangs up as soon as it has read enough; that is not ours to mind.
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    for _ in range(512):  # several MB, past any cap these tests set
                        self.wfile.write(b"%x\r\n" % len(SAMPLE) + SAMPLE + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_a_file_is_fetched_to_where_it_is_read_from(origin, tmp_path):
    dest = tmp_path / "nested" / "sample.mp3"
    result = asyncio.run(fetch_to_file(f"{origin}/sample", dest, 1 << 20, url_name="reference_url"))

    assert result == dest
    assert dest.read_bytes() == SAMPLE


def test_a_file_past_its_size_is_refused_without_being_kept(origin, tmp_path):
    dest = tmp_path / "sample.mp3"

    with pytest.raises(DownloadError, match="larger than 1 MB"):
        asyncio.run(fetch_to_file(f"{origin}/huge", dest, 1 << 20, url_name="reference_url"))
    # Half a sample left in a cache would be served as the whole one.
    assert not dest.exists()


def test_bytes_are_fetched_for_the_step_that_reads_bytes(origin):
    assert asyncio.run(fetch_bytes(f"{origin}/sample", 1 << 20)) == SAMPLE


def test_bytes_past_the_size_stop_at_the_chunk_that_passed_it(origin):
    with pytest.raises(DownloadError, match="larger than 1 MB"):
        asyncio.run(fetch_bytes(f"{origin}/huge", 1 << 20))


def test_an_empty_answer_is_the_callers_bad_url(origin, tmp_path):
    with pytest.raises(DownloadError, match="empty"):
        asyncio.run(fetch_bytes(f"{origin}/empty", 1 << 20))
    with pytest.raises(DownloadError, match="empty"):
        asyncio.run(fetch_to_file(f"{origin}/empty", tmp_path / "empty.mp3", 1 << 20))
