"""What a caller points us at is fetched in chunks, to a file or to bytes, and never past its size."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore
import httpx
import pytest

from .downloads import (
    DownloadError,
    _PublicOnlyBackend,
    _refuse_private_hop,
    fetch_bytes,
    fetch_to_file,
    public_address,
)

SAMPLE = b"sample audio" * 1000


@pytest.fixture(name="local")
def _local(monkeypatch):
    """The origin below is on loopback, which a real fetch refuses (see the private-address tests)."""
    monkeypatch.setenv("LZAI_ALLOW_PRIVATE_URLS", "1")


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


def test_a_file_is_fetched_to_where_it_is_read_from(local, origin, tmp_path):
    dest = tmp_path / "nested" / "sample.mp3"
    result = asyncio.run(fetch_to_file(f"{origin}/sample", dest, 1 << 20, url_name="reference_url"))

    assert result == dest
    assert dest.read_bytes() == SAMPLE


def test_a_file_past_its_size_is_refused_without_being_kept(local, origin, tmp_path):
    dest = tmp_path / "sample.mp3"

    with pytest.raises(DownloadError, match="larger than 1 MB"):
        asyncio.run(fetch_to_file(f"{origin}/huge", dest, 1 << 20, url_name="reference_url"))
    # Half a sample left in a cache would be served as the whole one.
    assert not dest.exists()


def test_bytes_are_fetched_for_the_step_that_reads_bytes(local, origin):
    assert asyncio.run(fetch_bytes(f"{origin}/sample", 1 << 20)) == SAMPLE


def test_bytes_past_the_size_stop_at_the_chunk_that_passed_it(local, origin):
    with pytest.raises(DownloadError, match="larger than 1 MB"):
        asyncio.run(fetch_bytes(f"{origin}/huge", 1 << 20))


def test_an_empty_answer_is_the_callers_bad_url(local, origin, tmp_path):
    with pytest.raises(DownloadError, match="empty"):
        asyncio.run(fetch_bytes(f"{origin}/empty", 1 << 20))
    with pytest.raises(DownloadError, match="empty"):
        asyncio.run(fetch_to_file(f"{origin}/empty", tmp_path / "empty.mp3", 1 << 20))


def test_only_public_addresses_count_as_public():
    for address in ["127.0.0.1", "10.0.0.2", "192.168.1.20", "172.17.0.1", "100.64.0.3", "169.254.169.254",
                    "0.0.0.0", "192.0.2.1", "224.0.0.1", "::1", "fe80::1%eth0", "fd00::1", "::ffff:127.0.0.1",
                    "::ffff:192.168.1.20"]:
        assert not public_address(address), address
    for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8"]:
        assert public_address(address), address


def test_a_url_on_this_machine_or_the_lan_is_refused(origin, tmp_path):
    with pytest.raises(DownloadError, match="not a public address"):
        asyncio.run(fetch_bytes(f"{origin}/sample", 1 << 20, url_name="audio_url"))
    with pytest.raises(DownloadError, match="not a public address"):
        asyncio.run(fetch_to_file(f"{origin}/sample", tmp_path / "x.mp3", 1 << 20))
    with pytest.raises(DownloadError, match="not a public address"):
        asyncio.run(fetch_bytes("http://localhost:1/x", 1 << 20))
    assert not (tmp_path / "x.mp3").exists()


def test_every_redirect_hop_is_checked_before_it_is_sent():
    # The hook httpx runs on each hop, the first and every Location after it.
    with pytest.raises(httpx.ConnectError, match="not a public address"):
        asyncio.run(_refuse_private_hop(httpx.Request("GET", "http://169.254.169.254/latest/meta-data/")))


def test_a_direct_connection_is_refused_at_connect_time(origin):
    # Whatever got past the hop check, the socket itself never opens to a private address.
    port = int(origin.rsplit(":", 1)[1])
    backend = _PublicOnlyBackend(httpcore.AnyIOBackend())
    with pytest.raises(httpcore.ConnectError, match="not a public address"):
        asyncio.run(backend.connect_tcp("127.0.0.1", port))
