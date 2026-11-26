"""Caller URLs reach only public addresses (``public_urls.py``): on the first hop, every redirect, and the socket."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import Request

import pytest

import public_urls


@pytest.fixture(name="origin")
def _origin():
    """A loopback server that answers anything with ``ok``."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (BaseHTTPRequestHandler's name)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_only_public_addresses_count_as_public():
    for address in ["127.0.0.1", "10.0.0.2", "192.168.1.20", "172.17.0.1", "100.64.0.3", "169.254.169.254",
                    "0.0.0.0", "192.0.2.1", "224.0.0.1", "::1", "fe80::1%eth0", "fd00::1", "::ffff:127.0.0.1"]:
        assert not public_urls.public_address(address), address
    for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8"]:
        assert public_urls.public_address(address), address


def test_a_url_on_this_machine_is_refused(origin):
    with pytest.raises(URLError, match="not a public address"):
        public_urls.urlopen(Request(f"{origin}/sample"), timeout=5)
    with pytest.raises(URLError, match="not a public address"):
        public_urls.urlopen(Request("http://localhost:1/x"), timeout=5)


def test_a_redirect_to_a_private_address_is_refused(origin):
    # What urllib calls on each 30x before following its Location.
    with pytest.raises(URLError, match="169.254.169.254 is not a public address"):
        public_urls._PublicRedirects().redirect_request(
            Request(f"{origin}/away"), None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/"
        )


def test_a_direct_connection_is_refused_at_connect_time(origin):
    # Whatever got past the hop check, the socket itself never opens to a private address.
    port = int(origin.rsplit(":", 1)[1])
    with pytest.raises(URLError, match="not a public address"):
        public_urls._create_public_connection(("127.0.0.1", port), 5)


def test_local_tests_can_lift_it(origin, monkeypatch):
    monkeypatch.setenv("LZAI_ALLOW_PRIVATE_URLS", "1")
    with public_urls.urlopen(Request(f"{origin}/sample"), timeout=5) as response:
        assert response.read() == b"ok"
