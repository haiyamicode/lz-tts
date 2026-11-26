"""Opening a URL a caller gave us, only ever at a public address.

The caller names the URL, so it may name us: a loopback admin port, the LAN, a cloud metadata address.
Every hop is checked: the first URL and each redirect's ``Location`` are resolved and refused unless
every address is public, and a direct connection connects to the address it vetted, so a name that
resolves public for the check and private for the connect (DNS rebinding) gets nowhere. Through an
HTTP proxy the proxy connects, so the hop check is what stands there. ``LZAI_ALLOW_PRIVATE_URLS=1``
lifts it for local tests against a server on this machine (the worker's ``src/api/downloads.py`` is
the same rule for its own fetches).
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import urllib.request
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit


def public_address(address: str) -> bool:
    """Whether an IP is one anyone on the internet could reach: not loopback, private, link-local,
    shared (CGNAT, our tailnet), reserved, documentation or multicast space."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _private_allowed() -> bool:
    return os.environ.get("LZAI_ALLOW_PRIVATE_URLS", "").strip().lower() in {"1", "true", "yes"}


def _public_addresses(host: str, port: int) -> list[str]:
    """``host``'s addresses, all of them public, or a refusal naming the host."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise URLError(f"{host} does not resolve: {exc}") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(public_address(address) for address in addresses):
        # One private answer is enough: which one a connect picks is not ours to bet on.
        raise URLError(f"{host} is not a public address")
    return addresses


def check_url(url: str) -> None:
    """Refuses a URL whose host is not public (a ``URLError``, like any unreachable link)."""
    if _private_allowed():
        return
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise URLError(f"{url!r} is not an http(s) URL")
    _public_addresses(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))


def _proxy_hosts() -> set[str]:
    return {host for proxy in urllib.request.getproxies().values() if (host := urlsplit(proxy).hostname)}


def _create_public_connection(address: tuple[str, int], timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,
                              source_address: Any = None) -> socket.socket:
    host, port = address
    if _private_allowed() or host in _proxy_hosts():
        # The proxy is ours to reach; where it goes was checked per hop (check_url).
        return socket.create_connection(address, timeout, source_address)
    last_error: OSError | None = None
    for vetted in _public_addresses(host, port):
        try:
            # TLS still verifies and sends SNI for the URL's host: http.client wraps with self.host.
            return socket.create_connection((vetted, port), timeout, source_address)
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


class _PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_public_connection


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_public_connection


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_PublicHTTPSConnection, req, context=self._context)


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001  (urllib's signature)
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_PublicHTTPHandler, _PublicHTTPSHandler, _PublicRedirects)


def urlopen(request: urllib.request.Request, timeout: float) -> http.client.HTTPResponse:
    """``urllib.request.urlopen`` that reaches only public addresses (module docstring)."""
    check_url(request.full_url)
    return _OPENER.open(request, timeout=timeout)
