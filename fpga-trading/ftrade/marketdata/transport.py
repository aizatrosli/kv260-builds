"""Small HTTPS GET transports (standard library only) with RTT measurement.

- UrllibTransport: urllib.request, a new TCP+TLS connection per request
  (honours HTTPS_PROXY via urllib's normal proxy handling).
- KeepAliveTransport: one persistent http.client connection (CONNECT tunnel
  through HTTPS_PROXY if set), so steady-state RTT excludes TCP/TLS setup.

Both return Response(status, headers, body, rtt_ms, rx_ts) and raise HttpError
for non-2xx statuses (with Retry-After parsed for 429).
"""
from __future__ import annotations
import gzip
import http.client
import json
import os
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

BASE_URL = "https://api.luno.com"
USER_AGENT = "ftrade-marketdata/0.1 (public market data only)"


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes
    rtt_ms: float
    rx_ts: int            # local receive time, ms since epoch

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class HttpError(Exception):
    def __init__(self, status: int, msg: str = "", retry_after: float | None = None,
                 rtt_ms: float | None = None):
        super().__init__(f"HTTP {status}: {msg[:200]}")
        self.status = status
        self.retry_after = retry_after
        self.rtt_ms = rtt_ms

    @property
    def retryable(self) -> bool:
        return self.status in (0, 408, 425, 429) or self.status >= 500


def _retry_after(headers) -> float | None:
    v = (headers or {}).get("retry-after")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def _decode(body: bytes, headers: dict) -> bytes:
    if headers.get("content-encoding", "").lower() == "gzip":
        return gzip.decompress(body)
    return body


def build_url(path: str, params: dict | None = None, base: str = BASE_URL) -> str:
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    return base + path + q


def now_ms() -> int:
    return int(time.time() * 1000)


class UrllibTransport:
    def __init__(self, base: str = BASE_URL, timeout: float = 10.0):
        self.base, self.timeout = base, timeout

    def get(self, path: str, params: dict | None = None) -> Response:
        req = urllib.request.Request(build_url(path, params, self.base), headers={
            "User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
                hdr = {k.lower(): v for k, v in r.headers.items()}
                status = r.status
        except urllib.error.HTTPError as e:
            rtt = (time.perf_counter() - t0) * 1000
            hdr = {k.lower(): v for k, v in (e.headers or {}).items()}
            raise HttpError(e.code, e.read().decode("utf-8", "replace") if e.fp else str(e),
                            _retry_after(hdr), rtt) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise HttpError(0, f"network error: {e}") from None
        rtt = (time.perf_counter() - t0) * 1000
        return Response(status, hdr, _decode(body, hdr), rtt, now_ms())

    def close(self) -> None:
        pass


def _proxy_for(host: str):
    """(host, port) of the HTTPS proxy from the environment, or None."""
    np = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    for pat in (p.strip() for p in np.split(",")):
        if pat and (host == pat.lstrip("*.") or host.endswith("." + pat.lstrip("*."))):
            return None
    p = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if not p:
        return None
    u = urllib.parse.urlsplit(p if "://" in p else "http://" + p)
    return u.hostname, u.port or 8080


class KeepAliveTransport:
    def __init__(self, base: str = BASE_URL, timeout: float = 10.0):
        u = urllib.parse.urlsplit(base)
        self.host, self.port = u.hostname, u.port or 443
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        self.conn: http.client.HTTPSConnection | None = None

    def _connect(self):
        px = _proxy_for(self.host)
        if px:
            c = http.client.HTTPSConnection(px[0], px[1], timeout=self.timeout, context=self.ctx)
            c.set_tunnel(self.host, self.port)
        else:
            c = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self.ctx)
        return c

    def connect(self) -> None:
        if self.conn is None:
            self.conn = self._connect()
            self.conn.connect()

    def get(self, path: str, params: dict | None = None) -> Response:
        url = path + (("?" + urllib.parse.urlencode(params)) if params else "")
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json",
                "Accept-Encoding": "gzip", "Connection": "keep-alive"}
        for attempt in (0, 1):     # one silent reconnect if the kept-alive socket died
            try:
                if self.conn is None:
                    self.conn = self._connect()
                t0 = time.perf_counter()
                self.conn.request("GET", url, headers=hdrs)
                r = self.conn.getresponse()
                body = r.read()
                rtt = (time.perf_counter() - t0) * 1000
                hdr = {k.lower(): v for k, v in r.getheaders()}
                if r.will_close:
                    self.close()
                break
            except (OSError, http.client.HTTPException) as e:
                self.close()
                if attempt:
                    raise HttpError(0, f"network error: {e}") from None
        body = _decode(body, hdr)
        if not 200 <= r.status < 300:
            raise HttpError(r.status, body.decode("utf-8", "replace"), _retry_after(hdr), rtt)
        return Response(r.status, hdr, body, rtt, now_ms())

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None


def make_transport(kind: str = "keepalive", **kw):
    return {"keepalive": KeepAliveTransport, "urllib": UrllibTransport}[kind](**kw)


@dataclass
class Backoff:
    """Exponential backoff with full jitter; honours Retry-After when given."""
    base: float = 1.0
    cap: float = 60.0
    n: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def next(self, retry_after: float | None = None) -> float:
        d = min(self.cap, self.base * (2 ** self.n))
        self.n += 1
        d = d * (0.5 + 0.5 * self.rng.random())
        if retry_after is not None:
            d = max(d, retry_after)
        return d

    def reset(self) -> None:
        self.n = 0
