"""Polling collector for Luno public market data (no API key, read-only).

Endpoints (https://www.luno.com/en/developers/api):
  GET /api/1/orderbook_top?pair=   top 100 bids/asks, aggregated per price
  GET /api/1/orderbook?pair=       full book, one entry per *order* (aggregated here)
  GET /api/1/trades?pair=&since=   <=100 trades executed after `since` (ms, must be
                                   within 24h), returned newest->oldest; each has
                                   sequence, timestamp, price, volume, is_buy
Limits: 300 calls/minute per client (HTTP 429 beyond that); market data may be
cached by Luno for up to 1 s, so polling faster than ~1 Hz buys nothing.

Output: normalized ftrade.events.Event objects (SPEC.md). Book snapshots are
diffed into `level` events; `clear` + a full snapshot is emitted at start, after
any request error (gap) and every `resync_every` polls. Trades are deduplicated
by Luno trade sequence; taker side = "buy" if is_buy else "sell".

Nothing here places orders; only GETs of public endpoints are made.
"""
from __future__ import annotations
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from ..events import Event
from .book import BookDiffer, aggregate, price_to_int, vol_to_units
from .transport import Backoff, HttpError, make_transport, now_ms

RATE_LIMIT_PER_MIN = 300
TRADES_PAGE = 100


def trade_event(t: dict, rx_ts: Optional[int] = None) -> Event:
    return Event("trade", int(t["timestamp"]), "buy" if t.get("is_buy") else "sell",
                 price_to_int(t["price"]), vol_to_units(t["volume"]), rx_ts=rx_ts)


class TradeTracker:
    """Keeps the `since` cursor for /api/1/trades and drops duplicates.

    `since` is set to the newest timestamp seen (not +1) so trades sharing that
    millisecond are not lost; the seen-sequence set removes the re-delivered ones.
    """

    def __init__(self, since_ms: int, memory: int = 20000):
        self.since = int(since_ms)
        self.start = int(since_ms)
        self._seen: set[int] = set()
        self._order: deque[int] = deque()
        self.memory = memory
        self.duplicates = 0

    def _remember(self, s: int) -> None:
        self._seen.add(s)
        self._order.append(s)
        while len(self._order) > self.memory:
            self._seen.discard(self._order.popleft())

    def ingest(self, trades: list | None, rx_ts: Optional[int] = None) -> list[Event]:
        """New trades from one /trades response, oldest first."""
        trades = trades or []
        fresh = []
        for t in trades:
            s = int(t["sequence"])
            if s in self._seen or int(t["timestamp"]) < self.start:
                self.duplicates += s in self._seen
                continue
            self._remember(s)
            fresh.append(t)
        fresh.sort(key=lambda t: (int(t["timestamp"]), int(t["sequence"])))
        if trades:
            newest = max(int(t["timestamp"]) for t in trades)
            if len(trades) >= TRADES_PAGE and newest <= self.since:
                newest = self.since + 1      # a full page inside one ms: step past it
            self.since = max(self.since, newest)
        return [trade_event(t, rx_ts) for t in fresh]


@dataclass
class RequestLog:
    """Per-request record: local send time, endpoint, HTTP status, RTT."""
    rows: list = field(default_factory=list)
    keep: bool = True

    def add(self, path: str, status: int, rtt_ms: Optional[float], nbytes: int = 0) -> None:
        if self.keep:
            self.rows.append({"t": now_ms(), "path": path, "status": status,
                              "rtt_ms": None if rtt_ms is None else round(rtt_ms, 2),
                              "bytes": nbytes})

    def summary(self) -> dict:
        out = {}
        for path in sorted({r["path"] for r in self.rows}):
            rs = [r for r in self.rows if r["path"] == path]
            rtts = sorted(r["rtt_ms"] for r in rs if r["status"] == 200 and r["rtt_ms"] is not None)
            out[path] = {"requests": len(rs), "errors": sum(r["status"] != 200 for r in rs),
                         **rtt_stats(rtts)}
        return out


def rtt_stats(vals) -> dict:
    v = sorted(vals)
    if not v:
        return {"n": 0}
    pick = lambda q: v[min(len(v) - 1, max(0, int(-(-q * len(v) // 1)) - 1))]   # nearest rank
    return {"n": len(v), "min_ms": round(v[0], 2), "median_ms": round(_median(v), 2),
            "p90_ms": round(pick(0.90), 2), "p99_ms": round(pick(0.99), 2),
            "max_ms": round(v[-1], 2), "mean_ms": round(sum(v) / len(v), 2)}


def _median(v):
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


class RestCollector:
    """Poll Luno REST market data and yield normalized events.

    collector = RestCollector("XBTMYR", interval=1.0)
    for ev in collector.events(seconds=60): ...

    interval      seconds between book polls (>= 0.5; Luno caches ~1 s)
    trades_every  poll /trades every k-th cycle
    resync_every  emit clear + full snapshot every K polls (0 = only at start/gaps)
    depth         keep only the best `depth` levels per side (0 = all returned)
    book          "top" (/orderbook_top, 100 levels, small) or "full" (/orderbook)
    """

    def __init__(self, pair: str = "XBTMYR", interval: float = 1.0, trades_every: int = 1,
                 resync_every: int = 300, depth: int = 0, book: str = "top",
                 transport=None, http: str = "keepalive", log: Optional[RequestLog] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 verbose: bool = False, max_trade_pages: int = 5):
        if interval < 0.5:
            raise ValueError("interval must be >= 0.5 s (Luno caches market data ~1 s)")
        calls = 60.0 / interval * (1 + 1.0 / max(1, trades_every))
        if calls > RATE_LIMIT_PER_MIN * 0.9:
            raise ValueError(f"{calls:.0f} calls/min exceeds the Luno budget "
                             f"({RATE_LIMIT_PER_MIN}/min); raise interval or trades_every")
        if book not in ("top", "full"):
            raise ValueError("book must be 'top' or 'full'")
        self.pair = pair
        self.interval = interval
        self.trades_every = max(1, trades_every)
        self.resync_every = resync_every
        self.book_path = "/api/1/orderbook_top" if book == "top" else "/api/1/orderbook"
        self.transport = transport or make_transport(http)
        self.log = log if log is not None else RequestLog()
        self.clock, self.sleep = clock, sleep
        self.verbose = verbose
        self.max_trade_pages = max_trade_pages
        self.differ = BookDiffer(depth)
        self.backoff = Backoff()
        self.trades: Optional[TradeTracker] = None
        self.polls = 0
        self.errors = 0
        self.seq = 0
        self.counts = {"clear": 0, "level": 0, "trade": 0}

    # -- requests ---------------------------------------------------------
    def _get(self, path: str, params: dict):
        try:
            r = self.transport.get(path, params)
        except HttpError as e:
            self.log.add(path, e.status, e.rtt_ms)
            self.errors += 1
            if not e.retryable:
                raise
            d = self.backoff.next(e.retry_after)
            self._say(f"{path}: {e} -> backoff {d:.1f}s")
            self.sleep(d)
            return None
        self.log.add(path, r.status, r.rtt_ms, len(r.body))
        self.backoff.reset()
        return r

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[marketdata] {msg}", file=sys.stderr, flush=True)

    def _number(self, evs: list[Event]) -> list[Event]:
        for e in evs:
            self.seq += 1
            e.seq = self.seq
            self.counts[e.kind] += 1
        return evs

    # -- one cycle --------------------------------------------------------
    def poll_book(self) -> list[Event]:
        r = self._get(self.book_path, {"pair": self.pair})
        if r is None:
            self.differ.reset()            # gap: next good snapshot is a full resync
            return []
        d = r.json()
        resync = bool(self.resync_every) and self.polls > 0 and self.polls % self.resync_every == 0
        self.polls += 1
        ts = int(d.get("timestamp") or r.rx_ts)
        return self.differ.update(aggregate(d.get("bids")), aggregate(d.get("asks")),
                                  ts, r.rx_ts, resync=resync)

    def poll_trades(self) -> list[Event]:
        out: list[Event] = []
        for _ in range(self.max_trade_pages):
            r = self._get("/api/1/trades", {"pair": self.pair, "since": self.trades.since})
            if r is None:
                break
            tr = r.json().get("trades") or []
            out += self.trades.ingest(tr, r.rx_ts)
            if len(tr) < TRADES_PAGE:
                break
        return out

    def cycle(self, i: int) -> list[Event]:
        evs = self.poll_book()
        if i % self.trades_every == 0:
            evs += self.poll_trades()
        return self._number(evs)

    # -- main loop --------------------------------------------------------
    def events(self, seconds: Optional[float] = None, max_polls: Optional[int] = None) -> Iterator[Event]:
        t_start = self.clock()
        if self.trades is None:
            self.trades = TradeTracker(int(t_start * 1000))
        i = 0
        next_t = t_start
        try:
            while True:
                if seconds is not None and self.clock() - t_start >= seconds:
                    break
                if max_polls is not None and i >= max_polls:
                    break
                yield from self.cycle(i)
                i += 1
                next_t += self.interval
                delay = next_t - self.clock()
                if delay > 0:
                    self.sleep(delay)
                else:
                    next_t = self.clock()      # fell behind (slow RTT / backoff); don't burst
                if self.verbose and i % 30 == 0:
                    self._say(f"poll {i}: {self.counts} errors={self.errors}")
        except KeyboardInterrupt:
            self._say("interrupted; closing")
        finally:
            self.transport.close()


def poll(pair: str = "XBTMYR", seconds: Optional[float] = None, **kw) -> Iterator[Event]:
    """Convenience generator: RestCollector(pair, **kw).events(seconds)."""
    return RestCollector(pair, **kw).events(seconds)
