"""Optional Luno streaming (websocket) market-data client.

wss://ws.luno.com/api/1/stream/<PAIR>. Luno requires API key credentials as the
first client message even for the market stream, so this only runs when the
user supplies a key via LUNO_API_KEY_ID / LUNO_API_KEY_SECRET. Use a key with
read-only ("View"/MP_None) permissions: this module never trades, and the key is
only ever sent to ws.luno.com.

Protocol (Luno docs, "Streaming API"):
  1. client sends {"api_key_id": ..., "api_key_secret": ...}
  2. server sends the order-level book {"sequence","asks":[{id,price,volume}],
     "bids":[...],"status","timestamp"}
  3. server sends updates {"sequence","trade_updates":[...]|null,
     "create_update":{order_id,type:BID|ASK,price,volume}|null,
     "delete_update":{order_id}|null,"status_update":{status}|null,"timestamp"}
     Message n applies to state n-1; any gap -> reconnect and rebuild.
     An empty message is a keep-alive.

StreamBook holds the order-level book and aggregates it to price-level
`level` events (and `trade` events); it is pure and unit-tested offline.
The websocket part needs the optional `websockets` package and is imported lazily.
"""
from __future__ import annotations
import json
import os
import queue
import sys
import threading
import time
from decimal import Decimal
from typing import Iterator, Optional

from ..events import Event
from .book import price_to_int, snapshot_events, vol_to_units
from .transport import Backoff, now_ms

WS_URL = "wss://ws.luno.com/api/1/stream/{pair}"
ENV_KEY_ID, ENV_KEY_SECRET = "LUNO_API_KEY_ID", "LUNO_API_KEY_SECRET"
_SIDE = {"BID": "bid", "ASK": "ask"}


class SequenceGap(Exception):
    pass


def _ts(msg: dict, rx_ts: int) -> int:
    t = msg.get("timestamp")
    try:
        t = int(t)
    except (TypeError, ValueError):
        return rx_ts
    return t if t > 10 ** 11 else t * 1000     # docs show both ms and (example) seconds


class StreamBook:
    """Order-level Luno book -> price-level normalized events."""

    def __init__(self):
        self.orders: dict[str, tuple[str, int, Decimal]] = {}   # id -> (side, price, vol)
        self.levels = {"bid": {}, "ask": {}}                     # price -> Decimal
        self.emitted = {"bid": {}, "ask": {}}                    # price -> last ubtc sent
        self.seq: Optional[int] = None
        self.status = ""
        self.unknown_makers = 0

    # internal book maintenance
    def _add(self, oid: str, side: str, price: int, vol: Decimal, touched: set) -> None:
        if vol <= 0:
            return
        self.orders[oid] = (side, price, vol)
        lv = self.levels[side]
        lv[price] = lv.get(price, Decimal(0)) + vol
        touched.add((side, price))

    def _reduce(self, oid: str, by: Optional[Decimal], touched: set):
        """Remove `by` volume (None = all) from order oid; returns its side or None."""
        o = self.orders.get(oid)
        if o is None:
            return None
        side, price, vol = o
        dv = vol if by is None or by >= vol else by
        rest = vol - dv
        if rest > 0:
            self.orders[oid] = (side, price, rest)
        else:
            del self.orders[oid]
        lv = self.levels[side]
        nv = lv.get(price, Decimal(0)) - dv
        if nv > 0:
            lv[price] = nv
        else:
            lv.pop(price, None)
        touched.add((side, price))
        return side

    def _level_events(self, touched: set, ts: int, rx_ts: int) -> list[Event]:
        out = []
        # removals first, then changes best-first (same convention as BookDiffer)
        items = sorted(touched, key=lambda sp: (sp[0], -sp[1] if sp[0] == "bid" else sp[1]))
        rem, upd = [], []
        for side, price in items:
            u = vol_to_units(self.levels[side].get(price, 0))
            if self.emitted[side].get(price, 0) == u:
                continue
            if u:
                self.emitted[side][price] = u
                upd.append(Event("level", ts, side, price, u, rx_ts=rx_ts))
            else:
                self.emitted[side].pop(price, None)
                rem.append(Event("level", ts, side, price, 0, rx_ts=rx_ts))
        return rem + upd

    # public API
    def on_snapshot(self, msg: dict, rx_ts: Optional[int] = None) -> list[Event]:
        rx_ts = now_ms() if rx_ts is None else rx_ts
        self.orders.clear()
        self.levels = {"bid": {}, "ask": {}}
        touched: set = set()
        for key, side in (("bids", "bid"), ("asks", "ask")):
            for o in msg.get(key) or []:
                self._add(o["id"], side, price_to_int(o["price"]), Decimal(str(o["volume"])), touched)
        self.seq = int(msg["sequence"])
        self.status = msg.get("status", "")
        agg = {s: {p: vol_to_units(v) for p, v in self.levels[s].items()} for s in ("bid", "ask")}
        agg = {s: {p: u for p, u in d.items() if u} for s, d in agg.items()}
        self.emitted = {s: dict(d) for s, d in agg.items()}
        return snapshot_events(agg["bid"], agg["ask"], _ts(msg, rx_ts), rx_ts)

    def on_update(self, msg, rx_ts: Optional[int] = None) -> list[Event]:
        """Apply one update message atomically. Raises SequenceGap if out of order."""
        rx_ts = now_ms() if rx_ts is None else rx_ts
        if not msg:                       # "" / {} keep-alive
            return []
        if self.seq is None:
            raise SequenceGap("update before snapshot")
        seq = int(msg["sequence"])
        if seq != self.seq + 1:
            raise SequenceGap(f"expected {self.seq + 1}, got {seq}")
        self.seq = seq
        ts = _ts(msg, rx_ts)
        touched: set = set()
        trades: list[Event] = []
        for tu in msg.get("trade_updates") or []:
            base = Decimal(str(tu["base"]))
            counter = Decimal(str(tu["counter"]))
            maker = tu.get("maker_order_id") or tu.get("order_id")
            mside = self._reduce(maker, base, touched) if maker else None
            if mside is None:
                self.unknown_makers += 1
            # taker is on the opposite side of the resting (maker) order
            tside = "sell" if mside == "bid" else "buy" if mside == "ask" else ""
            price = price_to_int(counter / base) if base else 0
            if not tside:   # unknown maker: infer from price vs best ask
                asks = self.levels["ask"]
                tside = "buy" if asks and price >= min(asks) else "sell"
            trades.append(Event("trade", ts, tside, price, vol_to_units(base), rx_ts=rx_ts))
        cu = msg.get("create_update")
        if cu:
            self._add(cu["order_id"], _SIDE[cu["type"]], price_to_int(cu["price"]),
                      Decimal(str(cu["volume"])), touched)
        du = msg.get("delete_update")
        if du:
            self._reduce(du["order_id"], None, touched)
        su = msg.get("status_update")
        if su:
            self.status = su.get("status", self.status)
        return trades + self._level_events(touched, ts, rx_ts)


# ---------------------------------------------------------------- websocket
def credentials() -> Optional[tuple[str, str]]:
    k, s = os.environ.get(ENV_KEY_ID), os.environ.get(ENV_KEY_SECRET)
    return (k, s) if k and s else None


def available() -> tuple[bool, str]:
    """(True, "") if the stream can run, else (False, reason)."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False, "the 'websockets' package is not installed (pip install websockets)"
    if credentials() is None:
        return False, f"set {ENV_KEY_ID} and {ENV_KEY_SECRET} (read-only Luno API key)"
    return True, ""


class StreamCollector:
    """Yields normalized events from the Luno websocket, reconnecting on gaps.

    Each (re)connect starts with `clear` + full snapshot. seq is per collector.
    """

    def __init__(self, pair: str = "XBTMYR", key_id: Optional[str] = None,
                 key_secret: Optional[str] = None, keepalive_s: float = 20.0,
                 idle_timeout_s: float = 60.0, verbose: bool = False):
        creds = (key_id, key_secret) if key_id and key_secret else credentials()
        if creds is None:
            raise RuntimeError(available()[1])
        self.pair, self._creds = pair, creds
        self.keepalive_s, self.idle_timeout_s = keepalive_s, idle_timeout_s
        self.verbose = verbose
        self.reconnects = 0
        self.seq = 0
        self.counts = {"clear": 0, "level": 0, "trade": 0}

    def _say(self, m):
        if self.verbose:
            print(f"[stream] {m}", file=sys.stderr, flush=True)

    async def _run(self, q: "queue.Queue", stop: threading.Event):
        import asyncio
        import websockets
        backoff = Backoff(base=1.0, cap=60.0)
        url = WS_URL.format(pair=self.pair)
        while not stop.is_set():
            book = StreamBook()
            try:
                async with websockets.connect(url, open_timeout=15, ping_interval=20,
                                              max_size=2 ** 24) as ws:
                    await ws.send(json.dumps({"api_key_id": self._creds[0],
                                              "api_key_secret": self._creds[1]}))
                    first = await asyncio.wait_for(ws.recv(), timeout=30)
                    snap = json.loads(first)
                    if "sequence" not in snap:
                        raise RuntimeError(f"unexpected first message: {str(first)[:200]}")
                    q.put(book.on_snapshot(snap, now_ms()))
                    backoff.reset()
                    last_send = time.monotonic()
                    while not stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=self.keepalive_s)
                        except asyncio.TimeoutError:
                            raw = None
                        if time.monotonic() - last_send >= self.keepalive_s:
                            await ws.send("")          # client keep-alive
                            last_send = time.monotonic()
                        if raw is None:
                            continue
                        rx = now_ms()
                        msg = json.loads(raw) if raw and raw.strip() else None
                        evs = book.on_update(msg, rx)
                        if evs:
                            q.put(evs)
            except Exception as e:                      # gap, network, auth, ...
                if stop.is_set():
                    break
                self.reconnects += 1
                d = backoff.next()
                self._say(f"{type(e).__name__}: {e} -> reconnect in {d:.1f}s")
                await asyncio.sleep(d)

    def events(self, seconds: Optional[float] = None) -> Iterator[Event]:
        import asyncio
        ok, why = available()
        if not ok:
            raise RuntimeError(why)
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        th = threading.Thread(target=lambda: asyncio.run(self._run(q, stop)), daemon=True)
        th.start()
        t_end = None if seconds is None else time.monotonic() + seconds
        try:
            while t_end is None or time.monotonic() < t_end:
                try:
                    evs = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                for e in evs:
                    self.seq += 1
                    e.seq = self.seq
                    self.counts[e.kind] += 1
                    yield e
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            th.join(timeout=5)
