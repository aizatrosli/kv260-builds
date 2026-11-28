"""Unit conversion and price-level book diffing shared by the REST and stream collectors.

Units follow SPEC.md: price = integer MYR, volume = integer 1e-6 BTC ("ubtc").
Decimal is used for parsing so "0.03678464" BTC converts exactly (round half even).
"""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable, Iterator, Optional

from ..events import Event

_UBTC = Decimal(1_000_000)
U32_MAX = 0xFFFFFFFF


def price_to_int(x) -> int:
    """Luno price string ("248001.00") -> integer MYR (XBTMYR tick is 1 MYR)."""
    return int(Decimal(str(x)).to_integral_value(rounding=ROUND_HALF_EVEN))


def vol_to_units(x) -> int:
    """BTC volume (string/Decimal) -> integer ubtc.

    A strictly positive volume never rounds to 0 (0 means "level removed" on the
    wire), it is clamped to 1 ubtc. Values are clamped to uint32.
    """
    d = Decimal(str(x))
    if d <= 0:
        return 0
    u = int((d * _UBTC).to_integral_value(rounding=ROUND_HALF_EVEN))
    return min(max(u, 1), U32_MAX)


def aggregate(levels: Iterable) -> dict[int, int]:
    """[(price, volume), ...] or [{"price":..,"volume":..}, ...] -> {price_int: ubtc}.

    Sums volumes per integer price in Decimal before converting, so order-level
    books (e.g. /api/1/orderbook, which lists every order) become price levels.
    """
    acc: dict[int, Decimal] = {}
    for lv in levels or ():
        if isinstance(lv, dict):
            p, v = lv["price"], lv["volume"]
        else:
            p, v = lv
        pi = price_to_int(p)
        acc[pi] = acc.get(pi, Decimal(0)) + Decimal(str(v))
    out = {}
    for p, v in acc.items():
        u = vol_to_units(v)
        if u > 0:
            out[p] = u
    return out


def _best_first(side: str, prices) -> list[int]:
    return sorted(prices, reverse=(side == "bid"))


def truncate(side: str, book: dict[int, int], depth: int) -> dict[int, int]:
    if not depth or len(book) <= depth:
        return dict(book)
    return {p: book[p] for p in _best_first(side, book)[:depth]}


def side_diff(side: str, old: dict[int, int], new: dict[int, int], ts: int,
              rx_ts: Optional[int]) -> list[Event]:
    """Level events turning `old` into `new`: removals first (volume 0), then
    inserts/changes best-first. Removing first means a finite-depth consumer
    (the N=8 TopBook) never drops a level that is still in the new top N."""
    out = [Event("level", ts, side, p, 0, rx_ts=rx_ts)
           for p in _best_first(side, [p for p in old if p not in new])]
    out += [Event("level", ts, side, p, new[p], rx_ts=rx_ts)
            for p in _best_first(side, new) if old.get(p) != new[p]]
    return out


def snapshot_events(bids: dict[int, int], asks: dict[int, int], ts: int,
                    rx_ts: Optional[int]) -> list[Event]:
    out = [Event("clear", ts, rx_ts=rx_ts)]
    out += [Event("level", ts, "bid", p, bids[p], rx_ts=rx_ts) for p in _best_first("bid", bids)]
    out += [Event("level", ts, "ask", p, asks[p], rx_ts=rx_ts) for p in _best_first("ask", asks)]
    return out


class BookDiffer:
    """Turns successive full price-level snapshots into normalized events.

    First call (or after reset()) emits `clear` + the full snapshot; afterwards
    only the levels that changed. depth > 0 keeps only the best `depth` levels
    per side, which makes a downstream depth-N engine exact when depth == N.
    """

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.synced = False

    def reset(self) -> None:
        self.synced = False

    def update(self, bids: dict[int, int], asks: dict[int, int], ts: int,
               rx_ts: Optional[int] = None, resync: bool = False) -> list[Event]:
        bids = truncate("bid", bids, self.depth)
        asks = truncate("ask", asks, self.depth)
        if resync or not self.synced:
            ev = snapshot_events(bids, asks, ts, rx_ts)
        else:
            ev = side_diff("bid", self.bids, bids, ts, rx_ts) + side_diff("ask", self.asks, asks, ts, rx_ts)
        self.bids, self.asks, self.synced = bids, asks, True
        return ev


def limit_depth(events: Iterable[Event], depth: int = 8) -> Iterator[Event]:
    """Re-express a recorded (deep) event stream as a depth-limited one.

    Replays the full book and emits only changes to the best `depth` levels per
    side, so ftrade.features.TopBook (N=8) sees an exact top-8 book even when
    levels deeper than 8 move into view. seq is renumbered from 1.
    """
    full = {"bid": {}, "ask": {}}
    view = {"bid": {}, "ask": {}}
    seq = 0

    def out(e: Event) -> Event:
        nonlocal seq
        seq += 1
        return Event(e.kind, e.ts, e.side, e.price, e.volume, seq, e.rx_ts)

    for e in events:
        if e.kind == "clear":
            full = {"bid": {}, "ask": {}}
            view = {"bid": {}, "ask": {}}
            yield out(e)
        elif e.kind == "trade":
            yield out(e)
        elif e.kind == "level":
            book = full[e.side]
            if e.volume:
                book[e.price] = e.volume
            else:
                book.pop(e.price, None)
            new = truncate(e.side, book, depth)
            for d in side_diff(e.side, view[e.side], new, e.ts, e.rx_ts):
                yield out(d)
            view[e.side] = new
        else:
            raise ValueError(e.kind)
