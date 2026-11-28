"""Full-depth L2 order book built from `clear` / `level` events (SPEC.md).

Unlike `ftrade.features.TopBook` (the 8-level hardware golden model), this keeps
every level so the simulated exchange can walk real depth when filling orders.
Prices are integer MYR, volumes integer ubtc (1e-6 BTC).
"""
from __future__ import annotations
import bisect
from typing import Optional

from ..events import Event


class L2Book:
    def __init__(self) -> None:
        self.bids: dict[int, int] = {}      # price -> volume
        self.asks: dict[int, int] = {}
        self._bid_px: list[int] = []        # ascending; best bid = last
        self._ask_px: list[int] = []        # ascending; best ask = first
        self.last_update_ts: Optional[int] = None   # ts of last clear/level
        self.last_trade: Optional[Event] = None
        self.n_updates = 0

    # ---------------------------------------------------------------- updates
    def clear(self) -> None:
        self.bids.clear(); self.asks.clear()
        self._bid_px.clear(); self._ask_px.clear()

    def set_level(self, side: str, price: int, volume: int) -> None:
        if side in ("bid", "buy"):
            d, px = self.bids, self._bid_px
        elif side in ("ask", "sell"):
            d, px = self.asks, self._ask_px
        else:
            raise ValueError(f"bad side {side!r}")
        if volume < 0:
            raise ValueError("negative volume")
        if volume == 0:
            if price in d:
                del d[price]
                i = bisect.bisect_left(px, price)
                px.pop(i)
            return
        if price not in d:
            bisect.insort(px, price)
        d[price] = volume

    def apply(self, e: Event) -> None:
        if e.kind == "clear":
            self.clear()
            self.last_update_ts = e.ts
        elif e.kind == "level":
            self.set_level(e.side, e.price, e.volume)
            self.last_update_ts = e.ts
        elif e.kind == "trade":
            self.last_trade = e
        else:
            raise ValueError(e.kind)
        self.n_updates += 1

    # ---------------------------------------------------------------- queries
    @property
    def best_bid(self) -> Optional[int]:
        return self._bid_px[-1] if self._bid_px else None

    @property
    def best_ask(self) -> Optional[int]:
        return self._ask_px[0] if self._ask_px else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (a + b) / 2.0

    def spread(self) -> Optional[int]:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else a - b

    def is_crossed(self) -> bool:
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b >= a

    def volume_at(self, side: str, price: int) -> int:
        return (self.bids if side in ("bid", "buy") else self.asks).get(price, 0)

    def levels(self, side: str, n: Optional[int] = None) -> list[tuple[int, int]]:
        """Best-first [(price, volume)] for 'bid' or 'ask'."""
        if side in ("bid", "buy"):
            px = self._bid_px[::-1] if n is None else self._bid_px[:-n - 1:-1]
            return [(p, self.bids[p]) for p in px]
        px = self._ask_px if n is None else self._ask_px[:n]
        return [(p, self.asks[p]) for p in px]

    def depth(self, side: str) -> int:
        return len(self.bids if side in ("bid", "buy") else self.asks)

    def copy(self) -> "L2Book":
        b = L2Book()
        b.bids = dict(self.bids); b.asks = dict(self.asks)
        b._bid_px = list(self._bid_px); b._ask_px = list(self._ask_px)
        b.last_update_ts = self.last_update_ts; b.last_trade = self.last_trade
        return b

    def walk(self, taker_side: str, volume: int, limit_price: Optional[int] = None,
             consumed: Optional[dict] = None) -> list[tuple[int, int]]:
        """Simulate a taker sweeping the opposite side. Returns [(price, vol)].

        taker_side 'buy' walks asks upward, 'sell' walks bids downward.
        `limit_price` stops the walk at that price. `consumed` maps
        (side, price) -> volume we already took (not yet refreshed by the feed),
        so repeated orders cannot re-use the same displayed liquidity.
        Does not modify the book.
        """
        side = "ask" if taker_side == "buy" else "bid"
        fills: list[tuple[int, int]] = []
        left = volume
        for p, v in self.levels(side):
            if left <= 0:
                break
            if limit_price is not None:
                if taker_side == "buy" and p > limit_price:
                    break
                if taker_side == "sell" and p < limit_price:
                    break
            if consumed:
                v -= consumed.get((side, p), 0)
            if v <= 0:
                continue
            q = min(v, left)
            fills.append((p, q))
            left -= q
        return fills

    def __repr__(self) -> str:
        return (f"L2Book(bid={self.best_bid} x{self.depth('bid')}, "
                f"ask={self.best_ask} x{self.depth('ask')})")
