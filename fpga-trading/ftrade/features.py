"""Golden model of the hardware top-of-book + feature engine. See SPEC.md.

The RTL in ../rtl must match this bit-exactly, message by message.
"""
from __future__ import annotations
from .events import Event

N_LEVELS = 8
FEATURE_NAMES = ["best_bid", "best_ask", "spread", "bid_vol5", "ask_vol5",
                 "imb_q15", "flow", "micro_off"]
M32 = 0xFFFFFFFF


def _s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


class TopBook:
    def __init__(self, n_levels: int = N_LEVELS):
        self.n = n_levels
        self.bids: list[list[int]] = []   # [price, vol], descending price
        self.asks: list[list[int]] = []   # ascending price
        self.flow = 0

    def _side(self, side: str):
        return (self.bids, True) if side in ("bid", "buy") else (self.asks, False)

    def _update(self, side: str, price: int, vol: int) -> None:
        lv, desc = self._side(side)
        better = (lambda a, b: a > b) if desc else (lambda a, b: a < b)
        for i, (p, _) in enumerate(lv):
            if p == price:
                if vol == 0:
                    lv.pop(i)
                else:
                    lv[i][1] = vol
                return
        if vol == 0:
            return
        pos = len(lv)
        for i, (p, _) in enumerate(lv):
            if better(price, p):
                pos = i
                break
        if pos >= self.n:
            return
        lv.insert(pos, [price, vol])
        del lv[self.n:]

    def apply(self, e: Event) -> list[int]:
        if e.kind == "clear":
            self.bids.clear(); self.asks.clear(); self.flow = 0
        elif e.kind == "level":
            self._update(e.side, e.price & M32, e.volume & M32)
        elif e.kind == "trade":
            s = 1 if e.side == "buy" else -1
            x = s * (e.volume & M32)
            self.flow = _s32(self.flow + ((x - self.flow) >> 4))
        else:
            raise ValueError(e.kind)
        return self.features()

    def features(self) -> list[int]:
        b1 = self.bids[0][0] if self.bids else 0
        a1 = self.asks[0][0] if self.asks else 0
        spread = (a1 - b1) if (self.bids and self.asks) else 0
        B = sum(v for _, v in self.bids[:5]) & M32
        A = sum(v for _, v in self.asks[:5]) & M32
        den = B + A
        if den == 0:
            imb = 0
        else:
            num = (B - A) << 15
            q = abs(num) // den
            imb = q if num >= 0 else -q
        bv1 = self.bids[0][1] if self.bids else 0
        av1 = self.asks[0][1] if self.asks else 0
        sp_u = spread & M32
        micro = 0 if (sp_u == 0 or bv1 + av1 == 0) else (sp_u * bv1) // (bv1 + av1)
        return [_s32(b1), _s32(a1), _s32(spread), _s32(B), _s32(A), _s32(imb),
                _s32(self.flow), _s32(micro)]


def run(events) -> list[list[int]]:
    tb = TopBook()
    return [tb.apply(e) for e in events]
