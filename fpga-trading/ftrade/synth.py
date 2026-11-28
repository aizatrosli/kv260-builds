"""Synthetic XBTMYR-like market for offline tests and backtests.

A latent "fair value" random walk; market makers quote around it with a
weak lead from order-book imbalance and taker flow, so a model has something
(small) to learn. Not a realistic market.
"""
from __future__ import annotations
import random
from .events import Event


def generate(n_events: int = 20000, seed: int = 0, start_price: int = 250_000,
             depth: int = 10, tick: int = 1, t0_ms: int = 1_790_000_000_000) -> list[Event]:
    rng = random.Random(seed)
    fair = float(start_price)
    pressure = 0.0               # hidden drift; shows up in imbalance and flow first
    ts = t0_ms
    out: list[Event] = []
    seq = 0

    def emit(**kw):
        nonlocal seq
        seq += 1
        out.append(Event(ts=ts, seq=seq, **kw))

    def book_levels():
        mid = round(fair)
        half = max(1, int(rng.choice([1, 1, 2, 3, 5]) ))
        bb, ba = mid - half, mid + half
        skew = pressure
        bids, asks = {}, {}
        for i in range(depth):
            bids[bb - i * tick * rng.randint(1, 4)] = max(1, int(rng.expovariate(1 / 20000) * (1 + max(skew, 0) * 2)))
            asks[ba + i * tick * rng.randint(1, 4)] = max(1, int(rng.expovariate(1 / 20000) * (1 + max(-skew, 0) * 2)))
        return bids, asks

    bids, asks = book_levels()
    emit(kind="clear")
    for p, v in sorted(bids.items(), reverse=True):
        emit(kind="level", side="bid", price=p, volume=v)
    for p, v in sorted(asks.items()):
        emit(kind="level", side="ask", price=p, volume=v)

    while len(out) < n_events:
        ts += rng.randint(1, 400)
        pressure = 0.97 * pressure + rng.gauss(0, 0.15)
        fair += rng.gauss(0, 8) + pressure * 6
        r = rng.random()
        if r < 0.15:
            side = "buy" if rng.random() < 0.5 + 0.3 * max(-1, min(1, pressure)) else "sell"
            ref = min(asks) if side == "buy" else max(bids)
            emit(kind="trade", side=side, price=ref, volume=max(1, int(rng.expovariate(1 / 5000))))
        elif r < 0.25:
            nb, na = book_levels()
            for p in [p for p in bids if p not in nb]:
                emit(kind="level", side="bid", price=p, volume=0)
            for p in [p for p in asks if p not in na]:
                emit(kind="level", side="ask", price=p, volume=0)
            for p, v in nb.items():
                if bids.get(p) != v:
                    emit(kind="level", side="bid", price=p, volume=v)
            for p, v in na.items():
                if asks.get(p) != v:
                    emit(kind="level", side="ask", price=p, volume=v)
            bids, asks = nb, na
        else:
            book, side = (bids, "bid") if rng.random() < 0.5 else (asks, "ask")
            p = rng.choice(list(book))
            v = max(1, int(book[p] * rng.uniform(0.5, 1.5)))
            book[p] = v
            emit(kind="level", side=side, price=p, volume=v)
    return out[:n_events]
