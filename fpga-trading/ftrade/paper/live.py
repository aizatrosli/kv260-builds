"""Live PAPER trading on Luno PUBLIC market data.

Market data comes from ftrade.marketdata (public REST polling, no API key).
Orders go only to the local SimExchange. There is no code path here that
places real orders or touches authenticated endpoints.
"""
from __future__ import annotations
import sys
import time
from typing import Optional

from ..features import N_LEVELS, _s32, M32
from .backtest import BacktestConfig, Engine
from .book import L2Book
from .strategy import Strategy


class L2Features:
    """TopBook-equivalent features computed from the full L2 book.

    A deep REST book fed level-by-level into an 8-level TopBook can lose
    levels (when a top level is removed, level 9 is not known to it). This
    computes the same 8 features as if TopBook saw an exact top-8 stream.
    """

    def __init__(self, n: int = N_LEVELS) -> None:
        self.n = n
        self.flow = 0

    def __call__(self, e, book: L2Book) -> list[int]:
        if e.kind == "clear":
            self.flow = 0
        elif e.kind == "trade":
            s = 1 if e.side == "buy" else -1
            self.flow = _s32(self.flow + ((s * (e.volume & M32) - self.flow) >> 4))
        bids, asks = book.levels("bid", self.n), book.levels("ask", self.n)
        b1 = bids[0][0] if bids else 0
        a1 = asks[0][0] if asks else 0
        spread = (a1 - b1) if (bids and asks) else 0
        B = sum(v for _, v in bids[:5]) & M32
        A = sum(v for _, v in asks[:5]) & M32
        den = B + A
        if den == 0:
            imb = 0
        else:
            num = (B - A) << 15
            q = abs(num) // den
            imb = q if num >= 0 else -q
        bv1 = bids[0][1] if bids else 0
        av1 = asks[0][1] if asks else 0
        sp = spread & M32
        micro = 0 if (sp == 0 or bv1 + av1 == 0) else (sp * bv1) // (bv1 + av1)
        return [_s32(x) for x in (b1, a1, spread, B, A, imb, self.flow, micro)]


def _import_marketdata():
    try:
        import importlib
        # import_module returns the sys.modules entry, so a stubbed module is honoured
        # even when the real package was imported earlier and bound on `ftrade`.
        md = importlib.import_module("ftrade.marketdata")
        if md is None:
            raise ImportError("ftrade.marketdata is disabled")
    except Exception as e:
        raise RuntimeError(
            "live paper trading needs ftrade.marketdata (public Luno REST collector), "
            f"which could not be imported: {type(e).__name__}: {e}") from e
    for name in ("iter_events", "poll"):
        fn = getattr(md, name, None)
        if callable(fn):
            return fn
    rc = getattr(md, "RestCollector", None)
    if rc is not None:
        return lambda pair, seconds=None, **kw: rc(pair, **kw).events(seconds=seconds)
    raise RuntimeError("ftrade.marketdata has no iter_events/poll/RestCollector entry point")


def run_live(strategy: Strategy, pair: str = "XBTMYR", seconds: Optional[float] = None,
             cfg: Optional[BacktestConfig] = None, interval: float = 1.0,
             status_every_s: float = 10.0, md_kwargs: Optional[dict] = None) -> Engine:
    source = _import_marketdata()
    eng = Engine(strategy, cfg, feature_fn=L2Features())
    kw = {"interval": interval}
    kw.update(md_kwargs or {})
    try:
        events = source(pair, mode="rest", seconds=seconds, **kw)
    except TypeError:
        events = source(pair, seconds=seconds, **kw)
    last = time.monotonic()
    try:
        for e in events:
            eng.step(e)
            if time.monotonic() - last >= status_every_s:
                last = time.monotonic()
                mid = eng.last_mid or 0.0
                print(f"[paper-live] events={eng.n_events} mid={mid:.0f} "
                      f"net={eng.pf.net / 1e6:+.6f}BTC pnl={eng.pf.trading_pnl(mid) if mid else 0:+.2f}MYR "
                      f"fees={eng.pf.fees_paid:.2f} fills={len(eng.ex.fills)} "
                      f"risk_rej={len(eng.risk.rejections)} halted={eng.risk.halted}",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("[paper-live] stopped by user", file=sys.stderr)
    eng.finish()
    return eng
