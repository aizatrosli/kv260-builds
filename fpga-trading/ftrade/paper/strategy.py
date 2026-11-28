"""Strategy interface and example strategies (paper trading only).

    class Strategy:
        def on_event(self, event, features, book, portfolio) -> list[OrderIntent]

* `event`     - ftrade.events.Event just applied
* `features`  - the 8 int32 TopBook features for that event (what the FPGA emits)
* `book`      - full-depth L2Book (the exchange's view, for reference prices)
* `portfolio` - Portfolio (balances, `net` position, working orders)

Strategies return OrderIntents; the risk gate decides what is actually sent.
"""
from __future__ import annotations
from collections import deque
from typing import Callable, Optional

import numpy as np

from ..events import Event
from ..features import FEATURE_NAMES
from .book import L2Book
from .orders import OrderIntent, Portfolio

F = {n: i for i, n in enumerate(FEATURE_NAMES)}


class Strategy:
    name = "base"

    def on_event(self, event: Event, features: list[int], book: L2Book,
                 portfolio: Portfolio) -> list[OrderIntent]:
        return []

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))}


class _TargetPositionMixin:
    """Move the net position toward a target with market orders."""
    size: int
    min_order: int
    cooldown_ms: int
    _last_order_ts: int

    def _go_to(self, target: int, now: int, pf: Portfolio, tag: str) -> list[OrderIntent]:
        if pf.has_working() or now - self._last_order_ts < self.cooldown_ms:
            return []
        delta = target - pf.net
        if abs(delta) < self.min_order:
            return []
        self._last_order_ts = now
        return [OrderIntent.market("buy" if delta > 0 else "sell", abs(delta), tag=tag)]


class MomentumTaker(_TargetPositionMixin, Strategy):
    """(a) Order-book imbalance + trade-flow momentum, taking liquidity.

    Goes long `size` when imb_q15 > imb_th AND flow > flow_th, short when both
    are below the negative thresholds, and goes flat after `hold_ms` without a
    fresh signal. Uses only TopBook features, so the same logic could live in
    the FPGA fabric.
    """
    name = "momentum"

    def __init__(self, size: int = 2_000, imb_th: int = 12_000, flow_th: int = 1_500,
                 hold_ms: int = 10_000, cooldown_ms: int = 2_000,
                 allow_short: bool = True, min_order: int = 500) -> None:
        self.size, self.imb_th, self.flow_th = size, imb_th, flow_th
        self.hold_ms, self.cooldown_ms = hold_ms, cooldown_ms
        self.allow_short, self.min_order = allow_short, min_order
        self._target = 0
        self._signal_ts = -10**18
        self._last_order_ts = -10**18

    def on_event(self, event, features, book, portfolio):
        if features[F["best_bid"]] == 0 or features[F["best_ask"]] == 0:
            return []
        imb, flow = features[F["imb_q15"]], features[F["flow"]]
        now = event.ts
        sig = 0
        if imb > self.imb_th and flow > self.flow_th:
            sig = 1
        elif imb < -self.imb_th and flow < -self.flow_th:
            sig = -1 if self.allow_short else 0
        if sig != 0:
            self._target = sig * self.size
            self._signal_ts = now
        elif now - self._signal_ts > self.hold_ms:
            self._target = 0
        return self._go_to(self._target, now, portfolio, "mom")


class MarketMaker(Strategy):
    """(b) Symmetric post-only market maker with inventory skew.

    Quotes bid/ask at mid -/+ half_spread_bps, both shifted against inventory
    by skew_bps * (net / max_inventory). Requotes (cancel + new) when the
    desired price moves by >= requote_ticks. Stops quoting the side that would
    grow inventory past max_inventory.
    """
    name = "mm"

    def __init__(self, size: int = 1_000, half_spread_bps: float = 2.0,
                 skew_bps: float = 20.0, max_inventory: int = 10_000,
                 requote_ticks: int = 25, min_requote_ms: int = 1_000) -> None:
        self.size, self.half_spread_bps, self.skew_bps = size, half_spread_bps, skew_bps
        self.max_inventory, self.requote_ticks = max_inventory, requote_ticks
        self.min_requote_ms = min_requote_ms
        self._last_ts = {"buy": -10**18, "sell": -10**18}

    def desired(self, mid: float, net: int) -> tuple[int, int]:
        inv = max(-1.0, min(1.0, net / self.max_inventory)) if self.max_inventory else 0.0
        centre = mid * (1 - self.skew_bps * inv / 1e4)
        half = mid * self.half_spread_bps / 1e4
        return int(np.floor(centre - half)), int(np.ceil(centre + half))

    def on_event(self, event, features, book, portfolio):
        bb, ba = features[F["best_bid"]], features[F["best_ask"]]
        if bb == 0 or ba == 0 or bb >= ba:
            return []
        mid = (bb + ba) / 2.0
        want_bid, want_ask = self.desired(mid, portfolio.net)
        # never cross: post-only would be rejected anyway
        want_bid, want_ask = min(want_bid, ba - 1), max(want_ask, bb + 1)
        now = event.ts
        out: list[OrderIntent] = []
        for side, want, allowed in (("buy", want_bid, portfolio.net < self.max_inventory),
                                    ("sell", want_ask, portfolio.net > -self.max_inventory)):
            mine = [o for o in portfolio.working_orders(side) if o.tag == "mm"]
            live = [o for o in mine if not o.cancel_requested]
            if now - self._last_ts[side] < self.min_requote_ms:
                continue
            if not allowed:
                for o in live:
                    out.append(OrderIntent.cancel(o.id, tag="mm"))
                if live:
                    self._last_ts[side] = now
                continue
            if live:
                o = live[0]
                if abs(o.price - want) < self.requote_ticks and len(live) == 1:
                    continue
                for o in live:
                    out.append(OrderIntent.cancel(o.id, tag="mm"))
            elif mine:
                continue        # cancels still in flight; wait
            out.append(OrderIntent.limit(side, want, self.size, post_only=True, tag="mm"))
            self._last_ts[side] = now
        return out


def load_predictor(model_path: Optional[str] = None, backend: Optional[str] = None):
    """Lazily import ftrade.predictor.infer.Predictor. Optional dependency."""
    try:
        from ..predictor.infer import Predictor  # type: ignore
    except Exception as e:  # ImportError or anything broken inside it
        raise RuntimeError(
            "PredictorStrategy needs ftrade.predictor.infer.Predictor, which could not be "
            f"imported ({type(e).__name__}: {e}). Build/train the predictor first, or pass "
            "your own predict callable to PredictorStrategy(predict=...).") from e
    kw = {"backend": backend} if backend else {}
    if model_path:
        if hasattr(Predictor, "load"):
            return Predictor.load(model_path, **kw)
        return Predictor(model_path, **kw)
    return Predictor(**kw)


class PredictorStrategy(_TargetPositionMixin, Strategy):
    """(c) Adapter around any `predict(window) -> {'signal': -1|0|1, ...}`.

    `predict` may be a callable or an object with a `.predict` method (e.g.
    ftrade.predictor.infer.Predictor). The window is an int64 numpy array of
    shape (W, 8), oldest first. A signal sets the target position to
    signal * size; the target decays to flat after hold_ms with no signal.
    """
    name = "predictor"

    def __init__(self, predict=None, window: Optional[int] = None, size: int = 2_000,
                 every_n: int = 1, hold_ms: int = 10_000, cooldown_ms: int = 2_000,
                 allow_short: bool = True, min_order: int = 500,
                 model_path: Optional[str] = None, backend: Optional[str] = None) -> None:
        if predict is None:
            predict = load_predictor(model_path, backend)
        model = predict
        self._predict: Callable = predict.predict if hasattr(predict, "predict") else predict
        if window is None:
            window = int(getattr(model, "window_size", getattr(model, "window", 16)) or 16)
        self.window, self.size, self.every_n = window, size, every_n
        self.hold_ms, self.cooldown_ms = hold_ms, cooldown_ms
        self.allow_short, self.min_order = allow_short, min_order
        self._buf: deque = deque(maxlen=window)
        self._n = 0
        self._target = 0
        self._signal_ts = -10**18
        self._last_order_ts = -10**18
        self.n_predictions = 0
        self.signal_counts = {-1: 0, 0: 0, 1: 0}

    def on_event(self, event, features, book, portfolio):
        self._buf.append(features)
        self._n += 1
        now = event.ts
        if len(self._buf) == self.window and self._n % self.every_n == 0:
            res = self._predict(np.asarray(self._buf, dtype=np.int64))
            if isinstance(res, dict):
                sig = int(res.get("signal", 0)) if res.get("valid", True) else 0
            else:
                sig = int(res)
            sig = max(-1, min(1, sig))
            if sig < 0 and not self.allow_short:
                sig = 0
            self.n_predictions += 1
            self.signal_counts[sig] += 1
            if sig != 0:
                self._target = sig * self.size
                self._signal_ts = now
        if now - self._signal_ts > self.hold_ms:
            self._target = 0
        return self._go_to(self._target, now, portfolio, "pred")


class FlowOracle:
    """Tiny built-in `predict` callable (no ML): sign of imbalance+flow over
    the window. Useful to test PredictorStrategy without the real predictor."""
    window_size = 16

    def __init__(self, imb_th: int = 8_000, flow_th: int = 800) -> None:
        self.imb_th, self.flow_th = imb_th, flow_th

    def predict(self, w):
        imb = float(np.mean(w[:, F["imb_q15"]]))
        flow = float(np.mean(w[:, F["flow"]]))
        s = 1 if (imb > self.imb_th and flow > self.flow_th) else \
            -1 if (imb < -self.imb_th and flow < -self.flow_th) else 0
        return {"signal": s, "valid": bool(w[-1, 0] and w[-1, 1])}


STRATEGIES = {"momentum": MomentumTaker, "mm": MarketMaker, "predictor": PredictorStrategy}
