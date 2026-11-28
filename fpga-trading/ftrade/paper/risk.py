"""Pre-trade risk gate, modelled on the checks HFT firms put in hardware
(in the FPGA order path, before anything reaches the wire):

* max_position     - |net BTC position incl. working orders + this order| cap
* max_order_size   - per-order BTC cap
* max_notional     - per-order MYR notional cap
* rate_limit       - max new orders per rolling second (Luno-style API limit)
* price_collar     - limit price (or expected market-order average price) too
                     far from mid
* stale_data       - book older than max_data_age_ms, empty side or crossed
* kill switch      - trading PnL for the (Malaysia-time) day below
                     -max_daily_loss: cancel everything, flatten the position
                     with market orders, and halt all new risk until reset()

Every rejection is recorded in `self.rejections` (and logged) with its rule and
reason. Cancels are never rejected (refusing a cancel only adds risk), but they
count toward the rate limit window.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

from .book import L2Book
from .orders import OrderIntent, Portfolio, UBTC

log = logging.getLogger("ftrade.paper.risk")

DAY_MS = 86_400_000


@dataclass
class RiskConfig:
    max_position_btc: float = 0.02
    max_order_btc: float = 0.01
    max_notional_myr: float = 3_000.0
    max_orders_per_sec: int = 5
    max_daily_loss_myr: float = 250.0
    collar_bps: float = 50.0
    max_data_age_ms: int = 5_000
    day_offset_hours: float = 8.0          # Malaysia time (UTC+8) day boundary
    flatten_on_kill: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RiskGate:
    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        self.rejections: list[dict] = []
        self.halted = False
        self.halt_reason = ""
        self.kill_events: list[dict] = []
        self._recent: deque[int] = deque()   # ts of recent order actions
        self._day: Optional[int] = None
        self._day_start_pnl = 0.0
        self.n_checked = 0
        self.n_approved = 0

    # ---------------------------------------------------------------- utils
    def _reject(self, ts: int, rule: str, reason: str, intent: OrderIntent) -> None:
        r = {"ts": ts, "rule": rule, "reason": reason, "intent": intent.to_dict()}
        self.rejections.append(r)
        log.info("risk reject [%s] %s %s", rule, reason, intent.to_dict())

    def day_of(self, ts: int) -> int:
        return int((ts + self.cfg.day_offset_hours * 3_600_000) // DAY_MS)

    def reset(self) -> None:
        """Manual reset after a kill (a human decision in real life)."""
        self.halted = False
        self.halt_reason = ""

    def rejection_counts(self) -> dict:
        d: dict = {}
        for r in self.rejections:
            d[r["rule"]] = d.get(r["rule"], 0) + 1
        return d

    # ---------------------------------------------------------------- checks
    def check(self, intents: list[OrderIntent], now: int, book: L2Book,
              pf: Portfolio) -> list[OrderIntent]:
        """Return the subset of intents allowed through, in order."""
        out: list[OrderIntent] = []
        c = self.cfg
        while self._recent and self._recent[0] <= now - 1000:
            self._recent.popleft()
        # exposure of orders approved earlier in this same batch
        extra_buy = extra_sell = 0
        for it in intents:
            self.n_checked += 1
            if it.action in ("cancel", "cancel_all"):
                self._recent.append(now)
                out.append(it)
                self.n_approved += 1
                continue
            if self.halted:
                self._reject(now, "kill_switch", f"halted: {self.halt_reason}", it); continue
            # stale / broken data
            bb, ba = book.best_bid, book.best_ask
            if book.last_update_ts is None or bb is None or ba is None:
                self._reject(now, "stale_data", "no book (empty side)", it); continue
            age = now - book.last_update_ts
            if age > c.max_data_age_ms:
                self._reject(now, "stale_data", f"book age {age} ms > {c.max_data_age_ms}", it); continue
            if bb >= ba:
                self._reject(now, "stale_data", f"crossed book {bb}>={ba}", it); continue
            mid = (bb + ba) / 2.0
            # size
            if it.volume <= 0:
                self._reject(now, "max_order_size", "non-positive volume", it); continue
            if it.volume > c.max_order_btc * UBTC:
                self._reject(now, "max_order_size",
                             f"{it.volume / UBTC:.6f} BTC > {c.max_order_btc}", it); continue
            # reference price for notional + collar
            if it.type == "limit":
                ref = float(it.price)
            else:
                fills = book.walk(it.side, it.volume)
                got = sum(q for _, q in fills)
                if got < it.volume:
                    self._reject(now, "price_collar", "insufficient visible depth for market order", it); continue
                ref = sum(p * q for p, q in fills) / got
            notional = ref * it.volume / UBTC
            if notional > c.max_notional_myr:
                self._reject(now, "max_notional",
                             f"{notional:.2f} MYR > {c.max_notional_myr}", it); continue
            dev_bps = abs(ref - mid) / mid * 1e4
            if dev_bps > c.collar_bps:
                self._reject(now, "price_collar",
                             f"{dev_bps:.1f} bps from mid > {c.collar_bps}", it); continue
            # position (worst case: all working orders on this side fill)
            lim = c.max_position_btc * UBTC
            if it.side == "buy":
                worst = pf.net + pf.working_volume("buy") + extra_buy + it.volume
                if worst > lim:
                    self._reject(now, "max_position",
                                 f"worst-case long {worst / UBTC:.6f} BTC > {c.max_position_btc}", it); continue
            else:
                worst = pf.net - pf.working_volume("sell") - extra_sell - it.volume
                if worst < -lim:
                    self._reject(now, "max_position",
                                 f"worst-case short {worst / UBTC:.6f} BTC < -{c.max_position_btc}", it); continue
            # rate limit
            if len(self._recent) >= c.max_orders_per_sec:
                self._reject(now, "rate_limit",
                             f"{len(self._recent)} orders in last 1s >= {c.max_orders_per_sec}", it); continue
            self._recent.append(now)
            if it.side == "buy":
                extra_buy += it.volume
            else:
                extra_sell += it.volume
            out.append(it)
            self.n_approved += 1
        return out

    # ---------------------------------------------------------------- kill
    def on_tick(self, now: int, book: L2Book, pf: Portfolio,
                min_order: int = 500) -> list[OrderIntent]:
        """Call after every event. Returns kill-switch intents (bypass check())."""
        mid = book.mid()
        if mid is None:
            return []
        pnl = pf.trading_pnl(mid)
        d = self.day_of(now)
        if self._day != d:
            self._day = d
            self._day_start_pnl = pnl
        out: list[OrderIntent] = []
        day_pnl = pnl - self._day_start_pnl
        if not self.halted and day_pnl < -self.cfg.max_daily_loss_myr:
            self.halted = True
            self.halt_reason = f"daily loss {day_pnl:.2f} MYR < -{self.cfg.max_daily_loss_myr}"
            self.kill_events.append({"ts": now, "day_pnl": day_pnl, "net_ubtc": pf.net})
            log.warning("KILL SWITCH: %s; cancelling all and flattening", self.halt_reason)
            out.append(OrderIntent.cancel_all(tag="kill"))
        if self.halted and self.cfg.flatten_on_kill:
            # keep trying to flatten until done; only when nothing is in flight
            in_flight = [o for o in pf.working.values() if o.tag == "kill"]
            resting = [o for o in pf.working.values() if o.tag != "kill"]
            if not in_flight and (not resting or out):
                net = pf.net
                if abs(net) >= min_order:
                    side = "sell" if net > 0 else "buy"
                    out.append(OrderIntent.market(side, abs(net), tag="kill"))
        return out
