"""Simulated (paper) Luno-like spot exchange.

PAPER TRADING ONLY. This module never talks to the network; every "order" is a
Python object matched against a locally replayed / polled public order book.

Model
-----
* Orders travel to the "exchange" with a simulated network latency
  (uniform in [latency_ms - jitter_ms, latency_ms + jitter_ms]), FIFO per
  connection, so a decision taken on the book at time t executes against the
  book at t + latency (stale book, like on the real internet).
* Market orders (IOC) walk the full-depth L2 book. Liquidity we take is
  remembered per level until the feed refreshes that level, so two orders
  cannot both eat the same displayed volume.
* Limit orders: the marketable part (if any) executes as taker up to the limit
  price, the rest rests. `post_only` orders that would cross are rejected
  (Luno's post-only behaviour).
* Resting limit fills come only from printed trades (`trade` events), using a
  queue-position model:
    - 'conservative' (default): queue_ahead = displayed volume at our price
      when we arrive. A trade strictly through our price fills us (up to the
      trade's size); a trade at our price first consumes queue_ahead and only
      the excess fills us. Level size decreases shrink queue_ahead.
    - 'through_only': only trades strictly through our price fill us.
    - 'optimistic': we are first in the queue (queue_ahead = 0).
* Fees in bps of notional, taker vs maker, charged in MYR. Minimum order size
  and price tick enforced. Spot balances enforced (no shorting / borrowing).
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass, asdict
from typing import Optional

from ..events import Event
from .book import L2Book
from .orders import Fill, Order, OrderIntent, Portfolio, UBTC

log = logging.getLogger("ftrade.paper.exchange")


@dataclass
class ExchangeConfig:
    # Luno publishes tiered fees; public reviews of Luno Malaysia quote roughly
    # 0.25%-0.35% maker and 0.25%-0.50% taker for retail 30-day volume, and
    # Luno's own help centre gives maker -0.02%..0.40% depending on tier/region.
    # Defaults are deliberately the conservative end. CHECK your own fee page.
    taker_fee_bps: float = 50.0
    maker_fee_bps: float = 25.0
    min_order_ubtc: int = 500         # 0.0005 BTC, Luno-like XBT minimum
    max_order_ubtc: int = 100 * UBTC
    price_tick: int = 1               # MYR
    latency_ms: float = 100.0         # one-way decision -> exchange
    latency_jitter_ms: float = 50.0   # uniform +/- jitter (default 50..150 ms)
    queue_model: str = "conservative" # conservative | through_only | optimistic
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


QUEUE_MODELS = ("conservative", "through_only", "optimistic")


class SimExchange:
    def __init__(self, config: Optional[ExchangeConfig] = None,
                 portfolio: Optional[Portfolio] = None,
                 book: Optional[L2Book] = None) -> None:
        self.cfg = config or ExchangeConfig()
        if self.cfg.queue_model not in QUEUE_MODELS:
            raise ValueError(f"queue_model must be one of {QUEUE_MODELS}")
        self.pf = portfolio or Portfolio()
        self.book = book or L2Book()
        self.rng = random.Random(self.cfg.seed)
        self.now = 0
        self._next_id = 1
        self._last_arrival = -1.0
        self._pending: list[tuple[float, int, str, object]] = []   # (arrival, seq, kind, payload)
        self._seq = 0
        self.consumed: dict[tuple[str, int], int] = {}
        self.orders: dict[int, Order] = {}
        self.fills: list[Fill] = []
        self.rejects: list[dict] = []
        self.latencies_ms: list[float] = []
        self.n_submitted = 0
        self.submitted_volume = 0

    # ------------------------------------------------------------ helpers
    def _fee(self, notional: float, liquidity: str) -> float:
        bps = self.cfg.taker_fee_bps if liquidity == "taker" else self.cfg.maker_fee_bps
        return notional * bps / 10_000.0

    def sample_latency(self) -> float:
        c, j = self.cfg.latency_ms, self.cfg.latency_jitter_ms
        lat = c if j <= 0 else self.rng.uniform(c - j, c + j)
        return max(0.0, lat)

    def _schedule(self, decision_ts: int, kind: str, payload) -> float:
        lat = self.sample_latency()
        arrival = max(decision_ts + lat, self._last_arrival)   # FIFO connection
        self._last_arrival = arrival
        self.latencies_ms.append(arrival - decision_ts)
        self._seq += 1
        self._pending.append((arrival, self._seq, kind, payload))
        return arrival

    def _reject(self, o: Order, reason: str) -> None:
        o.status = "rejected"
        o.reject_reason = reason
        self.pf.working.pop(o.id, None)
        self.rejects.append({"ts": self.now, "order_id": o.id, "reason": reason,
                             "side": o.side, "type": o.type, "price": o.price,
                             "volume": o.volume, "tag": o.tag})
        log.info("exchange reject %s: %s", o.id, reason)

    def available_cash(self) -> float:
        res = 0.0
        for o in self.pf.working.values():
            if o.status == "resting" and o.side == "buy":
                res += o.price * o.remaining / UBTC * (1 + self.cfg.maker_fee_bps / 1e4)
        return self.pf.cash - res

    def available_btc(self) -> int:
        res = sum(o.remaining for o in self.pf.working.values()
                  if o.status == "resting" and o.side == "sell")
        return self.pf.btc - res

    # ------------------------------------------------------------ API
    def submit(self, intent: OrderIntent, decision_ts: int) -> Optional[Order]:
        """Send an intent; it reaches the matching engine after latency."""
        if intent.action == "new":
            o = Order(id=self._next_id, side=intent.side, type=intent.type,
                      price=int(intent.price), volume=int(intent.volume),
                      post_only=intent.post_only, tag=intent.tag,
                      decision_ts=decision_ts, arrival_ts=0)
            self._next_id += 1
            self.orders[o.id] = o
            self.pf.working[o.id] = o
            self.n_submitted += 1
            self.submitted_volume += o.volume
            o.arrival_ts = int(self._schedule(decision_ts, "new", o))
            return o
        if intent.action == "cancel":
            o = self.orders.get(intent.order_id)
            if o is not None:
                o.cancel_requested = True
            self._schedule(decision_ts, "cancel", intent.order_id)
            return o
        if intent.action == "cancel_all":
            for o in self.pf.working.values():
                o.cancel_requested = True
            self._schedule(decision_ts, "cancel_all", None)
            return None
        raise ValueError(f"unknown action {intent.action!r}")

    def process_until(self, ts: float) -> None:
        """Execute everything that has arrived at the exchange by `ts`."""
        if not self._pending:
            return
        self._pending.sort()
        while self._pending and self._pending[0][0] <= ts:
            arrival, _, kind, payload = self._pending.pop(0)
            self.now = int(arrival)
            if kind == "new":
                self._on_new(payload)
            elif kind == "cancel":
                self._on_cancel(payload)
            else:
                # cancel_all hits what is on the book now; orders sent after
                # it are still in flight (FIFO) and are not affected
                for oid, o in list(self.pf.working.items()):
                    if o.status == "resting":
                        self._on_cancel(oid)

    def on_event(self, e: Event) -> list[Fill]:
        """Advance simulated time to the event, then apply it to the book."""
        n0 = len(self.fills)
        self.process_until(e.ts)
        self.now = e.ts
        self.book.apply(e)
        if e.kind == "clear":
            self.consumed.clear()
        elif e.kind == "level":
            side = "bid" if e.side in ("bid", "buy") else "ask"
            self.consumed.pop((side, e.price), None)
            our_side = "buy" if side == "bid" else "sell"
            for o in self.pf.working.values():
                if o.status == "resting" and o.side == our_side and o.price == e.price:
                    o.queue_ahead = min(o.queue_ahead, e.volume)
        elif e.kind == "trade":
            self._match_trade(e)
        return self.fills[n0:]

    def flush(self) -> None:
        self.process_until(float("inf"))

    # ------------------------------------------------------------ matching
    def _validate(self, o: Order) -> Optional[str]:
        if o.side not in ("buy", "sell"):
            return "bad_side"
        if o.type not in ("market", "limit"):
            return "bad_type"
        if o.volume < self.cfg.min_order_ubtc:
            return f"below_min_size({o.volume}<{self.cfg.min_order_ubtc})"
        if o.volume > self.cfg.max_order_ubtc:
            return "above_max_size"
        if o.type == "limit":
            if o.price <= 0:
                return "bad_price"
            if o.price % self.cfg.price_tick:
                return f"price_not_on_tick({self.cfg.price_tick})"
        return None

    def _taker_fill(self, o: Order, limit: Optional[int]) -> None:
        fills = self.book.walk(o.side, o.remaining, limit, self.consumed)
        side = "ask" if o.side == "buy" else "bid"
        for p, q in fills:
            if o.side == "buy":
                # spot: cannot spend more MYR than we have
                cost_per = p / UBTC * (1 + self.cfg.taker_fee_bps / 1e4)
                q = min(q, int(self.available_cash() / cost_per)) if cost_per > 0 else 0
            else:
                q = min(q, self.available_btc())
            if q <= 0:
                break
            self._fill(o, p, q, "taker")
            self.consumed[(side, p)] = self.consumed.get((side, p), 0) + q
            if q < dict(fills).get(p, 0):
                break

    def _fill(self, o: Order, price: int, q: int, liquidity: str) -> None:
        fee = self._fee(price * q / UBTC, liquidity)
        f = Fill(self.now, o.id, o.side, price, q, fee, liquidity, o.tag)
        o.filled += q
        self.pf.apply_fill(f)
        self.fills.append(f)
        if o.remaining == 0:
            o.status = "filled"
            self.pf.working.pop(o.id, None)

    def _on_new(self, o: Order) -> None:
        if o.status != "pending":
            return
        # NB: a cancel sent after this order arrives after it (FIFO), so the
        # order still gets its chance to execute first, as on a real venue.
        why = self._validate(o)
        if why:
            return self._reject(o, why)
        bb, ba = self.book.best_bid, self.book.best_ask
        if o.type == "market":
            if (ba if o.side == "buy" else bb) is None:
                return self._reject(o, "no_liquidity")
            if o.side == "sell" and self.available_btc() < self.cfg.min_order_ubtc:
                return self._reject(o, "insufficient_btc")
            if o.side == "buy" and self.available_cash() <= 0:
                return self._reject(o, "insufficient_myr")
            self._taker_fill(o, None)
            if o.status != "filled":            # IOC: remainder cancelled
                o.status = "filled" if o.filled else "cancelled"
                if not o.filled:
                    o.reject_reason = "no_fill"
                self.pf.working.pop(o.id, None)
            return
        # limit
        crosses = (o.side == "buy" and ba is not None and o.price >= ba) or \
                  (o.side == "sell" and bb is not None and o.price <= bb)
        if crosses and o.post_only:
            return self._reject(o, "post_only_would_take")
        if o.side == "buy" and self.available_cash() < o.price * o.remaining / UBTC * (1 + self.cfg.maker_fee_bps / 1e4):
            return self._reject(o, "insufficient_myr")
        if o.side == "sell" and self.available_btc() < o.remaining:
            return self._reject(o, "insufficient_btc")
        if crosses:
            self._taker_fill(o, o.price)
            if o.status == "filled":
                return
        o.status = "resting"
        if self.cfg.queue_model == "optimistic":
            o.queue_ahead = 0
        else:
            o.queue_ahead = self.book.volume_at("bid" if o.side == "buy" else "ask", o.price)

    def _on_cancel(self, oid) -> None:
        o = self.orders.get(oid)
        if o is None or not o.is_working:
            return               # already filled/cancelled: cancel is a no-op
        o.status = "cancelled"
        self.pf.working.pop(o.id, None)

    def _match_trade(self, t: Event) -> None:
        # a taker SELL prints into bids -> can fill our resting buys, and vice versa
        our_side = "buy" if t.side == "sell" else "sell"
        resting = [o for o in self.pf.working.values()
                   if o.status == "resting" and o.side == our_side]
        if not resting:
            return
        # best-priced first: highest buy / lowest sell
        resting.sort(key=lambda o: (-o.price if our_side == "buy" else o.price, o.id))
        avail = t.volume
        qm = self.cfg.queue_model
        for o in resting:
            if avail <= 0:
                break
            through = (t.price < o.price) if our_side == "buy" else (t.price > o.price)
            at = t.price == o.price
            if through:
                q = min(o.remaining, avail)
            elif at and qm != "through_only":
                excess = avail - o.queue_ahead
                o.queue_ahead = max(0, o.queue_ahead - avail)
                q = min(o.remaining, max(0, excess))
            else:
                continue
            if q > 0:
                self._fill(o, o.price, q, "maker")
                avail -= q

    # ------------------------------------------------------------ stats
    def stats(self) -> dict:
        all_o = list(self.orders.values())
        filled_any = sum(1 for o in all_o if o.filled > 0)
        lat = sorted(self.latencies_ms)
        return {
            "orders_submitted": self.n_submitted,
            "orders_with_fill": filled_any,
            "fill_ratio_orders": filled_any / self.n_submitted if self.n_submitted else 0.0,
            "fill_ratio_volume": (sum(o.filled for o in all_o) / self.submitted_volume
                                  if self.submitted_volume else 0.0),
            "n_fills": len(self.fills),
            "maker_fills": sum(1 for f in self.fills if f.liquidity == "maker"),
            "taker_fills": sum(1 for f in self.fills if f.liquidity == "taker"),
            "exchange_rejects": len(self.rejects),
            "exchange_reject_reasons": _count(r["reason"].split("(")[0] for r in self.rejects),
            "sim_network_latency_ms": {
                "mean": sum(lat) / len(lat) if lat else 0.0,
                "p50": lat[len(lat) // 2] if lat else 0.0,
                "max": lat[-1] if lat else 0.0,
            },
        }


def _count(it) -> dict:
    d: dict = {}
    for k in it:
        d[k] = d.get(k, 0) + 1
    return d
