"""Event-driven backtester (and the loop reused by live paper trading).

Per event:
  1. exchange.on_event(e)   - orders that reached the exchange before e.ts execute
                              against the book; then e updates the L2 book and
                              resting orders (trades -> queue/fills)
  2. features = TopBook.apply(e)     (the FPGA golden model)    [timed]
  3. intents  = strategy.on_event()                              [timed]
  4. approved = risk.check(intents)                              [timed]
  5. exchange.submit(approved, decision_ts=e.ts)  -> arrive after network latency
  6. risk.on_tick() - daily-loss kill switch (cancel all + flatten + halt)

PAPER TRADING ONLY.
"""
from __future__ import annotations
import csv
import json
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..events import Event
from ..features import TopBook
from .exchange import ExchangeConfig, SimExchange
from .orders import Portfolio, UBTC
from .risk import RiskConfig, RiskGate
from .strategy import Strategy

DAY_MS = 86_400_000


@dataclass
class BacktestConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    start_cash: float = 50_000.0
    start_btc: int = 50_000             # ubtc; lets a spot account sell first
    sample_ms: int = 1_000              # equity-curve resolution
    sharpe_bucket_ms: int = 60_000      # PnL-change buckets for the Sharpe-ish stat
    time_strategy: bool = True


class _Timer:
    __slots__ = ("vals",)

    def __init__(self) -> None:
        self.vals: list[int] = []

    def summary(self) -> dict:
        v = sorted(self.vals)
        if not v:
            return {"n": 0}
        n = len(v)
        return {"n": n, "mean_us": sum(v) / n / 1e3, "p50_us": v[n // 2] / 1e3,
                "p99_us": v[min(n - 1, int(n * 0.99))] / 1e3, "max_us": v[-1] / 1e3}


class Engine:
    """Holds exchange + risk + strategy; feed it events one at a time."""

    def __init__(self, strategy: Strategy, cfg: Optional[BacktestConfig] = None,
                 feature_fn=None) -> None:
        self.cfg = cfg or BacktestConfig()
        self.strategy = strategy
        self.pf = Portfolio(cash=self.cfg.start_cash, btc=self.cfg.start_btc)
        self.ex = SimExchange(self.cfg.exchange, self.pf)
        self.risk = RiskGate(self.cfg.risk)
        self.topbook = TopBook()
        self.feature_fn = feature_fn          # optional override (live, deep books)
        self.t_book, self.t_feat, self.t_strat, self.t_risk = _Timer(), _Timer(), _Timer(), _Timer()
        self.curve: list[tuple[int, float, float, int, float]] = []  # ts, mid, pnl, net, fees
        self._next_sample = None
        self.n_events = 0
        self.first_ts: Optional[int] = None
        self.last_ts: Optional[int] = None
        self.n_intents = 0
        self.last_mid: Optional[float] = None
        self.first_mid: Optional[float] = None

    def step(self, e: Event) -> list:
        pc = time.perf_counter_ns
        t0 = pc()
        self.ex.on_event(e)
        t1 = pc()
        feats = self.feature_fn(e, self.ex.book) if self.feature_fn else self.topbook.apply(e)
        t2 = pc()
        intents = self.strategy.on_event(e, feats, self.ex.book, self.pf)
        t3 = pc()
        approved = self.risk.check(intents, e.ts, self.ex.book, self.pf) if intents else []
        t4 = pc()
        self.t_book.vals.append(t1 - t0); self.t_feat.vals.append(t2 - t1)
        self.t_strat.vals.append(t3 - t2)
        if intents:
            self.t_risk.vals.append(t4 - t3)
        self.n_intents += len(intents)
        for it in approved:
            self.ex.submit(it, e.ts)
        for it in self.risk.on_tick(e.ts, self.ex.book, self.pf, self.cfg.exchange.min_order_ubtc):
            self.ex.submit(it, e.ts)
        # bookkeeping
        self.n_events += 1
        if self.first_ts is None:
            self.first_ts = e.ts
        self.last_ts = e.ts
        mid = self.ex.book.mid()
        if mid is not None and not self.ex.book.is_crossed():
            self.last_mid = mid
            if self.first_mid is None:
                self.first_mid = mid
        if self.last_mid is not None:
            if self._next_sample is None or e.ts >= self._next_sample:
                self.curve.append((e.ts, self.last_mid, self.pf.trading_pnl(self.last_mid),
                                   self.pf.net, self.pf.fees_paid))
                self._next_sample = e.ts + self.cfg.sample_ms
        return approved

    def finish(self) -> None:
        self.ex.flush()
        if self.last_mid is not None and self.last_ts is not None:
            self.curve.append((self.last_ts, self.last_mid, self.pf.trading_pnl(self.last_mid),
                               self.pf.net, self.pf.fees_paid))

    # ---------------------------------------------------------------- report
    def results(self) -> dict:
        mark = self.last_mid or 0.0
        pnl = self.pf.trading_pnl(mark) if mark else 0.0
        dur_ms = (self.last_ts - self.first_ts) if self.first_ts is not None else 0
        # max drawdown on the sampled PnL curve
        peak, mdd = -math.inf, 0.0
        for _, _, p, _, _ in self.curve:
            peak = max(peak, p)
            mdd = max(mdd, peak - p)
        # Sharpe-ish: PnL changes per bucket, scaled to one day
        buckets: dict[int, float] = {}
        for ts, _, p, _, _ in self.curve:
            buckets[ts // self.cfg.sharpe_bucket_ms] = p
        keys = sorted(buckets)
        diffs = [buckets[b] - buckets[a] for a, b in zip(keys, keys[1:])]
        sharpe = None
        if len(diffs) >= 2:
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
            if sd > 0:
                sharpe = m / sd * math.sqrt(DAY_MS / self.cfg.sharpe_bucket_ms)
        start_eq = self.pf.base_cash + self.pf.base_btc * (self.first_mid or mark) / UBTC
        ex = self.ex.stats()
        net_lat_ms = ex["sim_network_latency_ms"]["mean"]
        feat, strat, risk, book = (self.t_feat.summary(), self.t_strat.summary(),
                                   self.t_risk.summary(), self.t_book.summary())
        compute_us = sum(t.get("mean_us", 0.0) for t in (feat, strat)) + \
            (risk.get("mean_us", 0.0) * risk.get("n", 0) / max(1, self.n_events))
        return {
            "strategy": self.strategy.name,
            "strategy_params": self.strategy.params(),
            "events": self.n_events,
            "duration_hours": dur_ms / 3.6e6,
            "start_mid": self.first_mid, "end_mid": mark,
            "pnl_myr_net": pnl,
            "pnl_myr_gross": pnl + self.pf.fees_paid,
            "fees_myr": self.pf.fees_paid,
            "pnl_bps_of_start_equity": pnl / start_eq * 1e4 if start_eq else 0.0,
            "sharpe_per_day_est": sharpe,
            "max_drawdown_myr": mdd,
            "turnover_myr": self.pf.traded_notional,
            "turnover_x_equity": self.pf.traded_notional / start_eq if start_eq else 0.0,
            "traded_btc": self.pf.traded_volume / UBTC,
            "final_net_btc": self.pf.net / UBTC,
            "intents": self.n_intents,
            **ex,
            "risk_rejections": len(self.risk.rejections),
            "risk_rejections_by_rule": self.risk.rejection_counts(),
            "kill_switch": {"halted": self.risk.halted, "reason": self.risk.halt_reason,
                            "events": self.risk.kill_events},
            "latency": {
                "per_event_python": {"exchange_book_update": book, "features_topbook": feat,
                                     "strategy": strat, "risk_check_when_intents": risk},
                "python_decision_mean_us": compute_us,
                "sim_network_mean_ms": net_lat_ms,
                "network_over_compute_ratio": (net_lat_ms * 1e3 / compute_us) if compute_us else None,
            },
            "config": {"exchange": self.cfg.exchange.to_dict(), "risk": self.cfg.risk.to_dict(),
                       "start_cash": self.cfg.start_cash, "start_btc": self.cfg.start_btc / UBTC},
        }

    def write_curve(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "mid", "pnl_myr", "net_ubtc", "fees_myr"])
            for row in self.curve:
                w.writerow([row[0], f"{row[1]:.1f}", f"{row[2]:.4f}", row[3], f"{row[4]:.4f}"])


def run_backtest(events: Iterable[Event], strategy: Strategy,
                 cfg: Optional[BacktestConfig] = None) -> Engine:
    eng = Engine(strategy, cfg)
    for e in events:
        eng.step(e)
    eng.finish()
    return eng


def load_events(spec: str, n_events: int = 20_000) -> list[Event]:
    """'synth:SEED' or 'synth:SEED:N' or a .jsonl/.jsonl.gz path."""
    if spec.startswith("synth:"):
        parts = spec.split(":")
        seed = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        n = int(parts[2]) if len(parts) > 2 else n_events
        from ..synth import generate
        return generate(n, seed=seed)
    from ..events import read_jsonl
    return list(read_jsonl(spec))


def dump_results(res: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
