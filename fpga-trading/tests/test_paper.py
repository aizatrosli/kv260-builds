"""Tests for ftrade.paper: book, simulated exchange, risk gate, backtest."""
from __future__ import annotations
import json
import pathlib
import random
import re

import pytest

from ftrade.events import Event
from ftrade.features import TopBook
from ftrade.paper.book import L2Book
from ftrade.paper.exchange import ExchangeConfig, SimExchange
from ftrade.paper.orders import OrderIntent, Portfolio
from ftrade.paper.risk import RiskConfig, RiskGate
from ftrade.paper.strategy import (FlowOracle, MarketMaker, MomentumTaker,
                                   PredictorStrategy)
from ftrade.paper.backtest import BacktestConfig, Engine, run_backtest
from ftrade import synth

T0 = 1_790_000_000_000


def lv(side, price, vol, ts=T0):
    return Event("level", ts, side, price, vol)


def tr(side, price, vol, ts=T0):
    return Event("trade", ts, side, price, vol)


def snapshot(ts=T0, bids=((100, 1000), (99, 2000), (98, 3000)),
             asks=((101, 1000), (102, 2000), (103, 3000))):
    ev = [Event("clear", ts)]
    ev += [lv("bid", p, v, ts) for p, v in bids]
    ev += [lv("ask", p, v, ts) for p, v in asks]
    return ev


def make_ex(latency=0.0, jitter=0.0, queue="conservative", taker=50.0, maker=25.0,
            cash=1e9, btc=10_000_000, min_order=500):
    cfg = ExchangeConfig(taker_fee_bps=taker, maker_fee_bps=maker, latency_ms=latency,
                         latency_jitter_ms=jitter, queue_model=queue, min_order_ubtc=min_order)
    ex = SimExchange(cfg, Portfolio(cash=cash, btc=btc))
    for e in snapshot():
        ex.on_event(e)
    return ex


def tick(ex, ts):
    """advance time with a harmless event"""
    ex.on_event(Event("level", ts, "bid", 50, 1))


# ------------------------------------------------------------------ book
def test_book_sorted_and_remove():
    b = L2Book()
    for e in snapshot():
        b.apply(e)
    assert b.best_bid == 100 and b.best_ask == 101 and b.mid() == 100.5
    b.apply(lv("bid", 100, 0))
    assert b.best_bid == 99
    b.apply(lv("bid", 100, 0))        # removing absent level is a no-op
    b.apply(lv("ask", 100, 5))
    assert b.best_ask == 100 and b.levels("ask", 2) == [(100, 5), (101, 1000)]
    b.apply(lv("ask", 100, 7))        # overwrite
    assert b.volume_at("ask", 100) == 7 and b.depth("ask") == 4
    b.apply(Event("clear", T0))
    assert b.best_bid is None and b.mid() is None


def test_book_matches_topbook_top8_random():
    """Full book's top-8 equals TopBook whenever TopBook could not lose levels."""
    rng = random.Random(3)
    b, tb = L2Book(), TopBook()
    b.apply(Event("clear", T0)); tb.apply(Event("clear", T0))
    for _ in range(3000):
        side = rng.choice(["bid", "ask"])
        p = rng.randint(1000, 1100) if side == "bid" else rng.randint(1101, 1200)
        v = rng.choice([0, 0, rng.randint(1, 5000)])
        e = lv(side, p, v)
        b.apply(e); tb.apply(e)
        # TopBook holds a subset of the true book with correct volumes
        for p2, v2 in tb.bids:
            assert b.bids[p2] == v2
        for p2, v2 in tb.asks:
            assert b.asks[p2] == v2
        if tb.bids:
            assert tb.bids[0][0] <= b.best_bid


def test_book_walk():
    b = L2Book()
    for e in snapshot():
        b.apply(e)
    assert b.walk("buy", 2500) == [(101, 1000), (102, 1500)]
    assert b.walk("sell", 1500) == [(100, 1000), (99, 500)]
    assert b.walk("buy", 5000, limit_price=102) == [(101, 1000), (102, 2000)]
    assert b.walk("buy", 1500, consumed={("ask", 101): 800}) == [(101, 200), (102, 1300)]


# ------------------------------------------------------------------ fills
def test_market_order_walks_depth_and_charges_taker_fee():
    ex = make_ex()
    cash0, btc0 = ex.pf.cash, ex.pf.btc
    ex.submit(OrderIntent.market("buy", 2500), T0)
    tick(ex, T0 + 1)
    assert [(f.price, f.volume, f.liquidity) for f in ex.fills] == [(101, 1000, "taker"), (102, 1500, "taker")]
    notional = (101 * 1000 + 102 * 1500) / 1e6
    fee = notional * 50 / 1e4
    assert ex.pf.btc == btc0 + 2500
    assert ex.pf.cash == pytest.approx(cash0 - notional - fee)
    assert ex.pf.fees_paid == pytest.approx(fee)


def test_consumed_liquidity_not_reused_until_refresh():
    ex = make_ex()
    ex.submit(OrderIntent.market("buy", 1000), T0)
    ex.submit(OrderIntent.market("buy", 1000), T0)
    tick(ex, T0 + 1)
    assert [f.price for f in ex.fills] == [101, 102]
    ex.on_event(lv("ask", 101, 1000, T0 + 2))       # feed refreshes the level
    ex.submit(OrderIntent.market("buy", 500), T0 + 2)
    tick(ex, T0 + 3)
    assert ex.fills[-1].price == 101


def test_latency_means_stale_book():
    ex = make_ex(latency=100)
    ex.submit(OrderIntent.market("buy", 1000), T0)          # decided when ask=101
    ex.on_event(lv("ask", 101, 0, T0 + 50))                 # book moves before arrival
    assert not ex.fills
    ex.on_event(lv("ask", 105, 9000, T0 + 150))              # arrival at T0+100 happens first
    assert ex.fills[0].price == 102 and ex.fills[0].ts == T0 + 100


def test_latency_jitter_range_and_fifo():
    ex = make_ex(latency=125, jitter=75)
    for i in range(200):
        ex.submit(OrderIntent.limit("buy", 90, 500), T0 + i)
    lat = ex.latencies_ms
    assert min(lat) >= 50 - 1e-9 and max(lat) <= 200 + 1e-9
    arrivals = [o.arrival_ts for o in ex.orders.values()]
    assert arrivals == sorted(arrivals)


def test_min_size_and_tick():
    ex = SimExchange(ExchangeConfig(latency_ms=0, latency_jitter_ms=0, price_tick=5),
                     Portfolio(cash=1e9, btc=10**7))
    for e in snapshot(bids=((100, 1000),), asks=((105, 1000),)):
        ex.on_event(e)
    a = ex.submit(OrderIntent.limit("buy", 95, 499), T0)
    b = ex.submit(OrderIntent.limit("buy", 97, 1000), T0)
    c = ex.submit(OrderIntent.limit("buy", 95, 1000), T0)
    tick(ex, T0 + 1)
    assert a.status == "rejected" and "min_size" in a.reject_reason
    assert b.status == "rejected" and "tick" in b.reject_reason
    assert c.status == "resting"


def test_post_only_rejected_when_crossing():
    ex = make_ex()
    o = ex.submit(OrderIntent.limit("buy", 101, 1000, post_only=True), T0)
    tick(ex, T0 + 1)
    assert o.status == "rejected" and o.reject_reason == "post_only_would_take"
    o2 = ex.submit(OrderIntent.limit("buy", 101, 1500), T0 + 1)   # not post-only: takes
    tick(ex, T0 + 2)
    assert o2.filled == 1000 and o2.status == "resting" and o2.remaining == 500


def test_limit_conservative_queue():
    ex = make_ex()
    o = ex.submit(OrderIntent.limit("buy", 100, 1000, post_only=True), T0)
    tick(ex, T0 + 1)
    assert o.status == "resting" and o.queue_ahead == 1000
    ex.on_event(tr("buy", 101, 5000, T0 + 2))   # buy-side trade cannot fill our bid
    assert o.filled == 0
    ex.on_event(tr("sell", 100, 600, T0 + 3))   # at price, eats queue first
    assert o.filled == 0 and o.queue_ahead == 400
    ex.on_event(tr("sell", 100, 700, T0 + 4))   # 400 queue left, 300 for us
    assert o.filled == 300 and o.queue_ahead == 0
    f = ex.fills[-1]
    assert f.liquidity == "maker" and f.price == 100
    assert f.fee_myr == pytest.approx(100 * 300 / 1e6 * 25 / 1e4)
    ex.on_event(tr("sell", 99, 5000, T0 + 5))   # strictly through: rest fills at our price
    assert o.status == "filled" and ex.fills[-1].price == 100


def test_limit_queue_shrinks_on_level_decrease():
    ex = make_ex()
    o = ex.submit(OrderIntent.limit("buy", 100, 1000), T0)
    tick(ex, T0 + 1)
    ex.on_event(lv("bid", 100, 200, T0 + 2))     # cancellations ahead of us
    assert o.queue_ahead == 200
    ex.on_event(tr("sell", 100, 500, T0 + 3))
    assert o.filled == 300


def test_through_only_and_optimistic_models():
    ex = make_ex(queue="through_only")
    o = ex.submit(OrderIntent.limit("buy", 100, 1000), T0)
    tick(ex, T0 + 1)
    ex.on_event(tr("sell", 100, 10_000, T0 + 2))
    assert o.filled == 0
    ex.on_event(tr("sell", 99, 400, T0 + 3))
    assert o.filled == 400                       # through, limited by trade size
    ex2 = make_ex(queue="optimistic")
    o2 = ex2.submit(OrderIntent.limit("buy", 100, 1000), T0)
    tick(ex2, T0 + 1)
    ex2.on_event(tr("sell", 100, 300, T0 + 2))
    assert o2.filled == 300


def test_cancel_with_latency_and_spot_balance():
    ex = make_ex(latency=100)
    o = ex.submit(OrderIntent.limit("buy", 100, 1000), T0)
    ex.submit(OrderIntent.cancel(o.id), T0 + 10)
    tick(ex, T0 + 105)
    assert o.status == "resting"                 # cancel not there yet
    tick(ex, T0 + 300)
    assert o.status == "cancelled" and not ex.pf.working
    ex2 = make_ex(btc=1000)
    s = ex2.submit(OrderIntent.limit("sell", 110, 2000), T0)
    tick(ex2, T0 + 1)
    assert s.status == "rejected" and s.reject_reason == "insufficient_btc"


# ------------------------------------------------------------------ risk
def risk_setup(**kw):
    cfg = RiskConfig(**{"max_position_btc": 0.01, "max_order_btc": 0.005, "max_notional_myr": 1e9,
                        "max_orders_per_sec": 3, "max_daily_loss_myr": 10.0, "collar_bps": 50.0,
                        "max_data_age_ms": 1000, **kw})
    b = L2Book()
    for e in snapshot(bids=((250000, 10**6), (249900, 10**6)),
                      asks=((250010, 10**6), (250100, 10**6))):
        b.apply(e)
    return RiskGate(cfg), b, Portfolio(cash=1e6, btc=10**6)


def test_risk_max_order_size():
    g, b, pf = risk_setup()
    assert not g.check([OrderIntent.limit("buy", 250000, 6000)], T0, b, pf)
    assert g.rejections[-1]["rule"] == "max_order_size"
    assert g.check([OrderIntent.limit("buy", 250000, 5000)], T0, b, pf)


def test_risk_max_notional():
    g, b, pf = risk_setup(max_notional_myr=1000.0)
    assert not g.check([OrderIntent.limit("buy", 250000, 5000)], T0, b, pf)   # 1250 MYR
    assert g.rejections[-1]["rule"] == "max_notional"
    assert g.check([OrderIntent.limit("buy", 250000, 3000)], T0, b, pf)


def test_risk_max_position_counts_working_orders():
    g, b, pf = risk_setup(max_data_age_ms=10**9)
    pf.btc += 6000                                   # net +0.006
    ok = g.check([OrderIntent.limit("buy", 250000, 3000),
                  OrderIntent.limit("buy", 250000, 3000)], T0, b, pf)
    assert len(ok) == 1 and g.rejections[-1]["rule"] == "max_position"
    # sells reduce risk and pass
    assert g.check([OrderIntent.limit("sell", 250010, 5000)], T0 + 2000, b, pf)


def test_risk_rate_limit_rolls():
    g, b, pf = risk_setup()
    ints = [OrderIntent.limit("buy", 250000, 500) for _ in range(5)]
    ok = g.check(ints, T0, b, pf)
    assert len(ok) == 3 and g.rejection_counts()["rate_limit"] == 2
    assert len(g.check(ints[:1], T0 + 999, b, pf)) == 0
    assert len(g.check(ints[:1], T0 + 1000, b, pf)) == 1


def test_risk_price_collar_limit_and_market():
    g, b, pf = risk_setup(max_data_age_ms=10**9)
    assert not g.check([OrderIntent.limit("buy", 248000, 500)], T0, b, pf)   # ~80 bps away
    assert g.rejections[-1]["rule"] == "price_collar"
    assert g.check([OrderIntent.market("buy", 500)], T0 + 2000, b, pf)
    # a market order that would sweep far through the book
    b.set_level("ask", 250010, 100); b.set_level("ask", 250100, 0); b.set_level("ask", 260000, 10**6)
    assert not g.check([OrderIntent.market("buy", 5000)], T0 + 4000, b, pf)
    assert g.rejections[-1]["rule"] == "price_collar"


def test_risk_stale_data_and_crossed():
    g, b, pf = risk_setup()
    assert not g.check([OrderIntent.limit("buy", 250000, 500)], T0 + 1001, b, pf)
    assert g.rejections[-1]["rule"] == "stale_data" and "age" in g.rejections[-1]["reason"]
    b.set_level("bid", 250020, 5); b.last_update_ts = T0 + 2000
    assert not g.check([OrderIntent.limit("buy", 250000, 500)], T0 + 2000, b, pf)
    assert "crossed" in g.rejections[-1]["reason"]
    assert not g.check([OrderIntent.limit("buy", 250000, 500)], T0, L2Book(), pf)


def test_risk_cancels_always_pass_and_every_reject_logged():
    g, b, pf = risk_setup()
    g.halted, g.halt_reason = True, "test"
    ok = g.check([OrderIntent.cancel(1), OrderIntent.limit("buy", 250000, 500)], T0, b, pf)
    assert [i.action for i in ok] == ["cancel"]
    r = g.rejections[-1]
    assert r["rule"] == "kill_switch" and r["reason"] and r["intent"]["side"] == "buy"


def test_kill_switch_flattens_and_halts():
    ex = make_ex(cash=1e6, btc=1_000_000)
    g = RiskGate(RiskConfig(max_daily_loss_myr=0.0001, max_data_age_ms=10**9))
    pf = ex.pf
    assert g.on_tick(T0, ex.book, pf) == []             # sets day baseline
    # buy 3000 ubtc (pays fees -> loss), and rest a bid
    ex.submit(OrderIntent.market("buy", 3000), T0)
    ex.submit(OrderIntent.limit("buy", 95, 1000), T0)
    tick(ex, T0 + 1)
    assert pf.net == 3000 and len(pf.working) == 1
    kill = g.on_tick(T0 + 1, ex.book, pf)
    assert g.halted and [k.action for k in kill] == ["cancel_all", "new"]
    assert kill[1].side == "sell" and kill[1].volume == 3000 and kill[1].type == "market"
    for k in kill:
        ex.submit(k, T0 + 1)
    tick(ex, T0 + 2)
    assert pf.net == 0 and not pf.working
    # halted: strategy orders are now refused
    assert not g.check([OrderIntent.limit("buy", 100, 1000)], T0 + 3, ex.book, pf)
    assert g.rejections[-1]["rule"] == "kill_switch"
    assert g.on_tick(T0 + 3, ex.book, pf) == []          # nothing left to flatten
    g.reset()
    assert g.check([OrderIntent.limit("buy", 100, 1000)], T0 + 3, ex.book, pf)


def test_kill_switch_in_backtest_engine():
    cfg = BacktestConfig(risk=RiskConfig(max_daily_loss_myr=5.0))
    eng = run_backtest(synth.generate(20000, seed=2), MarketMaker(half_spread_bps=0.0), cfg)
    r = eng.results()
    assert r["kill_switch"]["halted"]
    assert abs(r["final_net_btc"]) < 0.0005            # flattened (below min order)
    k_ts = r["kill_switch"]["events"][0]["ts"]
    assert all(f.tag == "kill" for f in eng.ex.fills if f.ts > k_ts + 1000)


# ------------------------------------------------------------------ backtest
def _run(strategy, seed=5, n=15000, **cfgkw):
    return run_backtest(synth.generate(n, seed=seed), strategy, BacktestConfig(**cfgkw)).results()


def _strip(r):
    r = json.loads(json.dumps(r, default=str))
    r.pop("latency")
    return r


@pytest.mark.parametrize("mk", [MomentumTaker, MarketMaker,
                                lambda: PredictorStrategy(predict=FlowOracle())])
def test_backtest_deterministic(mk):
    a, b = _run(mk()), _run(mk())
    assert _strip(a) == _strip(b)
    assert a["events"] == 15000


def test_backtest_accounting_consistent():
    r = _run(MomentumTaker(), seed=7)
    assert r["pnl_myr_gross"] - r["fees_myr"] == pytest.approx(r["pnl_myr_net"])
    assert r["orders_submitted"] > 0 and r["taker_fills"] > 0
    assert r["fees_myr"] == pytest.approx(r["turnover_myr"] * 50 / 1e4, rel=1e-9)
    assert r["max_drawdown_myr"] >= 0


def test_zero_fee_and_latency_options_change_results():
    base = _run(MomentumTaker(), seed=7)
    nofee = _run(MomentumTaker(), seed=7,
                 exchange=ExchangeConfig(taker_fee_bps=0, maker_fee_bps=0))
    assert nofee["fees_myr"] == 0 and nofee["pnl_myr_net"] > base["pnl_myr_net"]
    slow = _run(MomentumTaker(), seed=7, exchange=ExchangeConfig(latency_ms=500, latency_jitter_ms=0))
    assert slow["sim_network_latency_ms"]["mean"] >= 500


def test_predictor_adapter_accepts_plain_callable_and_window():
    seen = []

    def predict(w):
        seen.append(w.shape)
        return {"signal": 1}

    s = PredictorStrategy(predict=predict, window=4, size=1000)
    r = _run(s, n=3000)
    assert seen and all(sh == (4, 8) for sh in seen)
    assert r["orders_submitted"] >= 1 and r["intents"] >= 1


def test_predictor_missing_module_fails_clearly(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ftrade.predictor.infer", None)
    with pytest.raises(RuntimeError, match="ftrade.predictor"):
        PredictorStrategy()


def test_live_fails_gracefully_without_marketdata(monkeypatch):
    import sys
    from ftrade.paper import live
    monkeypatch.setitem(sys.modules, "ftrade.marketdata", None)
    with pytest.raises(RuntimeError, match="ftrade.marketdata"):
        live.run_live(MomentumTaker(), seconds=0.1)


def test_live_engine_with_fake_marketdata(monkeypatch):
    import sys
    import types
    from ftrade.paper import live
    evs = synth.generate(3000, seed=4)
    fake = types.ModuleType("ftrade.marketdata")
    fake.iter_events = lambda pair, mode="rest", seconds=None, **kw: iter(evs)
    monkeypatch.setitem(sys.modules, "ftrade.marketdata", fake)
    eng = live.run_live(MarketMaker(), seconds=1, status_every_s=1e9)
    assert eng.n_events == 3000


def test_l2_features_equal_topbook_when_book_shallow():
    from ftrade.paper.live import L2Features
    evs = synth.generate(5000, seed=9, depth=6)    # never more than 8 levels/side? check
    b, tb, lf = L2Book(), TopBook(), L2Features()
    for e in evs:
        b.apply(e)
        f1 = tb.apply(e)
        f2 = lf(e, b)
        if b.depth("bid") <= 8 and b.depth("ask") <= 8:
            assert f1 == f2


def test_cli_backtest_writes_outputs(tmp_path):
    from ftrade.paper.__main__ import main
    out, csvp = tmp_path / "r.json", tmp_path / "eq.csv"
    rc = main(["backtest", "--data", "synth:3:4000", "--strategy", "mm",
               "--latency-ms", "150", "--fee-bps", "60", "--out", str(out), "--equity-csv", str(csvp)])
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["config"]["exchange"]["taker_fee_bps"] == 60
    assert "latency" in r and csvp.read_text().startswith("ts,mid,pnl_myr")


# ------------------------------------------------------------------ safety guard
FORBIDDEN = ["postorder", "marketorder", "stoporder", "api_key", "apikey", "api_secret",
             "luno_api_key", "/api/1/orders", "/api/exchange/1/orders", "/api/1/withdrawals",
             "/api/1/send", "/api/1/balance", "basic_auth", "httpbasicauth", "authorization"]


def test_no_private_endpoints_in_paper_package():
    pkg = pathlib.Path(__file__).resolve().parents[1] / "ftrade" / "paper"
    files = sorted(pkg.rglob("*.py"))
    assert files
    for f in files:
        low = f.read_text().lower()
        for text in (low, low.replace("_", "")):
            for bad in FORBIDDEN:
                assert bad not in text and bad.replace("_", "") not in text, \
                    f"{f.name} references {bad!r}"
        assert "import requests" not in f.read_text()
