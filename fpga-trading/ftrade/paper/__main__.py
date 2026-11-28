"""CLI: python -m ftrade.paper {backtest,live} ...   (PAPER TRADING ONLY)"""
from __future__ import annotations
import argparse
import json
import logging
import sys

from .backtest import BacktestConfig, dump_results, load_events, run_backtest
from .exchange import ExchangeConfig, QUEUE_MODELS
from .risk import RiskConfig
from .strategy import FlowOracle, MarketMaker, MomentumTaker, PredictorStrategy


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", choices=["momentum", "mm", "predictor"], default="momentum")
    p.add_argument("--latency-ms", type=float, default=100.0, help="mean one-way order latency")
    p.add_argument("--jitter-ms", type=float, default=50.0, help="uniform +/- latency jitter")
    p.add_argument("--fee-bps", type=float, default=50.0, help="taker fee, bps of notional")
    p.add_argument("--maker-fee-bps", type=float, default=25.0)
    p.add_argument("--queue", choices=QUEUE_MODELS, default="conservative")
    p.add_argument("--min-order-ubtc", type=int, default=500)
    p.add_argument("--seed", type=int, default=0, help="latency RNG seed")
    p.add_argument("--size-ubtc", type=int, default=None, help="strategy order/position size")
    p.add_argument("--half-spread-bps", type=float, default=None, help="mm only")
    p.add_argument("--predictor-model", default=None, help="path for Predictor.load")
    p.add_argument("--predictor-backend", default=None, help="float|int8|tpusim|board")
    p.add_argument("--oracle", action="store_true",
                   help="predictor strategy with the built-in FlowOracle instead of ML")
    p.add_argument("--max-pos-btc", type=float, default=0.02)
    p.add_argument("--max-order-btc", type=float, default=0.01)
    p.add_argument("--max-notional", type=float, default=3000.0)
    p.add_argument("--max-ops", type=int, default=5, help="max orders per second")
    p.add_argument("--max-daily-loss", type=float, default=250.0)
    p.add_argument("--collar-bps", type=float, default=50.0)
    p.add_argument("--max-age-ms", type=int, default=5000)
    p.add_argument("--out", default=None, help="results JSON path")
    p.add_argument("--equity-csv", default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="log every rejection")


def make_strategy(a):
    kw = {}
    if a.size_ubtc:
        kw["size"] = a.size_ubtc
    if a.strategy == "momentum":
        return MomentumTaker(**kw)
    if a.strategy == "mm":
        if a.half_spread_bps is not None:
            kw["half_spread_bps"] = a.half_spread_bps
        return MarketMaker(**kw)
    if a.oracle:
        return PredictorStrategy(predict=FlowOracle(), **kw)
    return PredictorStrategy(model_path=a.predictor_model, backend=a.predictor_backend, **kw)


def make_cfg(a) -> BacktestConfig:
    return BacktestConfig(
        exchange=ExchangeConfig(taker_fee_bps=a.fee_bps, maker_fee_bps=a.maker_fee_bps,
                                latency_ms=a.latency_ms, latency_jitter_ms=a.jitter_ms,
                                queue_model=a.queue, min_order_ubtc=a.min_order_ubtc,
                                seed=a.seed),
        risk=RiskConfig(max_position_btc=a.max_pos_btc, max_order_btc=a.max_order_btc,
                        max_notional_myr=a.max_notional, max_orders_per_sec=a.max_ops,
                        max_daily_loss_myr=a.max_daily_loss, collar_bps=a.collar_bps,
                        max_data_age_ms=a.max_age_ms))


def summary(r: dict) -> str:
    L = r["latency"]["per_event_python"]
    sh = r["sharpe_per_day_est"]
    lines = [
        f"strategy        {r['strategy']}   events={r['events']}  duration={r['duration_hours']:.2f} h",
        f"mid             {r['start_mid']} -> {r['end_mid']}",
        f"PnL net (MYR)   {r['pnl_myr_net']:+.2f}   gross {r['pnl_myr_gross']:+.2f}   fees {r['fees_myr']:.2f}",
        f"Sharpe/day est  {sh:+.2f}" if sh is not None else "Sharpe/day est  n/a",
        f"max drawdown    {r['max_drawdown_myr']:.2f} MYR",
        f"turnover        {r['turnover_myr']:.0f} MYR ({r['turnover_x_equity']:.2f}x equity), {r['traded_btc']:.6f} BTC",
        f"orders          {r['orders_submitted']} submitted, fill ratio {r['fill_ratio_orders']:.2%} (orders) "
        f"{r['fill_ratio_volume']:.2%} (volume); maker {r['maker_fills']} / taker {r['taker_fills']}",
        f"exch rejects    {r['exchange_rejects']} {r['exchange_reject_reasons']}",
        f"risk rejects    {r['risk_rejections']} {r['risk_rejections_by_rule']}",
        f"kill switch     {'HALTED: ' + r['kill_switch']['reason'] if r['kill_switch']['halted'] else 'not triggered'}",
        "latency (python, per event, us mean/p99):",
    ]
    for k, v in L.items():
        if v.get("n"):
            lines.append(f"  {k:<26} {v['mean_us']:8.2f} / {v['p99_us']:8.2f}   (n={v['n']})")
    ratio = r["latency"]["network_over_compute_ratio"]
    lines.append(f"  decision compute total    {r['latency']['python_decision_mean_us']:8.2f} us  vs simulated network "
                 f"{r['latency']['sim_network_mean_ms']:.1f} ms  (network is ~{ratio:,.0f}x larger)"
                 if ratio else "  (no orders sent, no network latency sampled)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ftrade.paper",
                                 description="Paper trading / backtesting for Luno XBTMYR (simulated orders only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="replay synthetic or recorded events")
    b.add_argument("--data", default="synth:0", help="synth:SEED[:N] or path.jsonl[.gz]")
    b.add_argument("--events", type=int, default=20_000, help="synthetic event count")
    _common(b)
    lv = sub.add_parser("live", help="paper trade on live PUBLIC market data")
    lv.add_argument("--pair", default="XBTMYR")
    lv.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    lv.add_argument("--interval", type=float, default=1.0, help="REST poll interval (s)")
    _common(lv)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        strat = make_strategy(a)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cfg = make_cfg(a)
    if a.cmd == "backtest":
        events = load_events(a.data, a.events)
        eng = run_backtest(events, strat, cfg)
    else:
        from .live import run_live
        try:
            eng = run_live(strat, a.pair, a.seconds, cfg, interval=a.interval)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    res = eng.results()
    res["data"] = getattr(a, "data", f"live:{getattr(a, 'pair', '')}")
    print(summary(res))
    if a.out:
        dump_results(res, a.out)
        print(f"results -> {a.out}")
    if a.equity_csv:
        eng.write_curve(a.equity_csv)
        print(f"equity curve -> {a.equity_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
