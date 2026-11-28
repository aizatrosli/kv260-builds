# ftrade.paper: paper trading, risk gate and backtester

**PAPER TRADING ONLY.** Nothing in this package places real orders. It does not
import an HTTP client, does not read API keys and does not touch authenticated
Luno endpoints. Orders are Python objects matched against a local copy of the
public order book. `tests/test_paper.py::test_no_private_endpoints_in_paper_package`
enforces this.

Run every command from `fpga-trading/` so that `ftrade` is importable.

## Quick start

```bash
# backtest on synthetic data (seed 1, 100k events, about 1.3 h of simulated time)
python -m ftrade.paper backtest --data synth:1:100000 --strategy momentum
python -m ftrade.paper backtest --data synth:1:100000 --strategy mm
python -m ftrade.paper backtest --data synth:1:100000 --strategy predictor --oracle  # built-in rule, no ML
python -m ftrade.paper backtest --data synth:1 --strategy predictor                  # needs ftrade/predictor/infer.py

# recorded data from ftrade.marketdata
python -m ftrade.paper backtest --data recordings/xbtmyr.jsonl.gz --strategy mm

# what-ifs
python -m ftrade.paper backtest --data synth:1:100000 --strategy momentum --fee-bps 0 --maker-fee-bps 0
python -m ftrade.paper backtest --data synth:1:100000 --strategy momentum --latency-ms 500 --jitter-ms 0
python -m ftrade.paper backtest --data synth:1:100000 --strategy mm --queue optimistic --half-spread-bps 0

# save outputs
python -m ftrade.paper backtest --data synth:2 --strategy mm --out res.json --equity-csv eq.csv

# paper trade on live PUBLIC data (needs ftrade.marketdata; Ctrl-C stops and prints the report)
python -m ftrade.paper live --pair XBTMYR --strategy mm --seconds 600
```

`python -m ftrade.paper backtest -h` lists every option. These include the risk
limits (`--max-pos-btc`, `--max-order-btc`, `--max-notional`, `--max-ops`,
`--max-daily-loss`, `--collar-bps`, `--max-age-ms`) and strategy size
(`--size-ubtc`).

Tests: `python -m pytest tests/test_paper.py -q`

## Modules

| file | what |
|---|---|
| `book.py` | `L2Book`: the full-depth book built from `clear`/`level` events. `walk()` simulates sweeping the book. `TopBook` keeps only 8 levels, so it is not enough to simulate fills. |
| `orders.py` | `OrderIntent` (what a strategy asks for), `Order`, `Fill`, `Portfolio` (spot balances, `net` position, PnL measured against holding the starting balances). |
| `exchange.py` | `SimExchange`: latency, market and limit orders, post-only, queue model, fees, minimum size and tick, spot balance checks. |
| `risk.py` | `RiskGate`: the pre-trade checks and the kill switch. |
| `strategy.py` | `Strategy` interface, `MomentumTaker`, `MarketMaker`, `PredictorStrategy`, `FlowOracle`. |
| `backtest.py` | `Engine` (the event loop shared by backtest and live), stats and timing. |
| `live.py` | Live paper trading on `ftrade.marketdata` (imported lazily). Features come from the L2 book, so a deep REST book still gives exact top-8 features. |

### Event loop (one iteration per market event)
1. **Exchange.** Any order whose arrival time is at or before the event's timestamp executes against the book as it stands just before the event. Then the event updates the L2 book and the resting orders.
2. **Features.** `TopBook.apply(event)` produces the 8 int32 features. This is the FPGA golden model.
3. **Strategy.** `strategy.on_event(event, features, book, portfolio)` returns a list of `OrderIntent`.
4. **Risk.** `risk.check(intents, ...)` passes or rejects each intent, and logs every rejection.
5. **Submit.** Approved intents are sent to the exchange. Each arrives at `now + latency`.
6. **Kill switch.** `risk.on_tick()` checks the daily loss. On a breach it cancels all orders, flattens the position and halts.

### Exchange model
- **Latency.** Each order or cancel is delayed by a uniform draw from `latency_ms ± jitter_ms`. The default is 100 ± 50 ms, i.e. 50 to 150 ms. Orders arrive FIFO, as on one TCP connection. The decision is made on the book at time t and executes on the book at t + latency.
- **Market orders** are immediate-or-cancel and walk the depth. Liquidity we take is remembered for each level until the feed refreshes that level, so two orders cannot both fill against the same displayed size.
- **Limit orders.**
  - The part that crosses fills as taker. The rest rests on the book.
  - A `post_only` order that would cross is rejected.
  - Resting orders fill **only from printed trades**, using a queue model:
    - `conservative` (default). The queue ahead of us is the displayed size at our price when the order arrives. A trade strictly through our price fills us, up to the trade's size. A trade at our price consumes the queue ahead first, and only the excess fills us. When the level shrinks, the queue ahead shrinks with it.
    - `through_only`. Only trades strictly through our price fill us.
    - `optimistic`. We are first in the queue.
  - The book crossing our resting price without a trade does **not** fill us.
- **Fees** are charged in bps of notional, in MYR. The defaults are **taker 50 bps, maker 25 bps**.
  - Public reviews of Luno Malaysia quote about 0.25 to 0.35% maker and 0.25 to 0.50% taker for retail 30-day volume.
  - Luno's own help centre gives maker fees from -0.02% to 0.40% depending on tier and region.
  - The defaults are the conservative end of those ranges. **Check the fee page in your own account**, then set `--fee-bps` and `--maker-fee-bps`.
- **Minimum order** is 500 ubtc (0.0005 BTC). **Tick** is 1 MYR.
- **Spot balances.** You cannot sell BTC you do not hold or spend MYR you do not have. The backtest starts with 50,000 MYR and 0.05 BTC, so a strategy can go short relative to its start. PnL is measured against simply holding those starting balances, so BTC price drift on the starting inventory is not counted as strategy PnL.

### Risk gate (`RiskConfig`)
These are the checks HFT firms put in hardware, on the order path:

| rule | default | |
|---|---|---|
| `max_position` | 0.02 BTC | worst case \|net + working orders on that side + this order\| |
| `max_order_size` | 0.01 BTC | per order |
| `max_notional` | 3000 MYR | per order. Market orders are priced by walking the book. |
| `rate_limit` | 5 / s | rolling 1 s window. Cancels count toward it but are never rejected. |
| `price_collar` | 50 bps | limit price, or a market order's expected average price, compared with mid |
| `stale_data` | 5000 ms | rejects if the book is older than this, a side is empty, or the book is crossed |
| `kill_switch` | 250 MYR / day | the day runs on Malaysia time (UTC+8). On a breach it cancels all, flattens with market orders (retrying until flat) and halts until `RiskGate.reset()`. |

Every rejection is appended to `RiskGate.rejections` as `{ts, rule, reason, intent}` and logged at INFO level; use `-v` to see them. The report gives counts per rule. Once the switch fires, every later strategy intent is rejected as `kill_switch`. So a large `kill_switch` count only means "the strategy kept asking after it was halted".

## What the numbers mean
- **PnL net (MYR)**: trading PnL after all fees, marked at the final mid. **gross** is the same before fees.
- **Sharpe/day est**: the mean over standard deviation of 1-minute PnL changes, scaled by √1440. This is an *extrapolation*: a synthetic run of 20k to 100k events covers only about 0.3 to 1.3 hours. Treat it as a sign and a rough size, not a real Sharpe ratio.
- **max drawdown**: the largest peak-to-trough fall of the PnL curve, sampled every second.
- **turnover**: total notional traded, in MYR, and as a multiple of starting equity.
- **fill ratio**: the share of orders with any fill, and the share of submitted volume that filled. A market maker's fill ratio is low by design, because most quotes are cancelled.
- **exch rejects**: orders the simulated exchange refused, such as a post-only order that would cross, an order below the minimum size, or insufficient balance.
- **risk rejects**: orders the pre-trade gate refused, counted by rule.
- **latency**: Python `perf_counter_ns` time per event for the book update, TopBook features, the strategy and the risk check (risk is timed only on events that produced intents). It is compared with the simulated network latency. Python decides in about 5 to 25 µs, and the simulated internet adds about 100 ms. The network is thousands of times larger, so on Luno over the public internet the FPGA's nanoseconds do not matter for PnL. The FPGA part of this project teaches how low-latency feature and risk logic is built; it is not a speed edge here.

## Example results (synthetic, 100k events, about 1.3 h each, default config)

| seed | momentum net / gross / fees | mm net / gross / fees | predictor (oracle) net / gross |
|---|---|---|---|
| 1 | -251 / +16 / 268 (kill) | -37 / -3 / 34 | -253 / +19 (kill) |
| 2 | -251 / +5 / 256 (kill) | -45 / -6 / 39 | -254 / +7 (kill) |
| 3 | -253 / +6 / 259 (kill) | -23 / -2 / 21 | -254 / +6 (kill) |
| 4 | -250 / +13 / 263 (kill) | -34 / -7 / 27 | -253 / +20 (kill) |

The momentum taker has a real but tiny gross edge on the synthetic market: with zero fees it makes +13 to +28 MYR per run. That is roughly 5 bps of edge on about 53k MYR of turnover. A 50 bps taker fee costs 100 bps per round trip. At 50 bps the strategy loses about 250 MYR and trips the daily kill switch. At 10 bps it still loses 60 to 85 MYR per run. Latency barely matters for this signal: zero-fee PnL goes from +29 at 0 ms, to +27 at 500 ms, to +22 at 2 s. The edge is slow and small, and the fees are large.

The market maker loses even with zero fees. Quoting 2 bps from mid on a synthetic book whose spread is about 0.1 bps, it is filled mostly when price moves through it (adverse selection). The maker fee makes the loss larger.

## Caveats (read these)
- **Synthetic data is not a market.** `ftrade.synth` has a deliberately learnable order-flow signal. Profits on it say nothing about XBTMYR.
- **Overfitting.** Tuning thresholds on a few seeds or a few hours of recordings will find "edges" that are noise. Hold out data, and test on several seeds or days. Be suspicious of any Sharpe computed from a few dozen fills.
- **Fees eat small edges.** With retail Luno fees of 25 to 50 bps per side, a strategy needs more than 50 to 100 bps of edge per round trip. Signals from microstructure (imbalance, flow) are usually a few bps.
- **Latency.** Public REST polling sees the book 0.5 to 1 s late, and orders over the internet take 50 to 200 ms or more. The simulator models the order leg; in live mode the data leg is real. Fills are reported to the strategy at once; a real exchange's acknowledgements are also delayed.
- **The fill model is an approximation.**
  - Queue position is estimated from displayed size.
  - Hidden orders, self-impact on later prints, and other traders reacting to our quotes are not modelled.
  - Trades in recorded data at a 1 s poll interval are aggregated.
  - `conservative` is the default for a reason.
- **There is no execution path to a real account, by design.** Turning this into a live trading system would need authenticated order placement, reconciliation and much more risk engineering. That is out of scope for this educational project.
