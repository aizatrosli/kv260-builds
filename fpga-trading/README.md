# fpga-trading: learning how trading firms use FPGAs, on a KV260

An educational, **paper-trading-only** BTC/MYR bot for Luno, built the way an HFT shop splits the work between software and an FPGA. Nothing here places real orders or uses private Luno endpoints. Not financial advice.

Start with [docs/HOW_TRADERS_USE_FPGAS.md](docs/HOW_TRADERS_USE_FPGAS.md), then [SPEC.md](SPEC.md) (the contract every piece builds against).

```
Luno public REST ──► ftrade.marketdata ──► Events (JSONL) ──► ftrade.wire (16-byte msgs, AXI-Stream)
                                                                   │
                                         ┌─────────────────────────┴──────────────┐
                                         ▼ (hardware, rtl/)                        ▼ (golden, ftrade/features.py)
                                 tob_engine.v: 8-level book + 8 int features  ═══ bit-exact ═══  TopBook
                                         │
                                         ▼
                          ftrade.predictor: 16→32→16→3 INT8 MLP on the tpuv1 TPU (tpuquant → tpucc → tpurt)
                                         │
                                         ▼
                          ftrade.paper: strategy → risk gate (kill switch) → simulated Luno exchange → PnL
```

| Piece | Folder | Status (simulation only, board not yet here) |
|---|---|---|
| Explainer | `docs/` | What HFT firms put in FPGAs, and why it doesn't carry over to Luno |
| Market data | `ftrade/marketdata/` | REST collector (no key) + optional websocket stream (read-only key); 28 tests; 5-min real sample in `data/` |
| Book + features in RTL | `rtl/`, `tb/` | Verilator: 0 mismatches over 10 × 70k vectors with random stalls; 25 cycles (100 ns @ 250 MHz); ~6.4k LUT, 4 DSP (Yosys estimate) |
| Predictor on the TPU | `ftrade/predictor/` | Bit-exact on the tpuv1 ISA sim; ~14 µs/inference estimated on the board; 97% int8/float agreement |
| Paper trading | `ftrade/paper/` | Simulated fills, latency, Luno-like fees, risk gate, backtester, live paper mode on public data |

## Run it

```bash
cd fpga-trading
python3 -m pytest -q tests                                    # 90 pass, 1 skip (board)
tb/run.sh                                                     # RTL vs golden model (needs verilator)
python3 -m ftrade.marketdata record --seconds 600 --out data/xbtmyr.jsonl.gz
python3 -m ftrade.marketdata latency --n 20                   # your RTT to api.luno.com
python3 -m ftrade.paper backtest --data synth:1 --strategy momentum
python3 -m ftrade.paper backtest --data synth:1 --strategy predictor
python3 -m ftrade.paper live --pair XBTMYR --strategy mm      # paper orders on live public data
```

The predictor and tpusim paths need `torch` (CPU) and tpuv1 cloned at `/home/user/tpuv1` (or `TPUV1_ROOT`), like `tpu-stack/`.

## What it shows (the honest takeaways)

- **Latency doesn't decide PnL here.** RTL: 100 ns. Python features + strategy: ~5 µs, ~240 µs with the MLP. Network to Luno: ~120-140 ms measured from a cloud container. Adding 2 s of order latency barely changed the momentum strategy's result.
- **Fees do.** Every strategy loses after Luno-like taker fees (50 bps default) and trips the 250 MYR daily kill switch. Momentum makes ~5 bps gross on synthetic data; the MLP is ~0 gross. At zero fees the results are within a few MYR of break-even.
- **The FPGA's real role at Luno's scale** is deterministic, bit-exact, low-power feature and model compute, and a place to learn the HFT building blocks (feed parsing, book building, fixed-point signals, hardware risk gates).

## Known quirks

- `micro_off` (F7) on a **crossed** book is computed on the spread as unsigned 32-bit, so it becomes huge (golden and RTL agree). Synthetic data never crosses; filter or clamp before using F7 on live data.
- The REST sample is quiet (2 trades, spread stuck at 1 MYR); the predictor needs hours of data or the websocket stream to learn anything real.
- `synth` has a generator bias (old levels removed before new ones are added) that makes "down" the majority label; see `ftrade/predictor/README.md`.
