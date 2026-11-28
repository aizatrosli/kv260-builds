# ftrade.predictor: tiny price-direction MLP on the KV260 TPU stack

A 16 -> 32 -> 16 -> 3 ReLU6 MLP predicts whether the mid price goes down, stays flat or goes up
over the next H = 20 messages. Its input is the last W = 16 TopBook feature vectors. The model is
trained in PyTorch, quantised to INT8 with `tpuquant`, compiled to the tpuv1 ISA with `tpucc` and
run on tpuv1's ISA simulator through `tpucc.runtime` and `tpurt`'s sim backend. Paper trading only.

| file | what it does |
|---|---|
| `dataset.py` | events -> integer input codes + labels, time-ordered purged split (normalisation table in the module docstring) |
| `model.py` | `DirectionMLP` (ReLU6 hidden layers) |
| `train.py` | training, baselines, INT8 PTQ, compile, bit-exact checks, tpuperf latency; `python -m ftrade.predictor.train` |
| `tpu.py` | bridge to tpu-stack: `quantize`, `to_tpucc`, `compile_tpucc`, `run_tpusim`, `TpurtMLP`, `board_latency` |
| `infer.py` | `Predictor` (backends `float` / `int8` / `tpusim` / `board`), `stream()`, `StreamPredictor` |
| `artifacts/` | `model.pt`, `norm.json`, `qmodel/` (tpuquant export), `metrics.json` (48 KB in total) |
| `../../tests/test_predictor.py` | 11 tests pass, plus 1 board-only test that is skipped without the board |

## Inputs and labels (all integer)

- The book is valid when bid > 0, ask > 0 and spread > 0. Invalid rows (one side empty, for
  example halfway through a snapshot rebuild) take the value of the last valid row.
- There are 16 int8 codes per message. The model sees `code / 32`, and the input scale is fixed
  to 1/32 in tpuquant (`fixed_qparams`), so the codes are exactly the TPU's input bytes. The codes
  come from shifts, adds and two integer divides (spread and window-mean depth):
  - imbalance: now, window mean, 4-message mean, change
  - taker flow: now, window mean, change over 4 messages
  - spread / pu
  - microprice offset relative to the spread
  - mid returns over 1, 2, 4, 8 and 15 messages
  - number of mid changes in the window
  - depth relative to its window mean
- `pu` (price unit), `fs` (flow shift) and `db` (deadband) are integers fitted on training rows
  only. On synth they are pu = 10, fs = 6, db = 7 (in mid2 = bid + ask units, so db = 3.5 MYR).
- Label: d = mid2[t+H] - mid2[t]. Up if d > db, down if d < -db, flat otherwise (about 30% flat).
- Split: within each stream, the first 70% / next 15% / last 15% of messages are
  train / val / test. Purge gaps make sure no two splits share a message. Tests check this, and
  also check that the normalisation does not depend on rows after the training part.

## Results on ftrade.synth

- Training data: seeds 0-3, 150k events each. That gives 420k train, 90k val and 90k test samples.
- Each model was trained 3 times with different torch seeds (AdamW, early stopping on val loss).
- "Fresh" means seeds 100 and 101 (75k events each): whole streams that played no part in
  training or in fitting the normalisation.
- Edge = mean over non-flat calls of signal x (mid[t+H] - mid[t]), in MYR, mid to mid, with no
  costs counted.

| predictor | test acc (mean ± sd, 3 seeds) | fresh acc | hit rate when it calls up/down (test) | share of samples called up/down | edge MYR/call (test / fresh) |
|---|---|---|---|---|---|
| **MLP (float)** | **0.5915 ± 0.0010** | **0.582** | **0.72** | 0.67 | **13.5 / 12.1** |
| MLP int8 (tpuquant, = TPU) | 0.5889 | | | | |
| majority class ("down") | 0.4492 | 0.443 | 0.60 | 1.00 | 0.7 / 0.6 |
| imbalance sign | 0.3559 | 0.349 | 0.45 | 0.99 | 0.8 / 0.4 |
| imbalance sign, flat band \|imb code\| <= 40 (fitted on train) | 0.3863 | 0.385 | 0.50 | 0.34 | 5.7 / 3.9 |

How to read these numbers:

- The MLP beats the majority baseline by about 14 points of accuracy. That signal is learnable
  in the *synthetic* market, and it holds up on unseen streams.
- Part of it comes from the generator rather than from "order flow". When synth rebuilds the
  book, it removes old levels before adding new ones. This makes mid2 changes skewed: the median
  is negative, the mean is about zero. That skew is why "down" is the majority class.
- The instantaneous imbalance sign is useless here, with a hit rate below 0.5. The features that
  correlate with the future move are window-mean imbalance (+0.10), flow (+0.14) and microprice
  offset (+0.14).
- The edge is small. About 13 MYR per call is roughly 1.3 median spreads (the spread is about
  10 MYR), mid to mid. Crossing the spread costs about 1 spread per round trip, and a Luno taker
  fee of about 0.1% is about 250 MYR at 250k MYR/BTC. After costs, a taker strategy with this
  model loses money even on synth.

### Real data (`data/xbtmyr_sample.jsonl.gz`)

This file is the 5-minute REST sample recorded by the marketdata agent: 1005 events, 3 clears,
2 trades. **The mid price never changed** during it: the spread was 1 MYR and the book sat at one
price. All labels are flat, and every accuracy is trivially 1.0, so nothing can be learned or
judged from it. The pipeline itself runs end to end on it, including a bit-exact sim run on 116
inputs.

## INT8, compile, simulator

| check | result |
|---|---|
| float vs int8 argmax agreement, test, 89,860 samples | minmax 0.9681, percentile 0.9689, **mse 0.9696 (shipped)** |
| ... on samples where the float top-2 logit margin is > 0.1 (88.6% of them) | 0.993 |
| float -> int8 test accuracy | 0.5902 -> 0.5889 |
| tpuquant `run_int` vs tpucc program on tpuv1's ISA sim (numpy-accelerated `FastCoreSim`), 300 inputs, B=300 | **bit-exact, 0 mismatches** |
| same, on tpuv1's *unmodified* pure-Python `CoreSim`, 300 inputs | **bit-exact** |
| same network as tpurt `Gemm` steps on tpurt's `sim` backend (the code path `board` uses), 300 inputs | **bit-exact** |
| program | 1 kernel, 156 instructions (132 of them SET_CONFIG), 2.5 KB, 1.3 KB of int8 weight tiles (1,072 real weights) |

- With plain ReLU the agreement was only 0.92. The first hidden layer had a long tail, and
  calibration clipped it (7 dB SQNR). ReLU6 bounds the hidden ranges, which fixed it at no cost
  in float accuracy.
- The remaining disagreements are near-ties in the 3 int8 logits.

### Estimated latency on the board (tpuperf, `tpuv1_board` config)

| batch | total | matmul | DMA I/O | host overhead | per inference |
|---|---|---|---|---|---|
| 1 | **14.3 us** | 7.4 us | 2.9 us | 4.0 us | 14.3 us |
| 16 | 15.9 us | 8.6 us | 3.3 us | 4.0 us | 1.0 us |

- About half of the single-inference latency is fixed overhead: the 4 us ioctl/host cost, plus
  program upload and DMA.
- The array is under 1% used at B=1. Batching windows, or the requant-table / no-SYNC control
  changes in `tpu-stack/perfmodel/RECOMMENDATIONS.md`, are the levers.
- Python feature computation and the host Predictor overhead are not included. They are larger
  than the TPU time.

## Using it

```python
from ftrade.predictor.infer import Predictor, StreamPredictor, stream
p = Predictor()                                  # artifacts/, backend "float"
p = Predictor(backend="int8")                    # or "tpusim", or "board" (needs /dev/tpu0)
p.window                                         # W = 16 rows needed
out = p.predict(rows)   # rows: [>= 16, 8] TopBook feature vectors, oldest first
# {'p_down':..., 'p_flat':..., 'p_up':..., 'signal': -1|0|1, 'valid': bool, 'cls': 0|1|2}

sp = StreamPredictor(p)                          # engine already runs TopBook
pred = sp.update(features)                       # None until 16 valid rows were seen
for ev, feats, pred in stream(events, p, every=1):   # or let it run TopBook
    ...
```

- `Predictor(min_conf=0.5)` turns an up/down call with probability under 0.5 into signal 0.
- `predict` holds invalid rows only within the rows you pass it. `StreamPredictor` and `stream`
  hold online, so their inputs match the training set exactly.
- `board` builds a `tpurt` `GraphExecutor` with `get_backend("board")`.
  `Predictor(backend="board", fallback_sim=True)` falls back to tpurt's sim backend when there
  is no board.

## Retraining on real recorded data

```bash
cd fpga-trading
python -m ftrade.predictor.train --data data/xbtmyr_a.jsonl.gz --data data/xbtmyr_b.jsonl.gz \
    --out ftrade/predictor/artifacts_real          # add --no-tpu to skip quantise/compile
python -m pytest tests/test_predictor.py           # the tests use artifacts/ (synth)
Predictor("ftrade/predictor/artifacts_real", backend="int8")
```

- Each file is one stream, and each stream is split by time on its own.
- `pu`, `fs` and `db` are refitted on the training part: pu = median spread, fs from the flow
  std, db so that about 30% of labels are flat.
- Make sure the mid price actually moves over H messages. Check `label_dist` in `metrics.json`.
  For a slow pair like XBTMYR, record hours of data, not minutes.
- Raise H (`NormConfig.H`) or use websocket diffs. REST snapshots make "20 messages" a very
  short horizon.
- With a few days of data, a walk-forward evaluation (several consecutive time splits) is more
  honest than one 70/15/15 split.

## tpu-stack gaps worked around here (nothing in tpu-stack was edited)

- **No tpucc frontend for tpuquant graphs.** `frontends.from_mlp` only takes tpuv1's
  `compile_mlp.Net`. `tpu.to_tpucc` builds the IR from tpuquant `linear` params with
  `tpucc.frontends._linear`, a private helper. ReLU6 becomes a ReLU with `act_max = round(6/s)`.
- **tpurt's MLP step builder is private** (`selftest._mlp_steps`, which needs K and N in multiples
  of 16). `tpu._tpurt_steps` is a copy that pads the 3 logits to 16 lanes.
- **`tpucc.refexec` imports jepa-tpu's `intops` for every op**, even for a pure MLP. That means
  `jepa-tpu/` has to be present to run the tpusim backend.
