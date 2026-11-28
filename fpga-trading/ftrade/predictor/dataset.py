"""Event stream -> (window input, direction label) training set.

Everything here is integer arithmetic on the 8 x int32 TopBook features (SPEC.md), so the same
normalisation can be done in RTL / on the A53 with shifts, adds and two small integer divides.

Pipeline (per stream, ``build_stream``)
---------------------------------------
1. ``F = ftrade.features.run(events)``  -> [n, 8] int64 (one row per message).
2. *Hold last valid book*: a row is valid when best_bid > 0, best_ask > 0 and spread > 0. Invalid
   rows (one side empty, e.g. in the middle of a snapshot rebuild, or crossed) are replaced by the
   last valid row (a "hold" register). Rows before the first valid row stay invalid.
3. Input at message t = 16 int8 codes ``q[t, 0..15]`` computed from the held rows t-W+1..t
   (``W = 16``). The model's float input is exactly ``q / 32`` (Q5, int8 scale 1/32, zp 0), so
   tpuquant's input quantiser is the identity on these values and the whole chain from features
   to TPU input is integer. ``>>`` is an arithmetic shift (floor), ``//`` is floor division,
   ``clip`` saturates to [-127, 127]:

   =====  ==========================================================  =========================
   idx    definition (all on held rows; mid2 = bid + ask = 2*mid)     meaning (32 = 1.0)
   =====  ==========================================================  =========================
   0      imb_t >> 9                                                   imbalance x2  (+-64)
   1      (sum_{W} imb) >> (9 + log2 W)                                window-mean imbalance
   2      (sum_{last 4} imb) >> 11                                     4-msg mean imbalance
   3      (imb_t - imb_{t-1}) >> 9                                     imbalance change
   4      flow_t >> fs                                                 taker flow
   5      (sum_{W} flow) >> (fs + log2 W)                              window-mean flow
   6      (flow_t - flow_{t-4}) >> fs                                  flow change
   7      (32 * spread) // pu                                          spread in price units
   8      (64 * (2*micro_off - spread)) // spread                      microprice vs mid, +-64
   9-13   (4 * (mid2_t - mid2_{t-k})) // pu, k = 1, 2, 4, 8, W-1       mid returns (1.0=4 pu)
   14     4 * #{mid2 changes inside the window}                        activity
   15     (32 * depth_t) // max(1, (sum_{W} depth) >> log2 W) - 32      depth vs window mean
   =====  ==========================================================  =========================

   with depth = bid_vol5 + ask_vol5 (uint32 sum, held in int64). ``pu`` (price unit, >= 1) and
   ``fs`` (flow shift, >= 0) are integers fitted on the *training* rows only (``fit_norm``):
   pu = median spread, fs = round(log2(std(flow) / 32)). On ftrade.synth seeds 0-3: pu=10, fs=6, db=7.
4. Label at t: d = mid2_{t+H} - mid2_t on held rows (H = 20 messages). class 2 (up) if d > db,
   0 (down) if d < -db, else 1 (flat). The deadband ``db`` (mid2 units, integer) is fitted on the
   training rows so that ~30% of training labels are flat (``fit_norm``).
5. A sample at t needs rows t-W+1 .. t+H to exist and row t-W+1 to be valid (held).

Split: time-ordered per stream, train / val / test = first 70% / next 15% / last 15% of the
messages, with a purge gap: a train sample's label horizon (t+H) ends before the val block starts,
and val/test windows (t-W+1) start after the previous block's last label. No sample shares a
message with a sample of another split. Normalisation constants come from train rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

import numpy as np

from ..features import run as features_run

W_DEFAULT = 16
H_DEFAULT = 20
N_IN = 16
IN_SCALE = 1.0 / 32          # float input = int8 code * IN_SCALE
CLASSES = ("down", "flat", "up")   # class index - 1 == signal
RET_LAGS = (1, 2, 4, 8, 15)
INPUT_NAMES = ["imb", "imb_mean", "imb_mean4", "imb_d1", "flow", "flow_mean", "flow_d4", "spread",
               "micro_rel", "ret1", "ret2", "ret4", "ret8", "ret15", "activity", "depth_rel"]


@dataclass
class NormConfig:
    W: int = W_DEFAULT
    H: int = H_DEFAULT
    pu: int = 4                  # price unit (MYR) for spread / returns
    fs: int = 5                  # flow right shift
    db: int = 4                  # label deadband on mid2 (= 2*mid) change
    flat_target: float = 0.30

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return NormConfig(**{k: d[k] for k in NormConfig.__dataclass_fields__ if k in d})


def _clip(x):
    return np.clip(x, -127, 127)


def features_array(events) -> np.ndarray:
    return np.asarray(features_run(events), np.int64).reshape(-1, 8)


def hold_valid(F: np.ndarray):
    """Replace invalid rows by the last valid row. Returns (held [n,8], has_valid [n] bool)."""
    F = np.asarray(F, np.int64)
    ok = (F[:, 0] > 0) & (F[:, 1] > 0) & (F[:, 2] > 0)
    idx = np.where(ok, np.arange(len(F)), -1)
    idx = np.maximum.accumulate(idx) if len(F) else idx
    has = idx >= 0
    Hd = F[np.maximum(idx, 0)].copy()
    Hd[~has] = 0
    return Hd, has


def _rolling_sum(x, w):
    c = np.concatenate([[0], np.cumsum(x)])
    out = np.zeros_like(x)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def _lag(x, k):
    out = np.empty_like(x)
    out[k:] = x[:-k] if k else x
    out[:k] = x[0]
    return out


def window_inputs(Hd: np.ndarray, cfg: NormConfig) -> np.ndarray:
    """Held feature rows [n, 8] -> int8 input codes [n, 16] for every t (rows t < W-1 are
    garbage; callers mask them). Vectorised version of ``inputs_from_window``."""
    W = cfg.W
    lw = int(np.log2(W))
    assert 1 << lw == W, "W must be a power of two"
    b, a, sp, B, A, imb, flow, mic = (Hd[:, i] for i in range(8))
    mid2 = b + a
    depth = (B & 0xFFFFFFFF) + (A & 0xFFFFFFFF)
    q = np.zeros((len(Hd), N_IN), np.int64)
    q[:, 0] = imb >> 9
    q[:, 1] = _rolling_sum(imb, W) >> (9 + lw)
    q[:, 2] = _rolling_sum(imb, 4) >> 11
    q[:, 3] = (imb - _lag(imb, 1)) >> 9
    q[:, 4] = flow >> cfg.fs
    q[:, 5] = _rolling_sum(flow, W) >> (cfg.fs + lw)
    q[:, 6] = (flow - _lag(flow, 4)) >> cfg.fs
    q[:, 7] = (32 * sp) // cfg.pu
    spd = np.where(sp > 0, sp, 1)
    q[:, 8] = np.where(sp > 0, (64 * (2 * mic - sp)) // spd, 0)
    for j, k in enumerate((1, 2, 4, 8, W - 1)):
        q[:, 9 + j] = (4 * (mid2 - _lag(mid2, k))) // cfg.pu
    ch = np.concatenate([[0], (np.diff(mid2) != 0).astype(np.int64)])
    q[:, 14] = 4 * _rolling_sum(ch, W - 1)          # changes between rows t-W+1..t
    dm = np.maximum(1, _rolling_sum(depth, W) >> lw)
    q[:, 15] = (32 * depth) // dm - 32
    return _clip(q)


def inputs_from_window(win: np.ndarray, cfg: NormConfig) -> np.ndarray:
    """One window of raw TopBook feature rows [>= W, 8] (oldest first) -> int8 codes [16].
    Uses the last W rows; invalid rows are held from the previous valid one."""
    win = np.asarray(win, np.int64).reshape(-1, 8)
    Hd, has = hold_valid(win)
    if len(Hd) < cfg.W:
        raise ValueError(f"need at least W={cfg.W} feature rows, got {len(Hd)}")
    return window_inputs(Hd[-cfg.W:], cfg)[-1]


def window_valid(win: np.ndarray, cfg: NormConfig) -> bool:
    win = np.asarray(win, np.int64).reshape(-1, 8)
    _, has = hold_valid(win)
    return len(win) >= cfg.W and bool(has[-cfg.W])


def future_move(Hd: np.ndarray, cfg: NormConfig) -> np.ndarray:
    """[n] mid2_{t+H} - mid2_t on held rows (0 for the last H rows)."""
    mid2 = Hd[:, 0] + Hd[:, 1]
    d = np.zeros_like(mid2)
    d[:-cfg.H] = mid2[cfg.H:] - mid2[:-cfg.H]
    return d


def labels(Hd: np.ndarray, cfg: NormConfig) -> np.ndarray:
    """[n] class index (0 down, 1 flat, 2 up) for every t (last H rows meaningless)."""
    d = future_move(Hd, cfg)
    return np.where(d > cfg.db, 2, np.where(d < -cfg.db, 0, 1))


def fit_norm(Hd: np.ndarray, has: np.ndarray, end: int, cfg: Optional[NormConfig] = None) -> NormConfig:
    """Fit pu, fs, db on held rows [0, end) (training part only)."""
    cfg = NormConfig(**(cfg or NormConfig()).to_dict())
    m = has[:end]
    X = Hd[:end][m]
    sp = X[:, 2]
    cfg.pu = max(1, int(np.median(sp)))
    sd = float(np.std(X[:, 6])) or 1.0
    cfg.fs = int(max(0, round(np.log2(max(sd, 1.0) / 32))))
    mid2 = X[:, 0] + X[:, 1]
    d = np.abs(mid2[cfg.H:] - mid2[:-cfg.H])
    cfg.db = int(np.quantile(d, cfg.flat_target)) if len(d) else 0
    return cfg


@dataclass
class Split:
    X: np.ndarray          # int8 codes [n, 16] (int64 container)
    y: np.ndarray          # [n] class
    t: np.ndarray          # message index of the sample
    stream: np.ndarray     # stream id
    d: np.ndarray = None   # future mid2 change over H (2 x MYR), for edge metrics

    @property
    def xf(self):
        return (self.X * IN_SCALE).astype(np.float32)

    def __len__(self):
        return len(self.y)


def split_bounds(n: int, cfg: NormConfig, fracs=(0.70, 0.15, 0.15)):
    """[(lo, hi)] sample-index ranges for train/val/test with purge gaps (see module doc)."""
    c1 = int(n * fracs[0])
    c2 = int(n * (fracs[0] + fracs[1]))
    W, H = cfg.W, cfg.H
    return {"train": (W - 1, c1 - H), "val": (c1 + W - 1, c2 - H), "test": (c2 + W - 1, n - H)}


def build_stream(F: np.ndarray, cfg: NormConfig, stream_id=0, fracs=(0.70, 0.15, 0.15)):
    Hd, has = hold_valid(F)
    Q = window_inputs(Hd, cfg)
    Y = labels(Hd, cfg)
    D = future_move(Hd, cfg)
    t_all = np.arange(len(F))
    start_ok = np.zeros(len(F), bool)
    start_ok[cfg.W - 1:] = has[:len(F) - cfg.W + 1]
    out = {}
    for k, (lo, hi) in split_bounds(len(F), cfg, fracs).items():
        m = (t_all >= lo) & (t_all < hi) & start_ok
        out[k] = Split(Q[m], Y[m], t_all[m], np.full(int(m.sum()), stream_id), D[m])
    return out


def concat(splits: Iterable[Split]) -> Split:
    s = list(splits)
    return Split(*(np.concatenate([getattr(x, f) for x in s]) for f in ("X", "y", "t", "stream", "d")))


def build(streams, cfg: Optional[NormConfig] = None, fit=True, fracs=(0.70, 0.15, 0.15)):
    """streams: list of feature arrays [n_i, 8] (or event lists). Fits the normalisation on the
    union of the training parts (if fit) and returns (cfg, {'train','val','test': Split})."""
    Fs = [s if isinstance(s, np.ndarray) else features_array(s) for s in streams]
    cfg = cfg or NormConfig()
    if fit:
        Hs = [hold_valid(F) for F in Fs]
        ends = [split_bounds(len(F), cfg, fracs)["train"][1] for F in Fs]
        Hcat = np.concatenate([h[0][:e] for h, e in zip(Hs, ends)])
        hcat = np.concatenate([h[1][:e] for h, e in zip(Hs, ends)])
        cfg = fit_norm(Hcat, hcat, len(Hcat), cfg)
    parts = [build_stream(F, cfg, i, fracs) for i, F in enumerate(Fs)]
    return cfg, {k: concat(p[k] for p in parts) for k in ("train", "val", "test")}


def synth_streams(seeds=(0, 1, 2, 3), n_events=100_000):
    from ..synth import generate
    return [features_array(generate(n_events, seed=s)) for s in seeds]


def load_jsonl_features(path):
    from ..events import read_jsonl
    return features_array(read_jsonl(path))
