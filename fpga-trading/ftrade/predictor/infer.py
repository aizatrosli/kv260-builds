"""Run the trained direction predictor.

    from ftrade.predictor.infer import Predictor, stream
    p = Predictor()                          # default artifacts, float backend
    p = Predictor("path/to/artifacts", backend="int8")    # or Predictor.load(path, backend=...)
    out = p.predict(window)   # window: int array [>= W, 8] of TopBook feature rows, oldest first
    # -> {"p_up", "p_flat", "p_down", "signal" (-1/0/1), "valid", "cls"}

Backends
  float   the PyTorch model on q/32 inputs
  int8    tpuquant's bit-exact integer interpreter (the spec the TPU must match)
  tpusim  the tpucc-compiled tpuv1 program on tpuv1's ISA simulator (bit-exact with int8)
  board   tpurt's board backend (/dev/tpu0, tpu_kv260 driver); same program as tpurt's sim
          backend, which ``board`` falls back to only if fallback_sim=True

For int8/tpusim/board the probabilities are softmax(dequantised int8 logits); signal = argmax - 1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from . import dataset as D

ART = Path(__file__).resolve().parent / "artifacts"
BACKENDS = ("float", "int8", "tpusim", "board")


def _softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Predictor:
    def __init__(self, path=None, backend="float", min_conf: float = 0.0, fallback_sim=False):
        path = Path(path) if path else ART
        if path.is_file():
            path = path.parent
        self.path = path
        self.cfg = D.NormConfig.from_dict(json.loads((path / "norm.json").read_text()))
        self.window = self.window_size = self.cfg.W
        self.horizon = self.cfg.H
        self.min_conf = float(min_conf)          # signal 0 unless max(p_up, p_down) >= min_conf
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r}: choose from {BACKENDS}")
        self.backend = backend
        self._model = self._qg = self._prog = self._tm = None
        if backend == "float":
            from .model import load
            self._model = load(path / "model.pt")
        else:
            from . import tpu  # noqa: F401  (tpu-stack on sys.path)
            from tpuquant import export
            self._qg = export.load(path / "qmodel")
            if backend == "board":
                from . import tpu as T
                be = "board"
                if fallback_sim and not os.path.exists("/dev/tpu0"):
                    be = "sim"
                self._tm = T.TpurtMLP(self._qg, be, B=1)
                self.device_backend = be

    @classmethod
    def load(cls, path=None, backend="float", **kw):
        return cls(path, backend, **kw)

    # ---- core -----------------------------------------------------------------------------
    def logits_codes(self, X):
        """X: int8 input codes [B, 16] -> float logits [B, 3] (dequantised for int backends)."""
        X = np.asarray(X, np.int64).reshape(-1, D.N_IN)
        if self.backend == "float":
            import torch
            with torch.no_grad():
                return self._model(torch.tensor((X * D.IN_SCALE).astype(np.float32))).double().numpy()
        from . import tpu as T
        if self.backend == "int8":
            q = T.int_codes(self._qg, X)
        elif self.backend == "tpusim":
            if self._prog is None or self._prog_B != len(X):
                self._prog, self._prog_B = T.compile_tpucc(self._qg, len(X)), len(X)
            q, _ = T.run_tpusim(self._prog, X)
        else:
            q = self._tm.run(X)
        return T.logits_from_int(self._qg, q)

    def _out(self, p, valid=True):
        cls = int(np.argmax(p))
        sig = cls - 1
        if not valid or (sig != 0 and p[cls] < self.min_conf):
            sig = 0
        return dict(p_down=float(p[0]), p_flat=float(p[1]), p_up=float(p[2]), signal=int(sig),
                    valid=bool(valid), cls=cls)

    def inputs(self, window):
        return D.inputs_from_window(window, self.cfg)

    def predict(self, window_features):
        """window_features: [>= W, 8] TopBook rows, oldest first. Invalid rows (empty side / crossed)
        are replaced by the previous valid row inside the given rows; rows before the last W are
        used only for that. If the book was invalid for longer than the rows given, pass more
        rows (or use StreamPredictor, which holds online and matches the dataset exactly).
        Window whose first row has no valid book -> flat prior, signal 0, valid False."""
        win = np.asarray(window_features, np.int64).reshape(-1, 8)
        if not D.window_valid(win, self.cfg):
            return self._out(np.array([0.0, 1.0, 0.0]), valid=False)
        x = self.inputs(win)
        return self._out(_softmax(self.logits_codes(x[None]))[0])

    def predict_batch(self, X):
        """X: int8 input codes [B, 16] (from dataset) -> list of prediction dicts."""
        P = _softmax(self.logits_codes(X))
        return [self._out(p) for p in P]

    def close(self):
        if self._tm is not None:
            self._tm.close()
            self._tm = None


class _Hold:
    """Online 'hold last valid book' register (dataset.hold_valid, one row at a time)."""

    def __init__(self):
        self.last = None

    def __call__(self, f):
        f = list(f)
        if f[0] > 0 and f[1] > 0 and f[2] > 0:
            self.last = f
        return self.last


def stream(events: Iterable, predictor: Optional[Predictor] = None, every: int = 1,
           book=None) -> Iterator[tuple]:
    """Consume Events, maintain a TopBook, and yield (event, features[8], prediction) after
    every `every`-th message once W rows are available. Pass `book` to share a TopBook.
    Invalid book rows are held online, so the inputs equal the dataset's exactly."""
    from ..features import TopBook
    p = predictor or Predictor()
    tb = book or TopBook()
    sp = StreamPredictor(p)
    n = 0
    for e in events:
        f = tb.apply(e)
        n += 1
        ready = sp.push(f)
        if n >= p.window and n % every == 0:
            yield e, f, (sp.predict() if ready else p._out(np.array([0.0, 1.0, 0.0]), valid=False))


class StreamPredictor:
    """Push-style helper for an engine that already runs TopBook: .update(features) -> pred|None
    (None until W valid rows have been seen). Holds the last valid book across empty/crossed rows."""

    def __init__(self, predictor: Optional[Predictor] = None):
        self.p = predictor or Predictor()
        self.hold = _Hold()
        self.buf = []

    def push(self, features) -> bool:
        h = self.hold(features)
        if h is not None:
            self.buf.append(h)
            if len(self.buf) > self.p.window:
                del self.buf[0]
        return len(self.buf) == self.p.window

    def predict(self):
        return self.p.predict(self.buf)

    def update(self, features):
        return self.predict() if self.push(features) else None
