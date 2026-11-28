"""Bridge from the trained PyTorch MLP to the KV260 TPU stack (tpu-stack/, imported read-only).

  quantize()        tpuquant INT8 PTQ (input scale fixed to 1/32 so the dataset's int8 codes
                    are the TPU input bytes, zp 0 everywhere)
  to_tpucc()        tpuquant integer graph -> tpucc IR (matmul + requant + relu/relu6 clamp per layer)
  compile_tpucc()   tpucc lower + codegen -> tpuv1 ISA program
  run_tpusim()      the compiled program on tpuv1's unmodified ISA simulator (tpucc.runtime)
  TpurtMLP          the same network as tpurt Gemm steps on a tpurt backend (sim / mock / board)
  board_latency()   tpuperf estimate of one inference on the board

tpucc has no frontend for tpuquant graphs (only jepa-tpu, tpuv1 compile_mlp and compile_cnn), so
``to_tpucc`` builds the IR directly from tpuquant's linear-node params; tpurt.selftest's MLP step
builder is private, so ``TpurtMLP`` has its own copy (pads the 3 logits to 16 lanes).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

BUILDS = Path(__file__).resolve().parents[3]
TPU_STACK = BUILDS / "tpu-stack"
os.environ.setdefault("TPUV1_ROOT", "/home/user/tpuv1")
os.environ.setdefault("TPUV1_DIR", os.environ["TPUV1_ROOT"])
for _sub in ("quant", "compiler", "runtime", "perfmodel"):
    _p = str(TPU_STACK / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .dataset import IN_SCALE, N_IN  # noqa: E402

I64 = np.int64


def _pad16(n):
    return (n + 15) // 16 * 16


# ------------------------------------------------------------------------------- tpuquant ----
def quantize(model, calib_x, act_method="minmax"):
    """INT8 PTQ of the DirectionMLP. calib_x: float [M, 16] (= codes / 32)."""
    import torch
    from tpuquant import Calibrator, QConfig
    cal = Calibrator(model.eval(), torch.as_tensor(np.asarray(calib_x, np.float32)))
    ph = cal.placeholders[0].name
    g = cal.build(QConfig(act_method=act_method, act_symmetric=True,
                          fixed_qparams={ph: (IN_SCALE, 0)}))
    return g


def qlayers(qg):
    """[(w int [K,N], bias, m0, shift, zp_out, act_min, act_max)] from the tpuquant graph."""
    out = []
    for n in qg.nodes:
        if n.op == "linear":
            a, p = n.attrs, n.params
            if a["zp_in"] != 0:
                raise ValueError("expected symmetric activations (zp_in 0)")
            out.append((p["w"].astype(I64), p["bias"].astype(I64), p["m0"].astype(I64),
                        p["shift"].astype(I64), int(a["zp_out"]), int(a["act_min"]), int(a["act_max"])))
        elif n.op not in ("quantize",):
            raise NotImplementedError(f"tpu bridge: unexpected tpuquant op {n.op}")
    return out


def int_codes(qg, xq):
    """Integer interpreter on int8 input codes: returns int8 logits [B, 3] (tpuquant run_int)."""
    xq = np.asarray(xq, I64)
    return qg.run_int(xq * IN_SCALE)[qg.outputs[0]]


def logits_from_int(qg, q):
    return qg.dequant(qg.outputs[0], q)


# ---------------------------------------------------------------------------------- tpucc ----
def to_tpucc(qg, B, name="ftrade_mlp"):
    from tpucc.ir import Graph
    from tpucc.frontends import _linear
    g = Graph(name)
    x = g.input("x", (B, N_IN))
    for l, (w, b, m0, sh, zp, lo, hi) in enumerate(qlayers(qg)):
        wt = g.const(f"fc{l}.w", w)
        relu = lo > -128 or hi < 127
        if relu and not (lo == zp == 0):
            raise NotImplementedError(f"fc{l}: clamp [{lo},{hi}] with zp {zp} is not a ReLU/ReLU6")
        # ReLU6 = activation-unit ReLU with act_max = round(6 / s_out) (act_min/act_max registers)
        x = _linear(g, f"fc{l}", x, wt, b, m0, sh, zp, act="relu" if relu else None,
                    act_min=lo, act_max=hi)
    g.mark_output(x)
    g.validate()
    return g


def compile_tpucc(qg, B):
    from tpucc import passes, codegen
    g = to_tpucc(qg, B)
    return codegen.compile_graph(passes.lower(g))


def run_tpusim(prog, xq, golden=False):
    """Run the compiled program on tpuv1's ISA simulator. golden=True: the unmodified pure-Python
    CoreSim (slow), else tpucc's numpy-vectorised subclass. Returns int8 logits [B, 3]."""
    from tpucc import runtime
    r = runtime.run_program(prog, {"x": np.asarray(xq, I64)}, fast=not golden)
    return r.outputs[prog.graph.outputs[0]], r


# ---------------------------------------------------------------------------------- tpurt ----
def _tpurt_steps(B, layers):
    from tpurt.program import Gemm, Host, Tensor, pack_weight_tiles
    tin = Tensor("x", (N_IN // 16) * B)

    def load(ctx):
        ctx.out(tin)[...] = np.ascontiguousarray(
            np.asarray(ctx.input).reshape(B, -1, 16).transpose(1, 0, 2)).reshape(-1, 16)
        ctx.done(tin)

    steps, cur = [Host("input", load, [], [tin])], tin
    for i, (w, b, m0, sh, zp, lo, hi) in enumerate(layers):
        K, Nn = w.shape
        Kp, Np = _pad16(K), _pad16(Nn)
        Wp = np.zeros((Kp, Np), I64); Wp[:K, :Nn] = w
        pad = lambda v: np.concatenate([np.asarray(v, I64), np.zeros(Np - Nn, I64)])
        out = Tensor(f"y{i}", (Np // 16) * B, "out" if i == len(layers) - 1 else "act")
        wt = Tensor(f"w{i}", Kp * Np // 16, "const", np.frombuffer(pack_weight_tiles(Wp), np.int8))
        steps.append(Gemm(f"fc{i}", cur, 0, B, Kp // 16, B, out, 0, B, Np // 16, wt, 0, pad(m0), pad(sh),
                          pad(b), act=(zp, lo, hi), op=f"fc{i}"))
        cur = out
    n_out = layers[-1][0].shape[1]
    Np = _pad16(n_out)

    def collect(ctx, y=cur):
        ctx.result = ctx.read(y).reshape(Np // 16, B, 16).transpose(1, 0, 2).reshape(B, Np)[:, :n_out].astype(I64)
    steps.append(Host("output", collect, [cur], []))
    return steps


class TpurtMLP:
    """The quantised MLP as a tpurt GraphExecutor on a backend ('sim', 'mock', 'board', ...)."""

    def __init__(self, qg, backend="sim", B=1, **backend_kw):
        from tpurt import get_backend
        from tpurt.executor import GraphExecutor
        self.B = B
        self.backend = get_backend(backend, **backend_kw) if isinstance(backend, str) else backend

        class _Ex(GraphExecutor):
            def __init__(s, be, steps):
                super().__init__(be, steps, contexts=1, ops="numpy", arena_size=1 << 20)
        self.ex = _Ex(self.backend, _tpurt_steps(B, qlayers(qg)))

    def run(self, xq):
        xq = np.asarray(xq, I64).reshape(-1, N_IN)
        outs = []
        for i in range(0, len(xq), self.B):
            blk = xq[i:i + self.B]
            n = len(blk)
            if n < self.B:
                blk = np.concatenate([blk, np.zeros((self.B - n, N_IN), I64)])
            outs.append(self.ex.run_one(blk.astype(np.int8))[:n])
        return np.concatenate(outs)

    def close(self):
        self.ex.close()
        try:
            self.backend.close()
        except Exception:
            pass


# -------------------------------------------------------------------------------- tpuperf ----
def board_latency(dims=(16, 32, 16, 3), batch=1):
    """tpuperf estimate (tpuv1 as built on the KV260, DDR-fed weights, program upload, 4 us host
    overhead) for one batch. Returns dict(total_us, per-part breakdown)."""
    from tpuperf import simulate, tpuv1_board
    from tpuperf.workloads import mlp
    r = simulate(mlp("ftrade_mlp", list(dims), batch), tpuv1_board())
    row = r.row()
    row["us_per_inference"] = round(r.total_us / batch, 2)
    return row


def sim_available():
    try:
        import tpucc.env  # noqa: F401  (imports tpuv1 model/isa sim + asm)
        import tpurt.backends.sim  # noqa: F401
        return True
    except Exception:
        return False
