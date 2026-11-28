"""Small self-checking TPU workloads, usable on every backend (board_smoke.py, tests, bench):

  dma_loopback  READ_HOST_MEM -> UB -> WRITE_HOST_MEM, 1D (incl. >4095 rows, split) and 2D strided
  matmul16      one 16x16x16 MATMUL + per-channel requant vs numpy
  mlp           a 3-layer int8 MLP (ReLU, ReLU, none) compiled by tpurt.program vs numpy
  tpuv1_bundle  tpuv1's own MLP bundles (model/mlp/export_bundle.py output) via tpu_mlp, if present

Each returns dict(ok=bool, detail=str, us=float).
"""
import time

import numpy as np

from . import intmath, isa
from .backends.base import TpuRunError
from .buffers import DmaPool
from .executor import GraphExecutor
from .program import Gemm, Host, Tensor, pack_weight_tiles


def _res(ok, detail, t0):
    return dict(ok=bool(ok), detail=detail, us=(time.perf_counter() - t0) * 1e6)


def dma_loopback(backend, rows=5000, seed=1):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    pool = DmaPool(backend, arena_size=1 << 20)
    try:
        src = pool.alloc(rows * 16, name="src")
        dst = pool.alloc(rows * 16 + 64, name="dst")
        data = rng.integers(0, 256, rows * 16, dtype=np.uint8)
        src.write(data)
        dst.np(np.uint8)[:] = 0xEE
        dst.flush()
        prog, rel = [], []
        for r in range(0, rows, isa.MAX_DMA_ROWS):
            n = min(isa.MAX_DMA_ROWS, rows - r)
            rel.append((len(prog), *src.reloc(r * 16)))
            prog.append(isa.read_host_mem(r, 0, n))
        prog.append(isa.sync(isa.U_DMA))
        for r in range(0, rows, isa.MAX_DMA_ROWS):
            n = min(isa.MAX_DMA_ROWS, rows - r)
            rel.append((len(prog), *dst.reloc(r * 16)))
            prog.append(isa.write_host_mem(r, 0, n))
        # 2D: 3 reps of 2 rows with an 80-byte DDR stride (as tpuv1 dma_2d) into the guard area
        s2 = pool.alloc(256, name="src2d")
        d2 = pool.alloc(256, name="dst2d")
        s2.write(rng.integers(0, 256, 256, dtype=np.uint8))
        d2.np(np.uint8)[:] = 0xEE
        d2.flush()
        prog.append(isa.sync(isa.U_ALL))
        rel.append((len(prog), *s2.reloc(0)))
        prog.append(isa.read_host_mem(65530, 0, 2, 3, 80))
        prog.append(isa.sync(isa.U_DMA))
        rel.append((len(prog), *d2.reloc(0)))
        prog.append(isa.write_host_mem(65530, 0, 2, 3, 80))
        prog += [isa.sync(isa.U_ALL), isa.interrupt_host(0x100B), isa.halt()]
        res = backend.run(isa.to_bytes(prog), rel)
        if not res.ok:
            return _res(False, f"run status {res}", t0)
        got = dst.read()
        ok1 = (got[:rows * 16] == data).all() and (got[rows * 16:] == 0xEE).all()
        a, b = s2.read(), d2.read()
        want = np.full(256, 0xEE, np.uint8)
        for rep in range(3):
            want[rep * 80:rep * 80 + 32] = a[rep * 80:rep * 80 + 32]
        ok2 = (b == want).all()
        return _res(ok1 and ok2, f"{rows} rows 1D {'ok' if ok1 else 'MISMATCH'}, 2D strided {'ok' if ok2 else 'MISMATCH'}"
                    f", kernel {res.wall_us:.1f} us", t0)
    finally:
        pool.close()


class _Mini(GraphExecutor):
    def __init__(self, backend, steps, **kw):
        super().__init__(backend, steps, contexts=1, ops="numpy", arena_size=1 << 20, **kw)


def _mlp_steps(x_q, layers):
    """layers: [(W int8 [K,N], bias, m0, shift, act)] -> steps computing the MLP on x_q [B, K]."""
    B = x_q.shape[0]
    tin = Tensor("x", (x_q.shape[1] // 16) * B)

    def load(ctx):
        ctx.out(tin)[...] = np.ascontiguousarray(
            ctx.input.reshape(B, -1, 16).transpose(1, 0, 2)).reshape(-1, 16)
        ctx.done(tin)
    steps, cur = [Host("input", load, [], [tin])], tin
    for i, (W, b, m0, sh, act) in enumerate(layers):
        K, Nn = W.shape
        out = Tensor(f"y{i}", (Nn // 16) * B, "out" if i == len(layers) - 1 else "act")
        w = Tensor(f"w{i}", K * Nn // 16, "const", np.frombuffer(pack_weight_tiles(W), np.int8))
        steps.append(Gemm(f"fc{i}", cur, 0, B, K // 16, B, out, 0, B, Nn // 16, w, 0, np.asarray(m0),
                          np.asarray(sh), np.asarray(b), act=act, op=f"fc{i}"))
        cur = out

    def collect(ctx, y=cur, Nn=layers[-1][0].shape[1]):
        ctx.result = ctx.read(y).reshape(Nn // 16, B, 16).transpose(1, 0, 2).reshape(B, Nn)
    steps.append(Host("output", collect, [cur], []))
    return steps


def _ref(x, layers):
    h = x.astype(np.int64)
    for W, b, m0, sh, act in layers:
        h = intmath.requant(h @ W.astype(np.int64), b, m0, sh, *act)
    return h


def _rand_layer(rng, K, Nn, relu):
    W = rng.integers(-127, 128, (K, Nn))
    m0 = rng.integers(1 << 30, (1 << 31) - 1, Nn)
    sh = rng.integers(-12, -6, Nn)
    b = rng.integers(-2000, 2000, Nn)
    return (W, b, m0, sh, (0, 0, 127) if relu else (0, -128, 127))


def run_mlp(backend, x_q, layers):
    ex = _Mini(backend, _mlp_steps(x_q, layers))
    try:
        return ex.run_one(x_q), ex
    finally:
        ex.close()


def matmul16(backend, seed=2):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (16, 16))
    layers = [_rand_layer(rng, 16, 16, False)]
    try:
        y, ex = run_mlp(backend, x, layers)
    except TpuRunError as e:
        return _res(False, str(e), t0)
    ref = _ref(x, layers)
    nz = int((ref != 0).sum())
    return _res((y == ref).all(), f"16x16x16 bit-exact={bool((y == ref).all())} ({nz}/256 non-zero outputs)", t0)


def mlp(backend, batch=32, dims=(64, 128, 64, 16), seed=3):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (batch, dims[0]))
    layers = [_rand_layer(rng, dims[i], dims[i + 1], i < len(dims) - 2) for i in range(len(dims) - 1)]
    try:
        y, ex = run_mlp(backend, x, layers)
    except TpuRunError as e:
        return _res(False, str(e), t0)
    ref = _ref(x, layers)
    s = ex.prof.summary()
    return _res((y == ref).all(), f"MLP {'-'.join(map(str, dims))} batch {batch}: bit-exact={bool((y == ref).all())}, "
                f"{len(ex.programs)} program(s), kernel {s['device_kernel_ms_per_image'] * 1e3:.1f} us", t0)


def tpuv1_bundle(backend, bundle_dir):
    """tpuv1's own compiled bundle (tpu_mlp.TpuMlp.run_sample) through tpu.Device (board/mock)."""
    t0 = time.perf_counter()
    dev = getattr(backend, "dev", None)
    if dev is None:
        return _res(True, "skipped (needs a tpu.Device backend: board or mock)", t0)
    import tpu_mlp
    with tpu_mlp.TpuMlp(dev, bundle_dir) as net:
        res, exact = net.run_sample()
    return _res(exact, f"tpuv1 bundle {bundle_dir}: {res}", t0)
