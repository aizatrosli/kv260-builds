"""Benchmark CLI: the numbers to collect on day one on the board.

    python -m tpurt.bench --backend board                        # everything, default sweep
    python -m tpurt.bench --backend board --batch 1,8,16 --vector host --images 64 --json day1.json
    python -m tpurt.bench --backend sim --images 4               # off-board dry run (estimated timing)
    python -m tpurt.bench --host-only                            # host-op cost only (LN/softmax/GELU/...)
    python -m tpurt.bench --backend null --skip overhead,dma,matmul --check 0   # host-side cost per image

Sections
  1. job overhead     round trip of a 3-instruction program (ioctl + validate + push + IRQ)
  2. DMA bandwidth    1 MiB DDR->UB and UB->DDR inside one program
  3. matmul           a 512x512 (fc1-sized x8) GEMM: MAC/s, array utilisation, weight stalls
  4. host ops         per-image cost of LayerNorm / softmax / GELU / residual / K,V packing (C, numpy)
  5. JEPA encoder     sweep batch x vector-mode x pipeline: ms/img, img/s, host vs device split,
                      per-op table, hardware counters; bit-exactness vs quantize.int_forward
"""
import argparse
import json
import platform
import sys
import time

import numpy as np

from . import get_backend, BackendUnavailable, data, isa, selftest
from .buffers import DmaPool
from .hostops import get_ops, NumpyOps
from .jepa import JepaExecutor
from .profiler import Profiler


def _timeit(fn, iters, warm=2):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


def job_overhead(be, iters=200):
    pool = DmaPool(be, arena_size=1 << 16)
    prog = isa.to_bytes([isa.sync(isa.U_ALL), isa.interrupt_host(1), isa.halt()])
    job = be.prepare(prog, [])
    rt = _timeit(lambda: be.run_job(job), iters)
    k = np.mean([be.run_job(job).wall_ns for _ in range(20)]) / 1e3
    pool.close()
    return dict(roundtrip_us=rt * 1e6, kernel_us=float(k), overhead_us=rt * 1e6 - float(k))


def dma_bandwidth(be, nbytes=1 << 20, iters=5):
    pool = DmaPool(be, arena_size=4 << 20)
    src, dst = pool.alloc(nbytes), pool.alloc(nbytes)
    src.write(np.random.default_rng(0).integers(0, 256, nbytes, dtype=np.uint8))
    rows = nbytes // 16

    def prog(read, write):
        p, rel = [], []
        for r in range(0, rows, isa.MAX_DMA_ROWS):
            n = min(isa.MAX_DMA_ROWS, rows - r)
            if read:
                rel.append((len(p), *src.reloc(r * 16)))
                p.append(isa.read_host_mem(r % 65536, 0, n))
        p.append(isa.sync(isa.U_ALL))
        for r in range(0, rows, isa.MAX_DMA_ROWS):
            n = min(isa.MAX_DMA_ROWS, rows - r)
            if write:
                rel.append((len(p), *dst.reloc(r * 16)))
                p.append(isa.write_host_mem(r % 65536, 0, n))
        p += [isa.sync(isa.U_ALL), isa.interrupt_host(2), isa.halt()]
        return be.prepare(isa.to_bytes(p), rel)
    rd, rw = prog(True, False), prog(True, True)
    t_rd = min(be.run_job(rd).wall_ns for _ in range(iters)) / 1e9
    t_rw = min(be.run_job(rw).wall_ns for _ in range(iters)) / 1e9
    ok = (dst.read() == src.read()).all()
    pool.close()
    return dict(bytes=nbytes, ddr_to_ub_MBps=nbytes / t_rd / 1e6,
                ub_to_ddr_MBps=nbytes / max(t_rw - t_rd, 1e-12) / 1e6, loopback_ok=bool(ok))


def matmul_bench(be, M=512, K=512, N=512):
    rng = np.random.default_rng(4)
    x = rng.integers(-128, 128, (M, K))
    layers = [selftest._rand_layer(rng, K, N, False)]
    t = time.perf_counter()
    steps = selftest._mlp_steps(x, layers)
    ex = selftest._Mini(be, steps)
    y = ex.run_one(x)
    ok = (y == selftest._ref(x, layers)).all()
    s = ex.prof.summary()
    perf = s["perf_per_image"]
    kern_s = s["device_kernel_ms_per_image"] / 1e3
    ex.close()
    return dict(shape=f"{M}x{K}x{N}", programs=len(ex.programs), kernel_ms=kern_s * 1e3,
                GMACs=M * K * N / kern_s / 1e9 if kern_s else 0, bit_exact=bool(ok),
                array_util=perf["array_active"] / perf["total"] if perf["total"] else None,
                w_stall_frac=perf["w_stall"] / perf["total"] if perf["total"] else None,
                wall_s=time.perf_counter() - t)


def hostops_bench(kind, B=1, iters=50, P=None):
    """Per-image ms of each host op as the executor calls them (6 layers)."""
    P = P if P is not None else data.load_qmodel()
    ops = get_ops(kind) if kind != "numpy" else NumpyOps()
    rng = np.random.default_rng(0)
    M = 64 * B
    i8 = lambda *s: rng.integers(-128, 128, s).astype(np.int8)
    x, a, h1, s, qkv = i8(8, M, 16), i8(8, M, 16), i8(32, M, 16), i8(B * 4, 4, 64, 16), i8(24, M, 16)
    o8, o32, os_ = np.empty_like(x), np.empty_like(h1), np.empty_like(s)
    kT, vw = np.empty((B * 4, 4, 2, 16, 16), np.int8), np.empty((B * 4, 2, 4, 16, 16), np.int8)
    ma, mb = P["adda0"]
    per = dict(layernorm=13, softmax=6, gelu=6, residual=13, pack_kv=6)
    fns = dict(layernorm=lambda: ops.layernorm(x, out=o8), softmax=lambda: ops.softmax(s, P["exp0"], out=os_),
               gelu=lambda: ops.lut8(h1, P["gelu0"], out=o32), residual=lambda: ops.ew_add(x, a, ma, mb, out=o8),
               pack_kv=lambda: ops.pack_kv(qkv, B, 4, kT, vw))
    r = {k: _timeit(f, iters) * 1e3 * per[k] / B for k, f in fns.items()}
    r["total_ln_sm_gelu_res"] = r["layernorm"] + r["softmax"] + r["gelu"] + r["residual"]
    r["total"] = sum(v for k, v in r.items() if k != "total_ln_sm_gelu_res")
    return r


def jepa_bench(be, P, xq, batch, vector, pipeline, ops, rw_group, check):
    prof = Profiler()
    ex = JepaExecutor(be, P, batch=batch, vector=vector, ops=ops, profiler=prof, rw_group=rw_group)
    ex.forward_q(xq[:batch])                       # warm-up (first-touch, caches)
    prof.reset()
    y = ex.run_many(xq, pipeline=pipeline)
    exact = None
    if check:
        exact = all((y[i] == data.golden_forward(P, xq[i])).all() for i in range(min(check, len(xq))))
    s = prof.summary()
    desc = ex.describe()
    rep = prof.report(f"--- JEPA batch={batch} vector={vector} pipeline={pipeline} ops={ex.ops.name} "
                      f"rw_group={rw_group}", estimated=be.estimated_timing)
    ex.close()
    return dict(batch=batch, vector=vector, pipeline=pipeline, ops=ex.ops.name, rw_group=rw_group, bit_exact=exact,
                programs_per_batch=len(ex.programs), **{k: v for k, v in s.items() if k != "perf_per_image"},
                perf_per_image=s["perf_per_image"]), rep, desc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tpurt.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", help="sim | mock | board | raw | null | auto (default)")
    ap.add_argument("--images", type=int, default=None, help="images per config (default 32 board, 4 sim)")
    ap.add_argument("--batch", default=None, help="comma list (default 1,4,16 board; 1,4 sim)")
    ap.add_argument("--vector", default="host", help="host | vpu | both")
    ap.add_argument("--pipeline", default="both", help="on | off | both")
    ap.add_argument("--ops", default="auto", help="host kernels: auto | c | numpy")
    ap.add_argument("--rw-group", action="store_true", help="one READ_WEIGHTS per output tile (experiment)")
    ap.add_argument("--check", type=int, default=2, help="verify the first N images bit-exact (0 = off)")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: overhead,dma,matmul,host,jepa")
    ap.add_argument("--host-only", action="store_true", help="only the host-op section (no device)")
    ap.add_argument("--json", help="write all results to this file")
    a = ap.parse_args(argv)
    out = dict(machine=platform.machine(), python=platform.python_version(), numpy=np.__version__,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    skip = set(filter(None, a.skip.split(",")))
    if a.host_only:
        skip |= {"overhead", "dma", "matmul", "jepa"}
    be = None
    if skip != {"overhead", "dma", "matmul", "host", "jepa"} and not a.host_only:
        try:
            be = get_backend(a.backend)
        except BackendUnavailable as e:
            print(f"BENCH: {e}", file=sys.stderr)
            return 2
        out["backend"] = be.info()
        print(f"backend: {be.info()}")
        if be.estimated_timing:
            print("NOTE: device times are ESTIMATES from the simulator's cycle model, not measurements")
    on_board = be is not None and be.name in ("board", "raw-uio", "raw-pynq")
    P = data.load_qmodel()

    if be and "overhead" not in skip:
        r = out["job_overhead"] = job_overhead(be, 200 if on_board else 20)
        print(f"[1] job overhead: round trip {r['roundtrip_us']:.1f} us (kernel {r['kernel_us']:.1f}, "
              f"ioctl/driver {r['overhead_us']:.1f})")
    if be and "dma" not in skip:
        r = out["dma"] = dma_bandwidth(be, (1 << 20) if on_board else (1 << 16))
        print(f"[2] DMA: DDR->UB {r['ddr_to_ub_MBps']:.0f} MB/s, UB->DDR {r['ub_to_ddr_MBps']:.0f} MB/s, "
              f"loopback {'ok' if r['loopback_ok'] else 'MISMATCH'}")
    if be and "matmul" not in skip:
        r = out["matmul"] = matmul_bench(be, *((512, 512, 512) if on_board else (256, 128, 256)))
        print(f"[3] matmul {r['shape']}: {r['kernel_ms']:.3f} ms, {r['GMACs']:.2f} GMAC/s, util {r['array_util']}, "
              f"w_stall {r['w_stall_frac']}, bit-exact {r['bit_exact']}")
    if "host" not in skip:
        out["host_ops"] = {}
        for kind in ("c", "numpy"):
            try:
                r = hostops_bench(kind, 1, 50, P)
            except RuntimeError as e:
                print(f"[4] host ops ({kind}): unavailable: {e}")
                continue
            out["host_ops"][kind] = r
            print(f"[4] host ops ({kind}, ms/img, batch 1): " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    if be and "jepa" not in skip:
        n = a.images or (32 if on_board else 4)
        imgs, src = data.load_images(n)
        xq = data.quant_img(P, imgs)
        batches = [int(b) for b in (a.batch or ("1,4,16" if on_board else "1,4")).split(",")]
        vecs = ["host", "vpu"] if a.vector == "both" else [a.vector]
        pipes = {"on": [True], "off": [False], "both": [False, True]}[a.pipeline]
        out["jepa"], rows = [], []
        print(f"[5] JEPA encoder, {n} images ({src})")
        for vec in vecs:
            if vec == "vpu" and not be.has_vpu:
                print("    vector=vpu skipped: backend has no VPU")
                continue
            for b in batches:
                for pipe in pipes:
                    try:
                        r, rep, desc = jepa_bench(be, P, xq, b, vec, pipe, a.ops, a.rw_group, a.check)
                    except Exception as e:          # keep sweeping; record the failure
                        print(f"    batch={b} vector={vec} pipeline={pipe}: FAILED {type(e).__name__}: {e}")
                        out["jepa"].append(dict(batch=b, vector=vec, pipeline=pipe, error=str(e)))
                        continue
                    print(desc)
                    print(rep)
                    out["jepa"].append(r)
                    rows.append(r)
        if rows:
            print("\nsummary (ms/img; device = kernel-measured RUN->IRQ)" +
                  (" [ESTIMATED]" if be.estimated_timing else ""))
            print(f"{'vector':6s} {'batch':>5s} {'pipe':>5s} {'ms/img':>8s} {'img/s':>7s} {'host':>7s} {'device':>7s} "
                  f"{'ovh':>7s} {'runs':>5s} {'util':>5s} {'exact':>5s}")
            for r in rows:
                p = r["perf_per_image"]
                util = f"{p['array_active'] / p['total']:.2f}" if p["total"] else "-"
                print(f"{r['vector']:6s} {r['batch']:5d} {str(r['pipeline']):>5s} {r['ms_per_image']:8.3f} "
                      f"{r['img_per_s']:7.1f} {r['host_ms_per_image']:7.3f} {r['device_kernel_ms_per_image']:7.3f} "
                      f"{r['driver_overhead_ms_per_image']:7.3f} {r['runs_per_image']:5.1f} {util:>5s} "
                      f"{str(r['bit_exact']):>5s}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"wrote {a.json}")
    if be:
        be.close()
    bad = [r for r in out.get("jepa", []) if r.get("error") or r.get("bit_exact") is False]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
