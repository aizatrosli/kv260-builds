#!/usr/bin/env python3
"""KV260 board smoke test: escalating stages, one PASS/FAIL line each, stops at the first failure.

    sudo python3 board_smoke.py                          # /dev/tpu0 (kernel driver)
    sudo python3 board_smoke.py --backend raw            # UIO + u-dma-buf (or PYNQ) without the driver
    python3 board_smoke.py --backend sim                 # dry run anywhere (also run by the tests)

  1 driver    /dev/tpu0 present and usable (PL loaded, tpu_kv260 bound), or the raw path's nodes
  2 regs      GET_INFO / register reads: VERSION (0x4xx/0x5xx), BUILD_ID (--build-id), state idle
  3 dma       DDR -> UB -> DDR loopback, 1D (5000 rows, split DMAs) and 2D strided, guard bytes
  4 matmul    one 16x16x16 MATMUL + ACTIVATE requant, bit-exact vs numpy
  5 mlp       3-layer int8 MLP compiled by tpurt (+ tpuv1's own bundles with --bundles DIR)
  6 jepa      full I-JEPA encoder on N images, bit-exact vs jepa-tpu quantize.int_forward, ms/img

Exit: 0 all PASS, 1 a stage FAILED, 2 backend unavailable.  Final line BOARD_SMOKE_PASS / _FAIL.
Needs on the board: this runtime dir, tpuv1/software/runtime (tpu.py, tpu_isa.py), jepa-tpu/{intops.py,
qmodel.npy}.  Paths: TPURT_TPUV1 / TPURT_JEPA if they are not siblings of kv260-builds.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="board", help="board (default) | raw | mock | sim")
    ap.add_argument("--build-id", help="expected BUILD_ID (hex), catches stale bitstreams")
    ap.add_argument("--bundles", help="tpuv1 bundle dir (build/bundles) to also run tpu_mlp samples")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--vector", default="host", help="host | vpu (needs a VPU bitstream)")
    ap.add_argument("--keep-going", action="store_true", help="run later stages after a failure")
    ap.add_argument("--stages", default="driver,regs,dma,matmul,mlp,jepa")
    a = ap.parse_args(argv)
    want = a.stages.split(",")
    fails, state = [], {}

    def stage(name, fn):
        if name not in want:
            return True
        if fails and not a.keep_going:
            print(f"[SKIP] {name}")
            return False
        t = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:                           # report, never crash the sequence
            ok, detail = False, f"{type(e).__name__}: {e}"
            if os.environ.get("TPURT_DEBUG"):
                traceback.print_exc()
        print(f"[{'PASS' if ok else 'FAIL'}] {name:7s} {detail}  ({(time.perf_counter() - t) * 1e3:.0f} ms)", flush=True)
        if not ok:
            fails.append(name)
        return ok

    from tpurt import get_backend, BackendUnavailable, selftest, data

    def s_driver():
        if a.backend == "board":
            from tpurt.backends.tpudev import driver_diagnosis
            why = driver_diagnosis()
            if why:
                return False, why
        try:
            state["be"] = get_backend(a.backend)
        except BackendUnavailable as e:
            state["unavailable"] = True
            return False, str(e)
        extra = ""
        for p in ("/sys/bus/platform/drivers/tpu_kv260/a0000000.fabric/version",):
            if os.path.exists(p):
                extra = f", sysfs version={Path(p).read_text().strip()}"
        return True, f"backend={state['be'].name}{extra}"

    def s_regs():
        be = state["be"]
        i = be.info()
        hv, bid = i.get("hw_version", 0), i.get("build_id", 0)
        ok = (hv >> 8) in (4, 5, 6) and i.get("state", "idle") == "idle"
        msg = f"VERSION=0x{hv:08x} BUILD_ID=0x{bid:08x} state={i.get('state', 'idle')} irq={i.get('irq', '-')}"
        if a.build_id:
            ok &= bid == int(a.build_id, 16)
            msg += f" (expected 0x{int(a.build_id, 16):08x})"
        if hasattr(be, "regs"):                          # raw path: SCRATCH read/write lanes (L1)
            for pat in (0xA5A5A5A5, 0x5A5A5A5A, 0x12345678):
                be.regs.wr(0x4, pat)
                ok &= be.regs.rd(0x4) == pat
        return ok, msg

    def s_dma():
        r = selftest.dma_loopback(state["be"])
        return r["ok"], r["detail"]

    def s_matmul():
        r = selftest.matmul16(state["be"])
        return r["ok"], r["detail"]

    def s_mlp():
        r = selftest.mlp(state["be"])
        ok, detail = r["ok"], r["detail"]
        if a.bundles:
            for name in ("fc1", "mlp3", "mlp_b1", "mlp_wide"):
                d = Path(a.bundles) / name
                if (d / "model.json").exists():
                    rb = selftest.tpuv1_bundle(state["be"], d)
                    ok &= rb["ok"]
                    detail += f"; {name} {'ok' if rb['ok'] else 'FAIL'}"
        return ok, detail

    def s_jepa():
        from tpurt.jepa import JepaExecutor
        be = state["be"]
        P = data.load_qmodel()
        imgs, src = data.load_images(a.images)
        xq = data.quant_img(P, imgs)
        ex = JepaExecutor(be, P, batch=a.batch, vector=a.vector)
        ex.forward_q(xq[:a.batch])                      # warm-up
        ex.prof.reset()
        y = ex.run_many(xq)
        bad = [i for i in range(len(xq)) if not (y[i] == data.golden_forward(P, xq[i])).all()]
        s = ex.prof.summary()
        print(ex.describe())
        print(ex.prof.report(estimated=be.estimated_timing))
        ex.close()
        est = " [device time ESTIMATED]" if be.estimated_timing else ""
        return not bad, (f"{len(xq) - len(bad)}/{len(xq)} images bit-exact ({src}); {s['ms_per_image']:.2f} ms/img "
                         f"(device {s['device_kernel_ms_per_image']:.2f}, host {s['host_ms_per_image']:.2f}, "
                         f"driver overhead {s['driver_overhead_ms_per_image']:.2f}){est}")

    stage("driver", s_driver)
    if "be" not in state:
        print("BOARD_SMOKE_FAIL (backend unavailable)")
        return 2
    print(f"backend info: {state['be'].info()}")
    stage("regs", s_regs)
    stage("dma", s_dma)
    stage("matmul", s_matmul)
    stage("mlp", s_mlp)
    stage("jepa", s_jepa)
    state["be"].close()
    print("BOARD_SMOKE_" + ("FAIL: " + ",".join(fails) if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
