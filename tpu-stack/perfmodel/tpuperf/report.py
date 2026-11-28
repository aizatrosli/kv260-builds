"""Tables, CSV output and the RECOMMENDATIONS.md generator."""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Sequence

from . import calibrate as K
from . import config as C
from . import dse
from . import resources as RS
from . import workloads as W
from .model import simulate


# ---------------------------------------------------------------------------------------------
def fmt_table(rows: List[dict], cols: Sequence[str] = None, floatfmt: str = "{:.2f}") -> str:
    if not rows:
        return "(empty)"
    cols = list(cols or rows[0].keys())
    cells = [[(floatfmt.format(r.get(c)) if isinstance(r.get(c), float) else str(r.get(c, "")))
              for c in cols] for r in rows]
    w = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w[i]) for i, c in enumerate(cols))
    sep = "  ".join("-" * w[i] for i in range(len(cols)))
    body = "\n".join("  ".join(x[i].ljust(w[i]) for i in range(len(cols))) for x in cells)
    return f"{line}\n{sep}\n{body}"


def md_table(rows: List[dict], cols: Sequence[str], headers: Sequence[str] = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


# ---------------------------------------------------------------------------------------------
TARGETS = ("jepa_tiny", "vit_tiny", "resnet18")


def ranked_steps(targets=TARGETS) -> List[dict]:
    """Rank roadmap steps by benefit / effort across the target workloads."""
    per: Dict[str, List[dict]] = {}
    for t in targets:
        for r in dse.roadmap(W.get(t)):
            per.setdefault(r["step"], []).append(r)
    base_res = RS.estimate(dse.jepa_baseline())
    out = []
    for s in dse.STEPS:
        rs = per.get(s.key)
        if not rs:
            continue
        alone = math.exp(sum(math.log(r["speedup_alone"]) for r in rs) / len(rs))
        loo = math.exp(sum(math.log(r["loss_if_dropped"]) for r in rs) / len(rs))
        benefit = max(alone, loo)
        res = RS.estimate(dse.jepa_baseline().with_(**s.changes),
                          W.jepa_tiny().param_bytes if s.key == "onchip" else 0)
        out.append(dict(step=s.key, change=s.label, effort=s.effort,
                        speedup_alone=round(alone, 2), loss_if_dropped=round(loo, 2),
                        score=round(math.log(benefit) / s.effort, 3),
                        n_workloads=len(rs),
                        d_lut=res.lut - base_res.lut, d_dsp=res.dsp - base_res.dsp,
                        d_bram=res.bram36 - base_res.bram36, d_uram=res.uram - base_res.uram,
                        what=s.what))
    out.sort(key=lambda r: -r["score"])
    return out


def zoo_table(cfgs: Sequence[C.HwConfig], names=None, batch=1) -> List[dict]:
    names = names or list(W.ZOO)
    rows = []
    for cfg in cfgs:
        for n in names:
            w = W.get(n, batch) if batch != 1 else W.get(n)
            c = cfg
            if c.weight_source == "onchip" and not RS.estimate(c, w.param_bytes).feasible:
                c = c.with_(weight_source="ddr_prefetch", ub_kib=1024,
                            name=c.name + "+ddr")     # weights do not fit: stream from DDR
            rows.append(dse.evaluate(w, c))
    return rows


def recommended() -> C.HwConfig:
    return C.proposed(32, 32, 200.0, vpu_pipelined=True)


def build_report(out_dir: str, md_path: str, quick: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cal = K.calibration_table()
    write_csv(os.path.join(out_dir, "calibration.csv"), cal)

    base = dse.jepa_baseline()
    rec = recommended()
    zoo = zoo_table([base, rec])
    write_csv(os.path.join(out_dir, "zoo.csv"), zoo)

    road = {}
    for t in TARGETS + ("deit_small",):
        road[t] = dse.roadmap(W.get(t))
        write_csv(os.path.join(out_dir, f"roadmap_{t}.csv"), road[t])
    ranks = ranked_steps()
    write_csv(os.path.join(out_dir, "ranked_steps.csv"), ranks)

    grid = dse.default_grid()
    if quick:
        grid.update(weight_bits=[8], control=["streamlined"], weight_source=["ddr_prefetch",
                                                                            "onchip"])
    sw = dse.sweep([W.jepa_tiny(), W.vit_tiny(), W.resnet18()] if not quick
                   else [W.jepa_tiny(), W.vit_tiny()], grid)
    write_csv(os.path.join(out_dir, "dse.csv"), sw)

    # array / clock summary (streamlined control, double buffering, best source per workload)
    arr_rows = []
    for (r, c) in grid["array"]:
        for f in grid["clock_mhz"]:
            for pk in (False, True):
                sel = [x for x in sw if x["rows"] == r and x["cols"] == c and x["clock_mhz"] == f
                       and x["packing"] == pk and x["control"] == "streamlined"
                       and x["buffering"] == "double" and x["weight_bits"] == 8]
                if not sel:
                    continue
                row = dict(array=f"{r}x{c}", clock_mhz=f, packing=pk)
                for wl in ("jepa_tiny", "vit_tiny"):
                    cands = [x for x in sel if x["workload"] == wl]
                    feas = [x for x in cands if not x["status"].startswith("INFEASIBLE")]
                    pick = min(feas or cands, key=lambda x: x["total_us"])
                    row[f"{wl}_us"] = pick["total_us"]
                    row[f"{wl}_src"] = pick["weight_source"]
                row.update(lut=sel[0]["lut"], dsp=sel[0]["dsp"], status=min(
                    (x["status"] for x in sel), key=lambda s: s.startswith("INFEASIBLE")))
                arr_rows.append(row)
    write_csv(os.path.join(out_dir, "array_clock.csv"), arr_rows)

    batch_rows = []
    for name in ("jepa_tiny", "vit_tiny", "deit_small"):
        for b in (1, 4, 16):
            w = W.get(name, b)
            for cfg in (base, rec.with_(weight_source="ddr_prefetch", ub_kib=1024)):
                e = dse.evaluate(w, cfg)
                batch_rows.append(dict(workload=name, batch=b, config=cfg.name,
                                       total_us=e["total_us"],
                                       ms_per_image=e["latency_ms_per_image"],
                                       images_per_s=e["images_per_s"], util=e["util"],
                                       weight_mb_moved=e["weight_mb_moved"]))
    write_csv(os.path.join(out_dir, "batch.csv"), batch_rows)

    int4 = []
    for name in W.ZOO:
        w = W.get(name)
        for bits in (8, 4):
            cfg = rec.with_(weight_bits=bits)
            res = RS.estimate(cfg, w.param_bytes)
            src = "onchip" if res.feasible else "ddr_prefetch"
            c2 = cfg if res.feasible else cfg.with_(weight_source="ddr_prefetch", ub_kib=1024)
            e = dse.evaluate(w, c2)
            int4.append(dict(workload=name, weight_bits=bits,
                             weights_mb=round(w.param_bytes * bits / 8 / 1e6, 2),
                             weights_on_chip=res.feasible, total_us=e["total_us"],
                             uram=RS.estimate(c2, w.param_bytes).uram))
    write_csv(os.path.join(out_dir, "int4.csv"), int4)

    md = render_md(cal, ranks, road, zoo, arr_rows, batch_rows, int4, base, rec)
    with open(md_path, "w") as fh:
        fh.write(md)
    return md


def render_md(cal, ranks, road, zoo, arr_rows, batch_rows, int4, base, rec) -> str:
    L = []
    a = L.append
    j_base = next(z for z in zoo if z["workload"] == "jepa_tiny" and z["config"] == base.name)
    j_rec = next(z for z in zoo if z["workload"] == "jepa_tiny" and z["config"].startswith(rec.name))
    rres = RS.estimate(rec, W.jepa_tiny().param_bytes)
    a("# KV260 TPU: hardware recommendations (from `tpuperf`)\n")
    a("Generated by `python -m tpuperf report`. Every number below comes from the "
      "cycle-approximate model in `tpuperf/`, calibrated against the tpuv1 board measurements "
      "and the jepa-tpu Verilator runs (table at the end). Treat speedups as +-15 %.\n")
    a("## Bottom line\n")
    a(f"* tpuv1 as built runs the jepa encoder in **{j_base['total_us'] / 1e3:.2f} ms** "
      f"(matmul {j_base['matmul_us'] / 1e3:.2f} ms + serial VPU {j_base['vpu_us'] / 1e3:.2f} ms, "
      f"{j_base['fps']:.0f} img/s).")
    a(f"* The recommended KV260 design ({rec.rows}x{rec.cols} array, {rec.clock_mhz:.0f} MHz, "
      f"DSP packing, double-buffered weights, weights in URAM, streamlined control, "
      f"pipelined/overlapped VPU) runs it in **{j_rec['total_us'] / 1e3:.3f} ms** "
      f"(**{j_base['total_us'] / j_rec['total_us']:.1f}x**, {j_rec['fps']:.0f} img/s) using "
      f"~{rres.lut / 1e3:.0f}k LUT, {rres.dsp} DSP, {rres.bram36} BRAM36, {rres.uram} URAM "
      f"({rres.status}).")
    a("* 64x64 and 32x64/64x32 arrays do not fit the XCK26 (DSP and LUT), even with packing. "
      "32x32 needs DSP packing to leave DSPs for the requant lanes and the VPU.")
    a("* The as-built design is control-bound, not MAC-bound: per output-column tile it spends "
      "2C+1 SET_CONFIGs (2 cycles each) that wait for the previous ACTIVATE to drain, plus a "
      "full SYNC(MATRIX) drain. That costs more than weight bandwidth does at batch >= 64 rows.\n")

    a("## Top hardware changes, ranked by speedup per unit of effort\n")
    a("Effort: 1 = compiler/config only, 2 = small RTL change, 3 = new unit, 4 = big RTL + "
      "timing closure. *Alone* = speedup of the change applied to tpuv1 by itself "
      "(geomean over jepa_tiny, ViT-Tiny, ResNet-18). *Loss if dropped* = how much slower "
      "the full recommended design gets without it (captures changes that only pay off "
      "together, e.g. double buffering only matters once the array is 32 wide). "
      "Resource deltas are for the change alone on tpuv1.\n")
    a(md_table(ranks[:8], ["step", "change", "effort", "speedup_alone", "loss_if_dropped",
                           "score", "d_lut", "d_dsp", "d_bram", "d_uram"],
               ["key", "change", "effort", "alone", "loss if dropped", "score",
                "dLUT", "dDSP", "dBRAM36", "dURAM"]))
    a("")
    a("### What each one is\n")
    for i, r in enumerate(ranks[:8], 1):
        a(f"{i}. **{r['change']}** (effort {r['effort']}). {r['what']}")
    a("")
    rk = {r["step"]: r for r in ranks}
    g = lambda k, f: rk[k][f] if k in rk else float("nan")          # noqa: E731
    res32 = RS.estimate(rec, W.jepa_tiny().param_bytes)
    a("### Suggested implementation order for the KV260\n")
    a("The score ranks raw value per effort; the order below also respects dependencies "
      "(cheap enablers first, then the array that needs them).\n")
    a(f"1. **Streamlined control** (effort {g('control', 'effort')}): "
      f"{g('control', 'speedup_alone'):.2f}x alone, {g('control', 'loss_if_dropped'):.2f}x "
      "lost if left out of the final design. Compiler + decoder/act_seq only, no resources, "
      "no timing risk. Its share grows as the array gets wider (2C+1 SET_CONFIGs per column "
      "tile).")
    a(f"2. **Double-buffered weight shadow** (+{g('double', 'd_lut')} LUT at 16x16): "
      f"{g('double', 'speedup_alone'):.2f}x on tpuv1 today (64-row tiles already cover the "
      f"49-cycle period) but {g('double', 'loss_if_dropped'):.2f}x lost without it in the "
      "32x32 design, where the single-shadow period is 2R+C+1 = 97 cycles > 64 tokens.")
    a(f"3. **32x32 array with DSP packing**: {g('array32', 'speedup_alone'):.2f}x alone, "
      f"{g('array32', 'loss_if_dropped'):.2f}x lost without it; +{g('array32', 'd_lut')} LUT, "
      f"+{g('array32', 'd_dsp')} DSP, +{g('array32', 'd_bram')} BRAM36. The full recommended "
      f"design lands at {res32.lut / 1e3:.0f}k LUT ({100 * res32.lut / 117120:.0f} %), "
      f"{res32.dsp} DSP ({100 * res32.dsp / 1248:.0f} %), {res32.bram36} BRAM36, "
      f"{res32.uram} URAM with jepa weights resident.")
    a(f"4. **200 MHz** (effort {g('clock200', 'effort')}): {g('clock200', 'speedup_alone'):.2f}x "
      "on everything; hardest item (tpuv1 has +1.25 ns slack at 100 MHz, i.e. ~114 MHz). "
      "150 MHz is a reasonable intermediate target (see the array x clock table).")
    a(f"5. **Integrated, pipelined, overlapped VPU**: {g('vpu_overlap', 'loss_if_dropped'):.2f}x "
      "lost without it. Once the array is ~8x faster the serial VPU (LayerNorm isqrt + divide, "
      "softmax divide: 52 / 30 scalar cycles per job) dominates transformer time; beyond "
      "overlap, a parallel divider per lane group is the next step.")
    a(f"6. **Weights on chip / weight prefetch / UB->weight path**: each is worth only "
      f"{g('onchip', 'speedup_alone'):.2f}x / {g('prefetch', 'speedup_alone'):.2f}x / "
      f"{g('ubw', 'speedup_alone'):.2f}x in time. With 64-row tiles one 256 B READ_WEIGHTS "
      "per tile already hides behind compute; the 58 % util of the jepa 'AXI' Verilator run "
      "came from the testbench's ~2.35 B/cycle push rate, not from DDR (an HP port sustains "
      "~12.8 B/cycle). Keep weights on chip for jepa-sized models because it removes DDR "
      "traffic and makes latency deterministic, not for speed.\n")

    for t, rows in road.items():
        a(f"### Cumulative roadmap: {t}\n")
        a(md_table(rows, ["step", "total_us", "speedup_cum", "marginal", "speedup_alone",
                          "loss_if_dropped", "lut", "dsp", "bram36", "uram", "status"]))
        a("")

    a("## Workloads on tpuv1 vs the recommended design (batch 1)\n")
    a("If the weights do not fit on chip the recommended design streams them from DDR with the "
      "prefetch engine (`+ddr`).\n")
    a(md_table(zoo, ["workload", "config", "gmacs", "param_mb", "total_us", "matmul_us",
                     "vpu_us", "util", "gops", "images_per_s", "status"]))
    a("")
    a("## Array size x clock (streamlined control, double buffering, INT8)\n")
    a("Best weight source per point; status is the resource check on XCK26 "
      f"(LUT/FF <= {int(RS.LUT_ROUTABLE * 100)} %, DSP/BRAM/URAM <= 100 %).\n")
    a("Subset (100 and 200 MHz); all four clocks are in `out/array_clock.csv`.\n")
    a(md_table([r for r in arr_rows if r["clock_mhz"] in (100, 200)],
               ["array", "clock_mhz", "packing", "jepa_tiny_us", "jepa_tiny_src",
                          "vit_tiny_us", "vit_tiny_src", "lut", "dsp", "status"]))
    a("")
    a("## Batch size (weights streamed from DDR)\n")
    a(md_table(batch_rows, ["workload", "batch", "config", "total_us", "ms_per_image",
                            "images_per_s", "util", "weight_mb_moved"]))
    a("")
    a("## INT8 vs INT4 weights on the recommended design\n")
    a("INT4 mostly buys capacity: models that fit on chip skip DDR entirely. With prefetch and "
      ">= 64 rows per tile the array is not weight-bandwidth bound, so INT4 saves little time "
      "for models that stream either way.\n")
    a(md_table(int4, ["workload", "weight_bits", "weights_mb", "weights_on_chip", "total_us",
                      "uram"]))
    a("")
    a("## Calibration\n")
    a(md_table(cal, ["benchmark", "source", "unit", "measured", "model", "error_pct"]))
    a("")
    a("Fitted: DMA issue-to-data latency 300 ns, host job overhead 4 us, AXI burst efficiency "
      "0.8 (board MLPs); testbench AXI weight push 2.35 B/cycle (jepa AXI stall count). "
      "Everything else (guard 2N, 2 cycles/instruction, 33 SET_CONFIG per column tile, 8-stage "
      "requant, VPU FSM cycles) is read from the RTL.\n")
    a("## Caveats\n")
    a("* Board measurements are wall time of one job; the model's fixed host overhead is a fit.")
    a("* VPU numbers assume the jepa-tpu vpu.sv FSM widened to C lanes; GELU fused into the "
      "activation LUT in the recommended design.")
    a("* Resource coefficients are linear fits around one measured build (16x16). Expect "
      "+-20 % at 32x32; the DSP count is exact arithmetic, the LUT count is not.")
    a("* No DDR contention from the PS, no refresh, no TLB/IOMMU effects.")
    a("* CNN stems use host-side im2col; depthwise convs map one channel per column "
      "(1/R useful lanes), which is why MobileNetV2 utilisation is low.")
    return "\n".join(L) + "\n"
