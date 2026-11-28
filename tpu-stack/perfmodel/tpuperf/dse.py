"""Design-space exploration: sweeps, one-at-a-time sensitivity and an upgrade roadmap."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import config as C
from . import resources as RS
from . import workloads as W
from .model import simulate, Result

# Control-path fixes that need no new datapath (compiler + small decoder/act_seq changes)
STREAMLINED = dict(cfg_mode="param_rom", act_blocks_cfg=False, sync_per_tile=False,
                   gelu_in_act=True)


def with_vpu_lanes(cfg: C.HwConfig) -> C.HwConfig:
    return cfg.with_(vpu_lanes=cfg.cols)


def onchip_store_bytes(cfg: C.HwConfig, w: W.Workload) -> int:
    return w.param_bytes if cfg.weight_source == "onchip" else 0


def evaluate(w: W.Workload, cfg: C.HwConfig) -> dict:
    """Simulate + resources + feasibility for one (workload, config)."""
    r = simulate(w, cfg)
    res = RS.estimate(cfg, onchip_store_bytes(cfg, w))
    row = r.row()
    row.update(batch=w.batch, rows=cfg.rows, cols=cfg.cols, packing=cfg.dsp_packing,
               buffering=cfg.weight_buffering, weight_source=cfg.weight_source,
               weight_bits=cfg.weight_bits, ub_kib=cfg.ub_kib, cfg_mode=cfg.cfg_mode,
               latency_ms_per_image=round(r.total_us / 1e3 / w.batch, 3),
               images_per_s=round(1e6 * w.batch / r.total_us, 1) if r.total_us else 0,
               param_mb=round(w.param_bytes / 1e6, 2), clock_risk=RS.clock_risk(cfg))
    row.update(res.row())
    return row


# ---------------------------------------------------------------------------------------------
def default_grid() -> Dict[str, Sequence]:
    return dict(
        array=[(16, 16), (16, 32), (32, 16), (32, 32), (32, 64), (64, 32), (64, 64)],
        clock_mhz=[100, 150, 200, 250],
        dsp_packing=[False, True],
        weight_buffering=["shadow", "double"],
        weight_source=["ddr", "ddr_prefetch", "onchip"],
        weight_bits=[8, 4],
        control=["tpuv1", "streamlined"],
    )


def config_from(point: dict, base: Optional[C.HwConfig] = None) -> C.HwConfig:
    base = base or C.tpuv1_board().with_(program_upload=False)
    kw = dict(point)
    r, c = kw.pop("array", (base.rows, base.cols))
    control = kw.pop("control", "tpuv1")
    if control == "streamlined":
        kw.update(STREAMLINED)
        kw.setdefault("ub_weight_path", True)
    src = kw.get("weight_source", base.weight_source)
    if src == "onchip":
        kw.setdefault("ub_kib", 256)           # leave URAM for the weights
    if r * c > 256 or kw.get("clock_mhz", 100) > 100:
        kw.setdefault("hp_ports", 2)
    cfg = base.with_(rows=r, cols=c, vpu_lanes=c, with_vpu=True, **kw)
    name = (f"{r}x{c}@{int(cfg.clock_mhz)}{'p' if cfg.dsp_packing else ''}"
            f"-{cfg.weight_buffering[0]}-{cfg.weight_source}-i{cfg.weight_bits}-{control[:4]}")
    return cfg.with_(name=name)


def sweep(workloads: Iterable[W.Workload], grid: Optional[Dict[str, Sequence]] = None,
          base: Optional[C.HwConfig] = None) -> List[dict]:
    grid = grid or default_grid()
    keys = list(grid)
    rows = []
    wl = list(workloads)
    for values in itertools.product(*(grid[k] for k in keys)):
        point = dict(zip(keys, values))
        cfg = config_from(point, base)
        for w in wl:
            row = evaluate(w, cfg)
            row["control"] = point.get("control", "tpuv1")
            rows.append(row)
    return rows


def pareto(rows: List[dict], workload: str, batch: int = 1) -> List[dict]:
    """Feasible configs, fastest first, dropping those dominated on (time, LUT)."""
    cand = [r for r in rows if r["workload"] == workload and r["batch"] == batch
            and not r["status"].startswith("INFEASIBLE")]
    cand.sort(key=lambda r: (r["total_us"], r["lut"]))
    out, best_lut = [], float("inf")
    for r in cand:
        if r["lut"] < best_lut:
            out.append(r)
            best_lut = r["lut"]
    return out


# ---------------------------------------------------------------------------------------------
# Upgrade roadmap: each step = (label, config changes, effort 1..5, resource note)
# ---------------------------------------------------------------------------------------------
@dataclass
class Step:
    key: str
    label: str
    changes: dict
    effort: int            # 1 = compiler only ... 5 = major RTL + timing closure
    what: str


STEPS: List[Step] = [
    Step("control", "Streamlined control (requant-param table, no per-tile SYNC, ACT overlaps)",
         dict(STREAMLINED), 2,
         "Per-layer requant table indexed by column tile (1 instr instead of 2C+1 SET_CONFIG), "
         "SET_CONFIG no longer waits for ACTIVATE, ACC ping-pong without SYNC(MATRIX) per tile, "
         "GELU as an activation-unit LUT. Decoder/act_seq + compiler change, ~0 resources."),
    Step("ubw", "UB -> weight-loader path for Q.K^T / P.V",
         dict(ub_weight_path=True), 2,
         "Read K/V tiles from the UB into the weight shift chain (with transpose for K^T) "
         "instead of a DDR round trip. One mux + address generator."),
    Step("prefetch", "Decoupled weight prefetch DMA (large bursts into the Weight FIFO)",
         dict(weight_source="ddr_prefetch"), 2,
         "A weight-stream engine that runs ahead of MATMUL bounded by FIFO space, one DMA per "
         "layer instead of one 256 B READ_WEIGHTS per tile. Hides DDR latency."),
    Step("onchip", "Weights resident on chip (URAM ROM)",
         dict(weight_source="onchip", ub_kib=256), 1,
         "Only for models <= ~1.5 MB INT8: preload weights into URAM once (UB cut to 256 KiB)."),
    Step("double", "Double-buffered weight shadow",
         dict(weight_buffering="double"), 2,
         "Second shadow register bank per PE, loaded alternately: the next tile loads while "
         "the previous one waits for its swap. Min tile period (2R+C+1)/2 instead of 2R+C+1 "
         "(49 vs 97 cycles at 32x32), i.e. 64-token tiles stop stalling."),
    Step("clock200", "200 MHz clock",
         dict(clock_mhz=200.0, hp_ports=2), 4,
         "Pipeline DSPs (A/B/M/P regs), register the requant/VPU multipliers, retime decoder "
         "legality checks. DDR latency in cycles doubles."),
    Step("array32", "32x32 array with DSP packing (2 INT8 MAC / DSP48E2)",
         dict(rows=32, cols=32, dsp_packing=True, vpu_lanes=32, hp_ports=2), 4,
         "4x MACs for ~2x DSPs; UB/ACC/WFIFO widths double. Needs packing correction logic."),
    Step("vpu_overlap", "VPU integrated, pipelined and overlapped with the array",
         dict(vpu_overlap=0.8, vpu_pipelined=True), 3,
         "VECTOR opcode with its own UB ports so LayerNorm/softmax/residual run while the "
         "array computes the next GEMM; isqrt/divide of one job overlap the next job's rows."),
]


def jepa_baseline() -> C.HwConfig:
    """tpuv1 as built running the jepa encoder from DDR: VPU attached but serial."""
    return C.tpuv1_board().with_(name="tpuv1 (baseline)", program_upload=False, with_vpu=True)


def roadmap(w: W.Workload, base: Optional[C.HwConfig] = None,
            steps: Optional[List[Step]] = None) -> List[dict]:
    """Apply steps cumulatively (skipping 'onchip' when weights do not fit) and also measure
    each step alone on the baseline."""
    base = base or jepa_baseline()
    steps = steps or STEPS
    r0 = simulate(w, base)
    t0 = r0.total_us
    out = [dict(step="baseline", label="tpuv1 as built", effort=0, total_us=round(t0, 1),
                speedup_cum=1.0, speedup_alone=1.0, marginal=1.0,
                **{k: v for k, v in RS.estimate(base).row().items()})]
    cfg = base
    t_prev = t0
    for s in steps:
        if s.key == "onchip":
            trial = cfg.with_(**s.changes)
            res = RS.estimate(trial, w.param_bytes)
            if not res.feasible:
                continue
        cfg = cfg.with_(**s.changes)
        t = simulate(w, cfg).total_us
        alone_cfg = base.with_(**s.changes)
        t_alone = simulate(w, alone_cfg).total_us
        res = RS.estimate(cfg, onchip_store_bytes(cfg, w))
        out.append(dict(step=s.key, label=s.label, effort=s.effort, total_us=round(t, 1),
                        speedup_cum=round(t0 / t, 2), speedup_alone=round(t0 / t_alone, 2),
                        marginal=round(t_prev / t, 2), **res.row()))
        t_prev = t
    # leave-one-out: how much slower the final design gets without each step
    final = cfg
    t_final = simulate(w, final).total_us
    applied = [s for s in steps if s.key in {o["step"] for o in out}]
    for o in out[1:]:
        cfg_wo = base
        for s in applied:
            if s.key != o["step"]:
                cfg_wo = cfg_wo.with_(**s.changes)
        o["loss_if_dropped"] = round(simulate(w, cfg_wo).total_us / t_final, 2)
    out[0]["loss_if_dropped"] = 1.0
    return out


def batch_scan(w_fn, cfg: C.HwConfig, batches=(1, 2, 4, 8, 16)) -> List[dict]:
    rows = []
    for b in batches:
        w = w_fn(batch=b)
        rows.append(evaluate(w, cfg))
    return rows
