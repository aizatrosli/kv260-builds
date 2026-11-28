"""Command line: python -m tpuperf <command> [options]

  calibrate                 model vs measured tpuv1 / jepa-tpu numbers
  run WORKLOAD              one workload on a preset (+ --set key=value overrides)
  zoo                       every workload on tpuv1 and on the recommended design
  resources                 resource estimate of a preset / overrides
  dse                       design-space sweep (CSV + Pareto table per workload)
  roadmap                   cumulative upgrade steps for one workload
  report                    all of the above -> CSVs + RECOMMENDATIONS.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import fields

from . import calibrate as K
from . import config as C
from . import dse
from . import report as RP
from . import resources as RS
from . import workloads as W
from .model import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PRESETS = {
    "tpuv1_board": C.tpuv1_board, "tpuv1_sim_rom": C.tpuv1_sim_rom,
    "tpuv1_sim_axi": C.tpuv1_sim_axi, "baseline": dse.jepa_baseline,
    "proposed": C.proposed, "recommended": RP.recommended,
}


def _coerce(cfg: C.HwConfig, key: str, val: str):
    types = {f.name: f.type for f in fields(cfg)}
    if key not in types:
        raise SystemExit(f"unknown config field {key!r}; one of: {', '.join(types)}")
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    return val


def make_cfg(args) -> C.HwConfig:
    cfg = PRESETS[args.preset]()
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cfg = cfg.with_(**{k: _coerce(cfg, k, v)})
    return cfg


def _add_cfg_args(p):
    p.add_argument("--preset", default="tpuv1_board", choices=sorted(PRESETS))
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a HwConfig field, e.g. --set rows=32 --set clock_mhz=200")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tpuperf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "out"), help="CSV output directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("calibrate")

    p = sub.add_parser("run")
    p.add_argument("workload", choices=sorted(W.ZOO))
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--layers", action="store_true", help="per-layer-type breakdown")
    _add_cfg_args(p)

    p = sub.add_parser("zoo")
    p.add_argument("--batch", type=int, default=1)

    p = sub.add_parser("resources")
    p.add_argument("--weights-kib", type=float, default=0.0, help="on-chip weight store")
    _add_cfg_args(p)

    p = sub.add_parser("dse")
    p.add_argument("--workloads", default="jepa_tiny,vit_tiny,resnet18")
    p.add_argument("--quick", action="store_true", help="smaller grid")
    p.add_argument("--top", type=int, default=8)

    p = sub.add_parser("roadmap")
    p.add_argument("--workload", default="jepa_tiny", choices=sorted(W.ZOO))

    p = sub.add_parser("report")
    p.add_argument("--md", default=os.path.join(ROOT, "RECOMMENDATIONS.md"))
    p.add_argument("--quick", action="store_true")

    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if a.cmd == "calibrate":
        rows = K.calibration_table()
        print(RP.fmt_table(rows, ["benchmark", "source", "unit", "measured", "model",
                                  "error_pct"]))
        worst = max(abs(r["error_pct"]) for r in rows)
        print(f"\nworst |error| = {worst:.1f} %")
        RP.write_csv(os.path.join(a.out, "calibration.csv"), rows)

    elif a.cmd == "run":
        cfg = make_cfg(a)
        w = W.get(a.workload, a.batch)
        row = dse.evaluate(w, cfg)
        print(RP.fmt_table([row], ["workload", "batch", "config", "total_us", "matmul_us",
                                   "vpu_us", "io_us", "util", "gops", "images_per_s",
                                   "weight_stall_cycles", "status"]))
        if a.layers:
            r = simulate(w, cfg)
            lay = [dict(layer=k, us=round(v / cfg.clock_mhz, 1)) for k, v in r.per_layer.items()]
            print()
            print(RP.fmt_table(lay))
        RP.write_csv(os.path.join(a.out, f"run_{a.workload}.csv"), [row])

    elif a.cmd == "zoo":
        rows = RP.zoo_table([dse.jepa_baseline(), RP.recommended()], batch=a.batch)
        print(RP.fmt_table(rows, ["workload", "config", "gmacs", "param_mb", "total_us",
                                  "util", "gops", "images_per_s", "status"]))
        RP.write_csv(os.path.join(a.out, "zoo.csv"), rows)

    elif a.cmd == "resources":
        cfg = make_cfg(a)
        res = RS.estimate(cfg, int(a.weights_kib * 1024))
        print(RP.fmt_table([dict(config=cfg.name, **res.row(), clock_risk=RS.clock_risk(cfg))]))

    elif a.cmd == "dse":
        names = a.workloads.split(",")
        grid = dse.default_grid()
        if a.quick:
            grid.update(weight_bits=[8], control=["streamlined"],
                        weight_source=["ddr_prefetch", "onchip"])
        t = time.time()
        rows = dse.sweep([W.get(n) for n in names], grid)
        path = os.path.join(a.out, "dse.csv")
        RP.write_csv(path, rows)
        n_inf = sum(r["status"].startswith("INFEASIBLE") for r in rows)
        print(f"{len(rows)} points ({n_inf} infeasible on XCK26) in {time.time() - t:.1f} s "
              f"-> {path}")
        for n in names:
            print(f"\nPareto (time vs LUT), {n}:")
            print(RP.fmt_table(dse.pareto(rows, n)[:a.top],
                               ["config", "total_us", "util", "gops", "lut", "dsp", "bram36",
                                "uram", "status"]))

    elif a.cmd == "roadmap":
        rows = dse.roadmap(W.get(a.workload))
        print(RP.fmt_table(rows, ["step", "effort", "total_us", "speedup_cum", "marginal",
                                  "speedup_alone", "loss_if_dropped", "lut", "dsp", "bram36",
                                  "uram", "status"]))
        RP.write_csv(os.path.join(a.out, f"roadmap_{a.workload}.csv"), rows)

    elif a.cmd == "report":
        t = time.time()
        RP.build_report(a.out, a.md, quick=a.quick)
        print(f"wrote {a.md} and CSVs in {a.out} ({time.time() - t:.1f} s)")
        print(RP.fmt_table(RP.ranked_steps()[:5], ["step", "effort", "speedup_alone",
                                                    "loss_if_dropped", "score", "d_lut",
                                                    "d_dsp"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
