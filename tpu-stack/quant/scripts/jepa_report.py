#!/usr/bin/env python3
"""Full accuracy study of the jepa-tpu encoder with tpuquant.

    python3 scripts/jepa_report.py [--calib 256] [--eval 200] [--out results]

Writes results/jepa_report.md, results/jepa_report.json and the recommended INT8 export in
results/export_jepa_int8/ (+ a mixed INT4/INT8 export in results/export_jepa_int4mix/).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tpuquant import QConfig, export                                     # noqa: E402
from tpuquant import jepa as J                                           # noqa: E402
from tpuquant.analysis import (output_metrics, compare_calibration, act_sensitivity,  # noqa: E402
                               weight_sensitivity, mixed_precision_sweep, layer_report, format_rows)
from tpuquant.int4 import hardware_report                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--sens", type=int, default=32, help="images for sensitivity analysis")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(4)
    enc = J.load_encoder()
    xtr, _, xte, _ = J.load_mnist()
    cal = J.calibrate_encoder(enc, xtr[:a.calib])
    x = xte[:a.eval].numpy()
    with torch.no_grad():
        ref = enc(xte[:a.eval]).double().numpy()
    R = {"calib_images": a.calib, "eval_images": a.eval}
    md = [f"# tpuquant study: jepa-tpu encoder\n",
          f"Calibration {a.calib} MNIST train images, evaluation {a.eval} test images. Metric: cosine of "
          "the integer encoder output (dequantised) vs float PyTorch, per image; SQNR over all outputs.\n"]

    # 1. jepa reference (bit-exact) for context
    P = J.load_jepa_qmodel()
    gj = cal.build(QConfig(fixed_qparams=J.jepa_fixed_qparams(cal, P)))
    R["jepa_reference"] = output_metrics(ref, gj.run(x)[0])
    md.append("## 1. Reproduction of jepa-tpu\n\nWith jepa-tpu's own scales the toolkit emits the identical "
              f"program (bit-exact, see tests). Its accuracy here: cos {R['jepa_reference']['cos_mean']:.5f} "
              f"(min {R['jepa_reference']['cos_min']:.5f}).\n")

    # 2. calibration comparison
    rows = compare_calibration(cal, x, ref, symmetric=(True, False))
    rows += [dict(method=r["method"] + " +sm", symmetric=r["symmetric"], **{k: v for k, v in r.items() if k not in ("method", "symmetric")})
             for r in compare_calibration(cal, x, ref, methods=("percentile", "mse"), symmetric=(True, False),
                                          base_cfg=QConfig(softmax_mult="mse"))]
    R["calibration"] = rows
    md.append("## 2. Activation calibration (INT8 weights)\n\n`+sm` = softmax output scale calibrated "
              "(`softmax_mult='mse'`) instead of the fixed 1/127.\n")
    md.append(format_rows(rows, ["method", "symmetric", "cos_mean", "cos_min", "sqnr_db"],
                          {"cos_mean": "{:.5f}", "cos_min": "{:.5f}", "sqnr_db": "{:.2f}"}) + "\n")

    best_cfg = QConfig(act_method="mse", softmax_mult="mse")
    g8 = cal.build(best_cfg)

    # 3. sensitivity
    xs = x[:a.sens]
    asens = act_sensitivity(g8, xs)
    wsens = weight_sensitivity(g8, xs, bits=4, scheme="group_int", method="mse")
    R["act_sensitivity"] = asens
    R["weight_sensitivity_int4"] = wsens
    md.append(f"## 3. Sensitivity ({a.sens} images, one tensor quantised at a time, rest float)\n")
    md.append("Activations (10 most harmful):\n\n" + format_rows(asens[:10], ["name", "op", "sqnr_db", "cos"],
                                                                {"sqnr_db": "{:.2f}", "cos": "{:.6f}"}) + "\n")
    md.append("Weights at INT4 group_int (10 most harmful):\n\n" + format_rows(
        wsens[:10], ["name", "params", "sqnr_db", "cos"], {"sqnr_db": "{:.2f}", "cos": "{:.6f}"}) + "\n")

    # 4. INT4
    int4 = []
    for sch in ("channel", "group_int", "group_pow2", "group_float"):
        for wm in ("minmax", "mse"):
            cfg = QConfig(**{**best_cfg.__dict__, "weight_bits": 4, "int4_scheme": sch, "weight_method": wm})
            g = cal.build(cfg)
            fq = output_metrics(ref, g.float_outputs(g.run_float(x, quant_acts="all", quant_weights="all"))[0])
            row = dict(scheme=sch, weight_method=wm, fakequant_cos=fq["cos_mean"], fakequant_sqnr=fq["sqnr_db"],
                       int_cos=float("nan"), int_cos_min=float("nan"), int_sqnr=float("nan"))
            if sch != "group_float":
                m = output_metrics(ref, g.run(x)[0])
                row.update(int_cos=m["cos_mean"], int_cos_min=m["cos_min"], int_sqnr=m["sqnr_db"])
            int4.append(row)
    R["int4"] = int4
    m8 = output_metrics(ref, g8.run(x)[0]); R["int8_best"] = m8
    md.append("## 4. INT4 weights (activations: best INT8 config, group size 32)\n\n"
              f"INT8 reference: cos {m8['cos_mean']:.5f}, SQNR {m8['sqnr_db']:.2f} dB. `group_float` has no "
              "integer-exact form on tpuv1 (fake-quant only).\n")
    md.append(format_rows(int4, ["scheme", "weight_method", "int_cos", "int_cos_min", "int_sqnr", "fakequant_cos"],
                          {k: "{:.5f}" for k in ("int_cos", "int_cos_min", "fakequant_cos")} | {"int_sqnr": "{:.2f}"}) + "\n")

    # 5. mixed precision
    ranking = [r["name"] for r in wsens]
    base4 = QConfig(**{**best_cfg.__dict__, "int4_scheme": "group_int", "weight_method": "mse"})
    sweep = mixed_precision_sweep(cal, base4, ranking, [0, 1, 3, 6, 12, 25], x, ref)
    R["mixed_precision"] = sweep
    md.append("## 5. Mixed precision: keep the k most INT4-sensitive layers at INT8\n\n" + format_rows(
        [dict(k=s["k"], weight_kib=s["weight_kib"], cos_mean=s["cos_mean"], cos_min=s["cos_min"], sqnr_db=s["sqnr_db"],
              int8=", ".join(s["int8_layers"][:6]) + (" ..." if s["k"] > 6 else "")) for s in sweep],
        ["k", "weight_kib", "cos_mean", "cos_min", "sqnr_db", "int8"],
        {"weight_kib": "{:.0f}", "cos_mean": "{:.5f}", "cos_min": "{:.5f}", "sqnr_db": "{:.2f}"}) + "\n")

    # 6. per-layer report of the recommended config
    lr = layer_report(g8, x[:a.sens])
    R["layer_report_int8_best"] = lr
    md.append("## 6. Per-layer accumulated error, recommended INT8 config (last block)\n\n" + format_rows(
        lr[-14:], ["name", "op", "sqnr_db", "cos", "scale", "sat"],
        {"sqnr_db": "{:.2f}", "cos": "{:.6f}", "scale": "{:.4g}", "sat": "{:.2e}"}) + "\n")

    # 7. hardware / exports
    g4 = cal.build(QConfig(**{**base4.__dict__, "weight_bits": 4, "layer_bits": {n: 8 for n in ranking[:3]}}))
    hw = hardware_report(g8, g4)
    R["hardware"] = hw
    export.save(g8, os.path.join(a.out, "export_jepa_int8"), extra=dict(metrics=m8))
    export.save(g4, os.path.join(a.out, "export_jepa_int4mix"),
                extra=dict(metrics=output_metrics(ref, g4.run(x)[0])))
    md.append("## 7. Memory\n\n" + "\n".join(f"- {k}: {v:.2f}" for k, v in hw.items() if isinstance(v, float)) + "\n")
    R["runtime_s"] = time.time() - t0
    md.append(f"\nRuntime {R['runtime_s']:.0f} s.\n")
    with open(os.path.join(a.out, "jepa_report.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(os.path.join(a.out, "jepa_report.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
