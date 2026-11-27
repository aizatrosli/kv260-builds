"""Accuracy tools: per-layer SQNR / cosine, sensitivity analysis, calibration comparison."""
import numpy as np

from .graph import LAYOUT_OPS, WEIGHT_OPS
from .qscheme import quantize_weight
from .ptq import QConfig


def sqnr_db(ref, x):
    ref = np.asarray(ref, np.float64); x = np.asarray(x, np.float64)
    noise = ((ref - x) ** 2).sum()
    return float("inf") if noise == 0 else float(10 * np.log10((ref ** 2).sum() / noise))


def cosine(ref, x):
    a = np.asarray(ref, np.float64).ravel(); b = np.asarray(x, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def per_sample_cosine(ref, x):
    a = np.asarray(ref, np.float64).reshape(len(ref), -1); b = np.asarray(x, np.float64).reshape(len(x), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def output_metrics(ref, y):
    c = per_sample_cosine(ref, y)
    return dict(cos_mean=float(c.mean()), cos_min=float(c.min()), sqnr_db=sqnr_db(ref, y))


def _quant_tensors(g):
    return [n.output for n in g.nodes if n.op not in LAYOUT_OPS and n.op != "const"]


def layer_report(g, x):
    """Integer graph vs float graph, tensor by tensor (accumulated error).
    Returns list of dict(name, op, sqnr_db, cos, scale, zp, sat) where sat is the fraction of
    int8 values pinned at the clamp limits (a clipping indicator)."""
    qi = g.run_int(x, keep_all=True)
    fl = g.run_float(x)
    rows = []
    for n in g.nodes:
        if n.op in LAYOUT_OPS or n.op == "const":
            continue
        t = g.tensors[n.output]
        q = qi[n.output]
        y = t.dequant(q)
        ref = fl[n.output]
        lo, hi = t.act_min, t.act_max
        sat = float(((q <= -128) | (q >= 127)).mean()) if n.op != "softmax" else float((q >= 127).mean())
        rows.append(dict(name=n.output, op=n.op, sqnr_db=sqnr_db(ref, y), cos=cosine(ref, y),
                         scale=float(np.max(t.scale)), zp=t.zp, sat=sat))
    return rows


def format_rows(rows, keys, fmt=None):
    fmt = fmt or {}
    head = " | ".join(keys)
    out = ["| " + head + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(k, "{}").format(r[k]) if not isinstance(r[k], float)
                                     else fmt.get(k, "{:.4g}").format(r[k]) for k in keys) + " |")
    return "\n".join(out)


def act_sensitivity(g, x):
    """Quantise ONE activation tensor at a time (weights + all other tensors float).
    Returns rows sorted from most to least harmful (lowest output SQNR first)."""
    ref = g.float_outputs(g.run_float(x))[0]
    rows = []
    for t in _quant_tensors(g):
        y = g.float_outputs(g.run_float(x, quant_acts={t}))[0]
        rows.append(dict(name=t, op=_op_of(g, t), sqnr_db=sqnr_db(ref, y), cos=cosine(ref, y)))
    return sorted(rows, key=lambda r: r["sqnr_db"])


def _op_of(g, t):
    for n in g.nodes:
        if n.output == t:
            return n.op
    return "?"


def weight_sensitivity(g, x, bits=4, scheme="group_int", group_size=32, method="minmax"):
    """Quantise ONE layer's weights at `bits` (everything else float). Lowest SQNR first."""
    ref = g.float_outputs(g.run_float(x))[0]
    rows = []
    for n in g.weight_nodes():
        wd = quantize_weight(n.meta["w_float"], bits, scheme, group_size, method)["w_deq"]
        y = g.float_outputs(g.run_float(x, weights_override={n.name: wd}))[0]
        rows.append(dict(name=n.name, op=n.op, params=int(n.meta["w_float"].size),
                         sqnr_db=sqnr_db(ref, y), cos=cosine(ref, y)))
    return sorted(rows, key=lambda r: r["sqnr_db"])


def mixed_precision_sweep(cal, base_cfg, ranking, ks, x, ref):
    """Keep the k most sensitive layers (ranking: names, most sensitive first) at INT8 and the
    rest at INT4; evaluate the INTEGER graph. Returns list of dict(k, int8_layers, metrics, kib)."""
    out = []
    for k in ks:
        bits = {name: (8 if i < k else 4) for i, name in enumerate(ranking)}
        cfg = QConfig(**{**base_cfg.__dict__, "layer_bits": bits, "weight_bits": 4})
        g = cal.build(cfg)
        y = g.run(x)[0]
        out.append(dict(k=k, int8_layers=list(ranking[:k]), weight_kib=g.weight_bytes() / 1024,
                        **output_metrics(ref, y)))
    return out


def compare_calibration(cal, x, ref, methods=("minmax", "percentile", "percentile:99.9", "mse"),
                        base_cfg=None, symmetric=(True,)):
    """Build + run the integer graph for each activation observer. Returns list of dict."""
    base = base_cfg or QConfig()
    rows = []
    for sym in symmetric:
        for m in methods:
            g = cal.build(QConfig(**{**base.__dict__, "act_method": m, "act_symmetric": sym}))
            rows.append(dict(method=m, symmetric=sym, **output_metrics(ref, g.run(x)[0])))
    return rows
