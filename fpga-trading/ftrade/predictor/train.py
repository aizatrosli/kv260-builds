"""Train / evaluate / quantise / compile the direction MLP.

    cd fpga-trading
    python -m ftrade.predictor.train                       # synth seeds 0-3, writes artifacts/
    python -m ftrade.predictor.train --data data/xbtmyr_sample.jsonl.gz --out /tmp/real
    python -m ftrade.predictor.train --data a.jsonl.gz --data b.jsonl.gz --no-tpu

Writes into --out: model.pt, norm.json, qmodel/ (tpuquant export), metrics.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import dataset as D
from .model import DirectionMLP, save as save_model

ART = Path(__file__).resolve().parent / "artifacts"


# ------------------------------------------------------------------------------- training ----
def train_mlp(tr: D.Split, va: D.Split, seed=0, epochs=30, lr=2e-3, wd=1e-4, bs=512, patience=5,
              verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DirectionMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(tr.xf), torch.tensor(tr.y)
    Xv, yv = torch.tensor(va.xf), torch.tensor(va.y)
    lossf = nn.CrossEntropyLoss()
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lossf(m(Xv), yv))
        if verbose:
            print(f"  ep {ep:2d} val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m.eval(), best


def predict_float(m, X):
    with torch.no_grad():
        return m(torch.tensor((np.asarray(X) * D.IN_SCALE).astype(np.float32))).numpy()


# -------------------------------------------------------------------------------- metrics ----
def imb_rule(X, thr):
    """Baseline: signal = sign(imbalance code) with a flat band |imb| <= thr. -> class."""
    imb = np.asarray(X)[:, 0]
    return np.where(imb > thr, 2, np.where(imb < -thr, 0, 1))


def fit_imb_thr(sp: D.Split):
    best = max(range(0, 64, 2), key=lambda t: (imb_rule(sp.X, t) == sp.y).mean())
    return best


def scores(pred, sp: D.Split, cfg: D.NormConfig):
    """accuracy + trading-ish edge: mean of signal * future mid move (MYR) over non-flat calls."""
    pred = np.asarray(pred)
    sig = pred - 1
    act = sig != 0
    move = sp.d / 2.0                                  # mid2 -> MYR
    hit = (np.sign(move[act]) == sig[act]).mean() if act.any() else float("nan")
    return dict(acc=float((pred == sp.y).mean()), coverage=float(act.mean()),
                hit_rate=float(hit),
                edge_myr=float((sig[act] * move[act]).mean()) if act.any() else 0.0,
                edge_per_pu=float((sig[act] * move[act]).mean() / cfg.pu) if act.any() else 0.0)


def evaluate(m, sp: D.Split, cfg, prior_cls, thr):
    lg = predict_float(m, sp.X)
    return {"mlp": scores(lg.argmax(1), sp, cfg),
            "majority": scores(np.full(len(sp), prior_cls), sp, cfg),
            "imb_sign": scores(imb_rule(sp.X, 0), sp, cfg),
            f"imb_band{thr}": scores(imb_rule(sp.X, thr), sp, cfg)}


def experiment(streams, seeds=(0, 1, 2), verbose=True, extra_test=None):
    cfg, sp = D.build(streams)
    prior = int(np.bincount(sp["train"].y, minlength=3).argmax())
    thr = fit_imb_thr(sp["train"])
    res = dict(norm=cfg.to_dict(), n={k: len(v) for k, v in sp.items()},
               label_dist={k: (np.bincount(v.y, minlength=3) / max(1, len(v))).round(4).tolist()
                           for k, v in sp.items()}, runs=[])
    models = []
    for s in seeds:
        t0 = time.time()
        m, vl = train_mlp(sp["train"], sp["val"], seed=s)
        r = dict(seed=s, val_loss=vl, val=evaluate(m, sp["val"], cfg, prior, thr),
                 test=evaluate(m, sp["test"], cfg, prior, thr), train_s=round(time.time() - t0, 1))
        if extra_test is not None:
            r["fresh"] = evaluate(m, extra_test(cfg), cfg, prior, thr)
        res["runs"].append(r)
        models.append((vl, s, m))
        if verbose:
            t = r["test"]
            print(f"seed {s}: val loss {vl:.4f}  test acc mlp {t['mlp']['acc']:.4f}  majority "
                  f"{t['majority']['acc']:.4f}  imb-sign {t['imb_sign']['acc']:.4f}  "
                  f"imb-band {t[f'imb_band{thr}']['acc']:.4f}  ({r['train_s']} s)")
    best = min(models, key=lambda z: z[0])
    res["best_seed"] = best[1]
    return cfg, sp, best[2], res


def summarize(res, split="test"):
    keys = list(res["runs"][0][split].keys())
    out = {}
    for k in keys:
        accs = [r[split][k]["acc"] for r in res["runs"]]
        edges = [r[split][k]["edge_myr"] for r in res["runs"]]
        hits = [r[split][k]["hit_rate"] for r in res["runs"]]
        out[k] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                      hit_mean=float(np.nanmean(hits)), edge_myr_mean=float(np.mean(edges)),
                      coverage=res["runs"][0][split][k]["coverage"])
    return out


# ------------------------------------------------------------------------------ quant + tpu --
def quant_and_check(m, sp, n_check=300, n_golden=300, verbose=True):
    from . import tpu
    calib = sp["train"].xf[np.random.default_rng(0).choice(len(sp["train"]), min(4000, len(sp["train"])), replace=False)]
    out, graphs = {}, {}
    test = sp["test"]
    lf = predict_float(m, test.X)
    for meth in ("minmax", "percentile", "mse"):
        qg = tpu.quantize(m, calib, meth)
        qi = tpu.int_codes(qg, test.X)
        ag = qi.argmax(1) == lf.argmax(1)
        srt = np.sort(lf, 1)
        clear = (srt[:, 2] - srt[:, 1]) > 0.1          # float top-2 logit margin > 0.1
        agree = float(ag.mean())
        acc = float((qi.argmax(1) == test.y).mean())
        out[meth] = dict(argmax_agree=agree, argmax_agree_margin01=float(ag[clear].mean()),
                         frac_margin01=float(clear.mean()), int8_test_acc=acc)
        graphs[meth] = qg
        if verbose:
            print(f"int8 {meth:10s}: float/int8 argmax agreement {agree:.4f}, int8 test acc {acc:.4f}")
    best = max(out, key=lambda k: out[k]["argmax_agree"])
    qg = graphs[best]
    res = dict(per_method=out, method=best, float_test_acc=float((lf.argmax(1) == test.y).mean()))
    # bit-exactness: tpuquant integer interpreter vs compiled tpuv1 program on the ISA simulator
    if tpu.sim_available():
        xs = test.X[:n_check]
        ref = tpu.int_codes(qg, xs)
        t0 = time.time()
        prog = tpu.compile_tpucc(qg, len(xs))
        got, rr = tpu.run_tpusim(prog, xs)
        res["tpusim"] = dict(n=len(xs), bit_exact=bool(np.array_equal(got, ref)),
                             mismatches=int((got != ref).any(1).sum()), instrs=prog.n_instrs,
                             program_bytes=prog.n_bytes, kernels=len(prog.kernels), s=round(time.time() - t0, 2))
        xg = xs[:n_golden]
        pg = tpu.compile_tpucc(qg, len(xg))
        gg, _ = tpu.run_tpusim(pg, xg, golden=True)
        res["tpusim_golden"] = dict(n=len(xg), bit_exact=bool(np.array_equal(gg, ref[:len(xg)])))
        p1 = tpu.compile_tpucc(qg, 1)
        res["tpucc_B1"] = dict(instrs=p1.n_instrs, program_bytes=p1.n_bytes,
                               opcodes={str(k): int(v) for k, v in p1.opcode_counts().items()})
        t0 = time.time()
        tm = tpu.TpurtMLP(qg, "sim", B=len(xs))
        gt = tm.run(xs)
        tm.close()
        res["tpurt_sim"] = dict(n=len(xs), bit_exact=bool(np.array_equal(gt, ref)), s=round(time.time() - t0, 2))
        if verbose:
            print("tpucc+ISA sim:", res["tpusim"], "| golden CoreSim:", res["tpusim_golden"],
                  "| tpurt sim backend:", res["tpurt_sim"])
    else:
        res["tpusim"] = "unavailable"
    res["board_latency_B1"] = tpu.board_latency(batch=1)
    res["board_latency_B16"] = tpu.board_latency(batch=16)
    if verbose:
        print("tpuperf board estimate B=1:", res["board_latency_B1"]["total_us"], "us; B=16:",
              res["board_latency_B16"]["us_per_inference"], "us/inference")
    return qg, res


# ------------------------------------------------------------------------------------ main ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", help="recorded events JSONL(.gz); repeatable. Default: synth")
    ap.add_argument("--synth-seeds", default="0,1,2,3")
    ap.add_argument("--synth-events", type=int, default=150_000)
    ap.add_argument("--train-seeds", default="0,1,2")
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--no-tpu", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tseeds = [int(s) for s in a.train_seeds.split(",")]
    extra = None
    if a.data:
        streams = [D.load_jsonl_features(p) for p in a.data]
        src = dict(kind="recorded", files=a.data, events=[int(len(s)) for s in streams])
    else:
        ss = [int(s) for s in a.synth_seeds.split(",")]
        streams = D.synth_streams(ss, a.synth_events)
        src = dict(kind="synth", seeds=ss, events=a.synth_events)
        fresh = D.synth_streams((100, 101), a.synth_events // 2)

        def extra(cfg):           # whole unseen streams (never touched by training or fitting)
            return D.concat(D.build_stream(F, cfg, 100 + i, fracs=(0.0, 0.0, 1.0))["test"]
                            for i, F in enumerate(fresh))
    print(f"data: {src}")
    cfg, sp, m, res = experiment(streams, tseeds, extra_test=extra)
    res["source"] = src
    res["summary_test"] = summarize(res, "test")
    if extra:
        res["summary_fresh"] = summarize(res, "fresh")
    print(json.dumps(res["summary_test"], indent=1))
    save_model(m, out / "model.pt")
    (out / "norm.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    if not a.no_tpu:
        from . import tpu  # noqa: F401  (puts tpu-stack on sys.path)
        from tpuquant import export
        qg, qres = quant_and_check(m, sp)
        export.save(qg, out / "qmodel")
        res["quant"] = qres
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
