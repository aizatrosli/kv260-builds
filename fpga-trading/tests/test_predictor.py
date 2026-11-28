"""ftrade.predictor: dataset construction, leakage, float vs int8, TPU sim bit-exactness, API."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftrade import synth                                 # noqa: E402
from ftrade.predictor import dataset as D                # noqa: E402
from ftrade.predictor.infer import Predictor, stream, StreamPredictor, ART   # noqa: E402

HAVE_ART = (ART / "model.pt").exists() and (ART / "qmodel").exists()
need_art = pytest.mark.skipif(not HAVE_ART, reason="no trained artifacts (run python -m ftrade.predictor.train)")


@pytest.fixture(scope="module")
def streams():
    return D.synth_streams((7, 8), 20_000)


@pytest.fixture(scope="module")
def built(streams):
    return D.build(streams)


@pytest.fixture(scope="module")
def fresh():
    """An unseen synth stream, normalised with the shipped config."""
    p = Predictor() if HAVE_ART else None
    cfg = p.cfg if p else D.NormConfig()
    F = D.features_array(synth.generate(15_000, seed=555))
    return D.build_stream(F, cfg, 0, fracs=(0.0, 0.0, 1.0))["test"]


# ---------------------------------------------------------------------------------- dataset --
def test_shapes_and_ranges(built):
    cfg, sp = built
    for k in ("train", "val", "test"):
        s = sp[k]
        assert len(s) > 1000
        assert s.X.shape == (len(s), D.N_IN) and s.y.shape == (len(s),)
        assert s.X.min() >= -127 and s.X.max() <= 127
        assert set(np.unique(s.y)) <= {0, 1, 2}
        assert s.xf.dtype == np.float32 and np.array_equal(s.xf * 32, s.X)
    assert cfg.pu >= 1 and cfg.fs >= 0 and cfg.db >= 0


def test_no_leakage_time_order(built):
    cfg, sp = built
    for sid in (0, 1):
        tr, va, te = (sp[k].t[sp[k].stream == sid] for k in ("train", "val", "test"))
        # every message a train sample touches (t-W+1 .. t+H) precedes every message of val, etc.
        assert tr.max() + cfg.H < va.min() - cfg.W + 1
        assert va.max() + cfg.H < te.min() - cfg.W + 1


def test_norm_fit_uses_train_rows_only(streams, built):
    cfg, _ = built
    tampered = []
    for F in streams:
        F = F.copy()
        end = D.split_bounds(len(F), cfg)["train"][1]
        F[end:, 2] *= 7                # wreck spreads / flow outside the training part
        F[end:, 6] *= 50
        tampered.append(F)
    cfg2, _ = D.build(tampered)
    assert cfg2 == cfg


def test_labels_match_future_mid(streams, built):
    cfg, sp = built
    Hd, _ = D.hold_valid(streams[0])
    mid2 = Hd[:, 0] + Hd[:, 1]
    s = sp["test"]
    t = s.t[s.stream == 0]
    d = mid2[t + cfg.H] - mid2[t]
    want = np.where(d > cfg.db, 2, np.where(d < -cfg.db, 0, 1))
    assert np.array_equal(want, s.y[s.stream == 0])


def test_window_function_matches_vectorised(streams, built):
    """The single-window path (used by Predictor / streaming) equals the dataset's vectorised one."""
    cfg, sp = built
    F = streams[1]
    Hd, _ = D.hold_valid(F)
    s = sp["val"]
    rng = np.random.default_rng(0)
    for i in rng.choice(np.where(s.stream == 1)[0], 300, replace=False):
        t = s.t[i]
        assert np.array_equal(D.inputs_from_window(Hd[t - cfg.W + 1:t + 1], cfg), s.X[i]), t
        raw = F[t - cfg.W + 1:t + 1]                     # raw rows, all valid: no holding needed
        if D.hold_valid(raw)[1].all() and (raw[:, 2] > 0).all():
            assert np.array_equal(D.inputs_from_window(raw, cfg), s.X[i]), t


def test_hold_valid():
    F = np.array([[0, 0, 0, 5, 0, 0, 0, 0], [10, 12, 2, 5, 6, 1, 2, 1], [10, 0, 0, 5, 0, 32768, 2, 0],
                  [11, 13, 2, 1, 1, 0, 3, 1]], np.int64)
    Hd, has = D.hold_valid(F)
    assert has.tolist() == [False, True, True, True]
    assert np.array_equal(Hd[2], F[1]) and np.array_equal(Hd[3], F[3]) and not Hd[0].any()


# ------------------------------------------------------------------------------ float/int8 ---
@need_art
def test_int8_agrees_with_float(fresh):
    pf, pi = Predictor(backend="float"), Predictor(backend="int8")
    lf, li = pf.logits_codes(fresh.X), pi.logits_codes(fresh.X)
    agree = (lf.argmax(1) == li.argmax(1)).mean()
    assert agree > 0.93, agree
    assert np.corrcoef(lf.ravel(), li.ravel())[0, 1] > 0.98


@need_art
def test_model_beats_majority_on_unseen_stream(fresh):
    import json
    p = Predictor(backend="int8")
    acc = (p.logits_codes(fresh.X).argmax(1) == fresh.y).mean()
    maj = np.bincount(fresh.y, minlength=3).max() / len(fresh)
    assert acc > maj + 0.03, (acc, maj)
    m = json.loads((ART / "metrics.json").read_text())
    assert m["quant"]["tpusim"]["bit_exact"] is True


# ----------------------------------------------------------------------------------- TPU sim -
def _sim_ok():
    try:
        from ftrade.predictor import tpu
        return tpu.sim_available()
    except Exception:
        return False


need_sim = pytest.mark.skipif(not (HAVE_ART and _sim_ok()), reason="tpuv1 ISA simulator / artifacts unavailable")


@need_sim
def test_tpusim_bit_exact_small_batch(fresh):
    from ftrade.predictor import tpu
    p = Predictor(backend="int8")
    X = fresh.X[:48]
    ref = tpu.int_codes(p._qg, X)
    prog = tpu.compile_tpucc(p._qg, len(X))
    assert len(prog.kernels) == 1 and prog.n_instrs <= 2048
    got, _ = tpu.run_tpusim(prog, X)
    assert np.array_equal(got, ref)
    gold, _ = tpu.run_tpusim(tpu.compile_tpucc(p._qg, 8), X[:8], golden=True)   # unmodified CoreSim
    assert np.array_equal(gold, ref[:8])
    tm = tpu.TpurtMLP(p._qg, "sim", B=16)                   # the board code path, on tpurt's sim
    try:
        assert np.array_equal(tm.run(X), ref)
    finally:
        tm.close()


@need_sim
def test_predictor_backends_consistent(fresh, streams):
    F = streams[0][5000:5000 + 40]
    pi, ps = Predictor(backend="int8"), Predictor(backend="tpusim")
    a, b = pi.predict(F), ps.predict(F)
    assert a == b
    assert set(a) >= {"p_up", "p_flat", "p_down", "signal"} and a["signal"] in (-1, 0, 1)
    assert abs(a["p_up"] + a["p_flat"] + a["p_down"] - 1) < 1e-9


@need_art
def test_stream_helper_and_invalid_window():
    ev = synth.generate(400, seed=9)
    p = Predictor(backend="int8")
    outs = list(stream(ev, p, every=10))
    assert len(outs) == 40 - 1 and all(o[2]["signal"] in (-1, 0, 1) for o in outs)
    # same numbers as the dataset's inputs for that message
    from ftrade.features import run
    F = np.asarray(run(ev), np.int64)
    Hd, _ = D.hold_valid(F)
    e, f, pr = outs[-1]
    x = D.window_inputs(Hd[:400], p.cfg)[399]
    assert f == F[399].tolist() and pr == p.predict_batch(x[None])[0]
    sp = StreamPredictor(p)
    res = [sp.update(r) for r in F[:50]]
    assert res[0] is None and res[-1] == p.predict(Hd[34:50])
    assert p.predict(np.zeros((16, 8), np.int64))["valid"] is False


@pytest.mark.skipif(not os.path.exists("/dev/tpu0") or not HAVE_ART, reason="/dev/tpu0 absent (board-only)")
def test_board_backend(fresh):
    pb, pi = Predictor(backend="board"), Predictor(backend="int8")
    try:
        assert np.array_equal(pb.logits_codes(fresh.X[:64]), pi.logits_codes(fresh.X[:64]))
    finally:
        pb.close()
