import numpy as np
import pytest

from tpuquant.observers import make_observer
from tpuquant.qscheme import act_qparams, quantize_weight, quantize_np
from tpuquant.int4 import pack_int4, unpack_int4, pack_uint4, unpack_uint4, expand_group_weights


def _obs(method, x, chunks=4):
    o = make_observer(method)
    for c in np.array_split(x, chunks):
        o.observe_range(c)
    for c in np.array_split(x, chunks):
        o.observe_hist(c)
    return o


def _qmse(x, lo, hi, sym):
    s, zp = act_qparams(lo, hi, sym)
    return float(((x - (quantize_np(x, s, zp) - zp) * s) ** 2).mean())


def test_minmax_percentile():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200000)
    lo, hi = _obs("minmax", x).range(True)
    assert hi == pytest.approx(np.abs(x).max())
    lo, hi = _obs("minmax", x).range(False)
    assert (lo, hi) == (pytest.approx(x.min()), pytest.approx(x.max()))
    for p in (99.0, 99.9, 99.99):
        a = _obs(f"percentile:{p}", x).range(True)[1]
        assert a == pytest.approx(np.percentile(np.abs(x), p), rel=5e-3)
    lo, hi = _obs("percentile:99", x).range(False)
    assert lo == pytest.approx(np.percentile(x, 1), rel=1e-2)
    assert hi == pytest.approx(np.percentile(x, 99), rel=1e-2)


def test_mse_beats_minmax_on_heavy_tails():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.laplace(0, 1, 100000), [60.0, -45.0]])
    for sym in (True, False):
        mm = _obs("minmax", x).range(sym)
        ms = _obs("mse", x).range(sym)
        assert ms[1] < mm[1]
        assert _qmse(x, *ms, sym) < 0.8 * _qmse(x, *mm, sym)


def test_act_qparams_tpuv1_convention():
    s, zp = act_qparams(-1.0, 3.0, symmetric=False)
    assert s == pytest.approx(4 / 255) and zp == int(np.round(-128 + 1 / s))
    s, zp = act_qparams(0.0, 1.0, symmetric=False)
    assert zp == -128
    s, zp = act_qparams(-2.0, 1.0, symmetric=True)
    assert zp == 0 and s == pytest.approx(2 / 127)


def test_int8_weights_match_jepa_recipe():
    rng = np.random.default_rng(2)
    W = rng.normal(0, 0.1, (128, 96))
    q = quantize_weight(W, 8)
    s = np.maximum(np.abs(W).max(0), 1e-8) / 127
    assert np.array_equal(q["w"], np.clip(np.round(W / s), -127, 127))
    assert np.allclose(q["scale"], s)


@pytest.mark.parametrize("scheme", ["channel", "group_int", "group_pow2", "group_float"])
def test_int4_schemes(scheme):
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.1, (128, 64)) * rng.uniform(0.2, 3, (128, 1))   # row-varying magnitude
    q = quantize_weight(W, 4, scheme, 32, "minmax")
    assert np.abs(q["q4"]).max() <= 7
    if scheme != "group_float":
        assert np.abs(q["w"]).max() <= 127
        assert np.allclose(q["w_deq"], q["w"] * q["scale"])
    if scheme.startswith("group_") and scheme != "group_float":
        assert np.array_equal(expand_group_weights(q["q4"], q["gmul"], 32), q["w"])
        assert q["gmul"].min() >= 1 and q["gmul"].max() <= 16
        if scheme == "group_pow2":
            assert set(np.unique(q["gmul"])) <= {1, 2, 4, 8, 16}
    err = np.mean((W - q["w_deq"]) ** 2)
    err8 = np.mean((W - quantize_weight(W, 8)["w_deq"]) ** 2)
    assert err8 < err


def test_int4_group_and_mse_reduce_error():
    rng = np.random.default_rng(4)
    W = rng.normal(0, 0.1, (256, 64)) * rng.uniform(0.2, 3, (256, 1))
    e = lambda **k: np.mean((W - quantize_weight(W, 4, **k)["w_deq"]) ** 2)
    assert e(scheme="group_int") < e(scheme="channel")
    assert e(scheme="group_float") < e(scheme="channel")
    assert e(scheme="channel", method="mse") < e(scheme="channel", method="minmax")


def test_pack_unpack():
    rng = np.random.default_rng(5)
    for shape, axis in [((7, 33), -1), ((16, 16), 0), ((3, 5, 9), 1)]:
        q = rng.integers(-8, 8, shape)
        p = pack_int4(q, axis)
        assert p.dtype == np.uint8 and p.shape[axis] == (shape[axis] + 1) // 2
        assert np.array_equal(unpack_int4(p, shape[axis], axis), q)
    assert pack_int4(np.array([1, -1]))[0] == 0xF1
    u = rng.integers(0, 16, (4, 10))
    assert np.array_equal(unpack_uint4(pack_uint4(u), 10), u)
    with pytest.raises(ValueError):
        pack_int4(np.array([9]))
