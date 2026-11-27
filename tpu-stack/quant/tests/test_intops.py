import os
import sys
import importlib.util
import numpy as np
import pytest

from tpuquant import intops as I
from tpuquant.jepa import JEPA_DIR
from conftest import TPUV1_QUANT


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(m)
    finally:
        sys.dont_write_bytecode = old
    return m


@pytest.fixture(scope="module")
def tfl():
    p = os.path.join(TPUV1_QUANT, "tflite_int8.py")
    if not os.path.exists(p):
        pytest.skip("tpuv1 not available")
    return _load(p, "tflite_int8_ref")


@pytest.fixture(scope="module")
def jint():
    p = os.path.join(JEPA_DIR, "intops.py")
    if not os.path.exists(p):
        pytest.skip("jepa-tpu not available")
    return _load(p, "jepa_intops_ref")


def test_requant_matches_tpuv1(tfl):
    rng = np.random.default_rng(0)
    n = 20000
    acc = rng.integers(-2**31, 2**31, n)
    acc[: n // 2] = rng.integers(-2**20, 2**20, n // 2)
    bias = rng.integers(-2**24, 2**24, n)
    m0 = rng.integers(2**30, 2**31, n)
    sh = rng.integers(-31, 8, n)
    zp = rng.integers(-128, 128, n)
    lo = rng.integers(-128, 0, n); hi = rng.integers(0, 128, n)
    got = I.requant(acc, bias, m0, sh, zp, lo, hi)
    want = np.array([tfl.requant(int(acc[i]), int(bias[i]), int(m0[i]), int(sh[i]), int(zp[i]),
                                 int(lo[i]), int(hi[i])) for i in range(n)])
    assert np.array_equal(got, want)
    # corners
    assert int(I.srdhm(I.INT32_MIN, I.INT32_MIN)) == I.INT32_MAX
    for a, b in [(I.INT32_MIN, I.INT32_MAX), (-3, 1 << 30), (-(1 << 15), 1 << 15), (I.INT32_MAX, I.INT32_MAX)]:
        assert int(I.srdhm(a, b)) == tfl.srdhm(a, b)
    for x, e in [(5, 1), (-5, 1), (6, 2), (-6, 2), (2, 2), (-2, 2), (I.INT32_MIN, 31)]:
        assert int(I.rdbpot(x, e)) == tfl.rdbpot(x, e)


def test_quantize_multiplier_and_range(tfl):
    rng = np.random.default_rng(1)
    for r in np.concatenate([10 ** rng.uniform(-12, 2, 3000), [0.0, 0.5, 1.0, 3.9, 1e-12]]):
        assert I.quantize_multiplier(r) == tfl.quantize_multiplier(float(r))
    for kind in ("none", "relu", "relu6"):
        for zp in (-128, -5, 0, 40):
            assert I.act_range(kind, 0.05, zp) == tfl.act_range(kind, 0.05, zp)


def test_vector_ops_match_jepa(jint):
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 128, (500, 128))
    assert np.array_equal(I.layernorm(x), jint.layernorm(x))
    assert np.array_equal(I.layernorm(x * 0), jint.layernorm(x * 0))
    for s in (0.01, 0.05, 0.2):
        lut = I.exp_lut(s)
        assert np.array_equal(lut, jint.exp_lut(s))
        assert np.array_equal(I.softmax(x[:, :64], lut), jint.softmax(x[:, :64], lut))
        assert np.array_equal(I.make_lut("gelu", s, s / 2), jint.gelu_lut(s, s / 2))
    a, b = rng.integers(-128, 128, (2, 1000))
    for sa, sb, so in [(0.03, 0.01, 0.031), (0.1, 0.2, 0.05)]:
        ma, mb = I.ew_mult(sa, so), I.ew_mult(sb, so)
        assert (ma, mb) == (jint.ew_mult(sa, so), jint.ew_mult(sb, so))
        assert np.array_equal(I.ew_add(a, b, ma, mb), jint.ew_add(a, b, ma, mb))
    acc = rng.integers(-2**24, 2**24, (64, 32)); bias = rng.integers(-1000, 1000, 32)
    m0, sh = I.qmul_vec(10 ** rng.uniform(-5, -1, 32))
    assert np.array_equal(I.requant(acc, bias, m0, sh), jint.requant(acc, bias, m0, sh))
    w = rng.integers(-127, 128, (128, 32))
    assert np.array_equal(I.matmul(x, w), x @ w)


def test_zero_point_generalisation():
    rng = np.random.default_rng(3)
    x = rng.integers(-100, 100, (50, 64))
    # LayerNorm and softmax are invariant to a constant zero-point shift
    assert np.array_equal(I.layernorm(x), I.layernorm(x + 20))
    lut = I.exp_lut(0.05)
    assert np.array_equal(I.softmax(x, lut), I.softmax(x + 20, lut))
    # asymmetric add == symmetric add of the shifted operands
    a, b = rng.integers(-100, 100, (2, 200))
    assert np.array_equal(I.ew_add(a + 7, b - 3, 30000, 40000, 7, -3, 5),
                          np.clip(I.ew_add(a, b, 30000, 40000) + 5, -128, 127))
