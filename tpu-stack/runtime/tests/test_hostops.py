"""Host kernels (C and numpy) vs jepa-tpu/intops.py, on the tiled layout."""
import numpy as np
import pytest

import intops as I
from tpurt import hostops as H

OPS = [H.NumpyOps()]
try:
    OPS.append(H.get_ops("c"))
except RuntimeError:
    pass


def t8(X):
    return H.tile(np.asarray(X)).astype(np.int8)


@pytest.fixture(params=OPS, ids=lambda o: o.name)
def ops(request):
    return request.param


def test_c_kernels_available():
    assert any(o.name == "c" for o in OPS), "C host kernels failed to build (gcc + make needed)"


def test_tile_roundtrip():
    X = np.random.default_rng(0).integers(-128, 128, (64, 128))
    assert (H.untile(H.tile(X)) == X).all()
    assert (H.tile(X)[3, 5] == X[5, 48:64]).all()


@pytest.mark.parametrize("M", [1, 64, 300])
def test_layernorm(ops, M):
    rng = np.random.default_rng(M)
    for X in (rng.integers(-128, 128, (M, 128)), rng.integers(-3, 4, (M, 128)), np.full((M, 128), 7),
              np.where(rng.random((M, 128)) < 0.5, -128, 127), rng.integers(-128, 128, (M, 512))):
        assert (H.untile(ops.layernorm(t8(X))) == I.layernorm(X)).all()


def test_softmax(ops, P):
    rng = np.random.default_rng(1)
    for l in range(6):
        lut = P[f"exp{l}"]
        S = np.concatenate([rng.integers(-128, 128, (5, 64, 64)), np.full((1, 64, 64), -128),
                            np.full((1, 64, 64), 127), rng.integers(100, 128, (1, 64, 64))])
        st = np.stack([t8(s) for s in S])
        got = ops.softmax(st, lut)
        assert all((H.untile(got[g]) == I.softmax(S[g], lut)).all() for g in range(len(S)))


def test_gelu_lut(ops, P):
    X = np.arange(-128, 128).repeat(4).reshape(64, 16)
    for l in range(6):
        assert (H.untile(ops.lut8(t8(X), P[f"gelu{l}"])) == I.lut8(X, P[f"gelu{l}"])).all()


def test_residual(ops, P):
    rng = np.random.default_rng(2)
    A, B = rng.integers(-128, 128, (128, 128)), rng.integers(-128, 128, (128, 128))
    for k in [k for k in P if k.startswith("add")]:
        ma, mb = P[k]
        assert (H.untile(ops.ew_add(t8(A), t8(B), ma, mb)) == I.ew_add(A, B, ma, mb)).all()
    # broadcast (positional embedding repeated per image), and the int64 path for huge multipliers
    pos = rng.integers(-128, 128, (64, 128))
    got = ops.ew_add(t8(A), np.tile(t8(pos), (1, 2, 1)), *P["add0"])
    assert (H.untile(got) == I.ew_add(A, np.tile(pos, (2, 1)), *P["add0"])).all()
    assert (H.untile(ops.ew_add(t8(A), t8(B), 1 << 26, -(1 << 25))) == I.ew_add(A, B, 1 << 26, -(1 << 25))).all()


def test_out_param_writes_in_place(ops):
    X = np.random.default_rng(3).integers(-128, 128, (64, 128))
    out = np.zeros((8, 64, 16), np.int8)
    r = ops.layernorm(t8(X), out=out)
    assert r is out and (H.untile(out) == I.layernorm(X)).all()


@pytest.mark.parametrize("B", [1, 3])
def test_pack_kv(ops, B):
    """kT / v weight tiles == tpuv1 weight-tile order of k^T and v (definition via pack_weight_tiles)."""
    from tpurt.program import pack_weight_tiles
    rng = np.random.default_rng(B)
    QKV = rng.integers(-128, 128, (64 * B, 384))
    kT, v = ops.pack_kv(t8(QKV), B, 4)
    for b in range(B):
        for h in range(4):
            k = QKV[b * 64:(b + 1) * 64, 128 + 32 * h:128 + 32 * h + 32]
            vv = QKV[b * 64:(b + 1) * 64, 256 + 32 * h:256 + 32 * h + 32]
            assert kT[b * 4 + h].tobytes() == pack_weight_tiles(k.T)
            assert v[b * 4 + h].tobytes() == pack_weight_tiles(vv)
