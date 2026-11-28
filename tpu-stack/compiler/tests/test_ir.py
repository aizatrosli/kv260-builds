import numpy as np
import pytest

from tpucc import env, frontends, refexec
from tpucc.ir import Graph
import compile_mlp as CM
import compile_cnn as CC


def test_jepa_graph_structure(jepa_graph, P):
    g = jepa_graph
    kinds = {}
    for o in g.ops:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    assert kinds["matmul"] == 73 and kinds["requant"] == 73
    assert kinds["layernorm"] == 13 and kinds["softmax"] == 24 and kinds["lut"] == 6 and kinds["add"] == 13
    assert kinds["slice"] == 72 and kinds["concat"] == 6
    assert [o.name for o in g.ops if o.kind == "matmul"] == frontends.jepa_matmul_names(P)
    assert g.validate()


def test_jepa_reference_matches_intops(jepa_graph, jepa_ref, jepa_golden):
    out, log = jepa_golden
    assert np.array_equal(jepa_ref["ln_out"], out)
    assert len(log) == 73
    I = env.jepa_modules()
    for name, x, w, acc in log:
        assert np.array_equal(jepa_ref[f"{name}.acc"], acc), name


def test_requant_matches_tpuv1_scalar():
    from tflite_int8 import requant
    rng = np.random.default_rng(0)
    acc = rng.integers(-2**31, 2**31, 2000)
    bias = rng.integers(-2**20, 2**20, 2000)
    m0 = rng.integers(2**30, 2**31, 2000)
    sh = rng.integers(-20, 5, 2000)
    got = refexec.requant(acc, bias, m0, sh, 7, -100, 120)
    exp = [requant(int(a), int(b), int(m), int(s), 7, -100, 120) for a, b, m, s in zip(acc, bias, m0, sh)]
    assert got.tolist() == exp


def test_mlp_frontend_matches_tpuv1():
    net = CM.Net([64, 128, 64, 16], ["relu", "relu6", "none"], seed=5)
    x = CM.quantize(np.random.default_rng(1).uniform(-1, 1, (8, 64)), net.in_scale, net.in_zp)
    g = frontends.from_mlp(net, 8)
    v = refexec.run(g, {"x": x})
    assert np.array_equal(v[g.outputs[0]], net.int_forward(x)[:, :16])


@pytest.mark.parametrize("case", ["cnn_tiny", "conv3x3_s2p1", "conv_pool_avg", "conv3x3_ct2_co32"])
def test_cnn_frontend_matches_tpuv1(case):
    spec, seed = CC.CASES[case]
    net = CC.Net(spec, seed)
    xq = _cnn_input(net, spec, seed)
    final, _ = net.int_forward(xq)
    g = frontends.from_cnn(net)
    v = refexec.run(g, {"x": xq[:, :, :spec["c"]].reshape(-1, spec["c"])})
    assert np.array_equal(v[g.outputs[0]], _cnn_gold(net, final))


def _cnn_input(net, spec, seed):
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(spec["lo"], spec["hi"], (spec["h"], spec["w"], spec["c"]))
    xq = np.full((spec["h"], spec["w"], CC.pad16(spec["c"])), net.in_zp, np.int64)
    xq[:, :, :spec["c"]] = CC.quantize(x, net.in_scale, net.in_zp)
    return xq


def _cnn_gold(net, final):
    L = net.layers[-1]
    return final[:L["nout"]][None] if L["kind"] == "fc" else final[:, :, :L["cout"]].reshape(-1, L["cout"])


def test_validate_rejects_bad_graphs():
    g = Graph()
    x = g.input("x", (4, 32))
    w = g.const("w", np.zeros((16, 16)))
    with pytest.raises(ValueError):
        g.op("matmul", "m", [x, w], (4, 16), "int32")
        g.validate()
    g2 = Graph()
    x = g2.input("x", (4, 40))
    g2.op("slice", "s", [x], (4, 8), c0=8, c1=16)
    with pytest.raises(ValueError):
        g2.validate()
