"""tpuv1's own MLP and CNN test networks through tpucc, bit-exact in the unmodified ISA sim."""
import numpy as np
import pytest

from tpucc import frontends, passes, codegen, runtime
import compile_mlp as CM
import compile_cnn as CC


def _run(g, feeds, **kw):
    prog = codegen.compile_graph(passes.lower(g), **kw)
    return prog, runtime.run_program(prog, feeds, fast=False)


@pytest.mark.parametrize("dims,acts,B", [
    ([16, 16], ["none"], 16),
    ([64, 128, 64, 16], ["relu", "relu", "none"], 32),
    ([64, 128, 64, 16], ["relu", "relu", "none"], 1),
    ([256, 256], ["relu6"], 100),
    ([20, 50, 10], ["relu", "none"], 7),                 # ragged K / N (padding lanes)
])
def test_mlp(dims, acts, B):
    net = CM.Net(dims, acts, seed=33)
    x = CM.quantize(np.random.default_rng(2).uniform(-1, 1, (B, dims[0])), net.in_scale, net.in_zp)
    xp = np.full((B, CM.pad16(dims[0])), net.in_zp, np.int64)
    xp[:, :dims[0]] = x
    g = frontends.from_mlp(net, B)
    prog, r = _run(g, {"x": x})
    assert len(prog.kernels) == 1                      # whole MLP stays on chip
    assert np.array_equal(r.outputs[g.outputs[0]], net.int_forward(xp)[:, :dims[-1]])


def test_mlp_m_blocking():
    net = CM.Net([16, 32], ["relu"], seed=3)
    B = 1100                                           # > one 1024-row ACC slot
    x = CM.quantize(np.random.default_rng(4).uniform(-1, 1, (B, 16)), net.in_scale, net.in_zp)
    g = frontends.from_mlp(net, B)
    prog, r = _run(g, {"x": x})
    assert prog.stats["fc0"].weight_tiles == 2 * 2
    assert np.array_equal(r.outputs[g.outputs[0]], net.int_forward(x))


def test_mlp_kernel_split():
    net = CM.Net([64, 128, 64, 16], ["relu", "relu", "none"], seed=33)
    x = CM.quantize(np.random.default_rng(2).uniform(-1, 1, (8, 64)), net.in_scale, net.in_zp)
    g = frontends.from_mlp(net, 8)
    prog, r = _run(g, {"x": x}, max_kernel=60)
    assert len(prog.kernels) > 5 and all(len(k.words) <= 60 for k in prog.kernels)
    assert np.array_equal(r.outputs[g.outputs[0]], net.int_forward(x)[:, :16])


def _cnn(spec, seed):
    net = CC.Net(spec, seed)
    rng = np.random.default_rng(seed + 1)
    xf = rng.uniform(spec["lo"], spec["hi"], (spec["h"], spec["w"], spec["c"]))
    xq = np.full((spec["h"], spec["w"], CC.pad16(spec["c"])), net.in_zp, np.int64)
    xq[:, :, :spec["c"]] = CC.quantize(xf, net.in_scale, net.in_zp)
    final, _ = net.int_forward(xq)
    L = net.layers[-1]
    gold = final[:L["nout"]][None] if L["kind"] == "fc" else final[:, :, :L["cout"]].reshape(-1, L["cout"])
    return net, xq[:, :, :spec["c"]].reshape(-1, spec["c"]), gold


@pytest.mark.parametrize("case", sorted(CC.CASES))
def test_cnn_tpuv1_cases(case):
    spec, seed = CC.CASES[case]
    net, x, gold = _cnn(spec, seed)
    g = frontends.from_cnn(net)
    _, r = _run(g, {"x": x})
    assert np.array_equal(r.outputs[g.outputs[0]], gold)


def test_cnn_multi_window():
    spec = dict(h=40, w=40, c=3, lo=-0.5, hi=1.0, layers=[CC.C(16, pool="max"), CC.C(16, stride=2)])
    net, x, gold = _cnn(spec, 77)
    g = frontends.from_cnn(net)
    prog, r = _run(g, {"x": x})
    assert "windows=2" in prog.stats["l0"].plan
    assert np.array_equal(r.outputs[g.outputs[0]], gold)
