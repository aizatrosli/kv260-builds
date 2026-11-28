import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tpuquant import Calibrator, QConfig, export
from tpuquant.analysis import output_metrics, layer_report, act_sensitivity, weight_sensitivity


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool(x)
        x = torch.relu(self.c2(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


class TinyViT(nn.Module):
    """Pre-LN transformer block with explicit attention, residuals, GELU, mean-pool head."""
    def __init__(self, d=32, h=2, t=16):
        super().__init__()
        self.h = h
        self.inp = nn.Linear(8, d)
        self.pos = nn.Parameter(torch.randn(1, t, d) * 0.1)
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 5)

    def forward(self, x):
        x = self.inp(x) + self.pos
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D // self.h))
        o = (a.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(o)
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return self.head(self.ln(x).mean(1))


def _data(shape, n=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n,) + shape, generator=g)


@pytest.fixture(scope="module")
def cnn():
    torch.manual_seed(0)
    m = SmallCNN().eval()
    return m, Calibrator(m, _data((1, 12, 12)))


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    m = TinyViT().eval()
    return m, Calibrator(m, _data((16, 8)))


def _ref(m, x):
    with torch.no_grad():
        return m(x).double().numpy()


@pytest.mark.parametrize("fixture", ["cnn", "vit"])
def test_float_mode_matches_torch(fixture, request):
    m, cal = request.getfixturevalue(fixture)
    g = cal.build(QConfig())
    x = _data(cal.calib[0].shape[1:], 8, seed=1)
    y = g.float_outputs(g.run_float(x.numpy()))[0]
    assert np.allclose(y, _ref(m, x), atol=1e-4)


@pytest.mark.parametrize("fixture", ["cnn", "vit"])
@pytest.mark.parametrize("method", ["minmax", "percentile", "mse"])
@pytest.mark.parametrize("sym", [True, False])
def test_int_graph_accuracy(fixture, method, sym, request):
    m, cal = request.getfixturevalue(fixture)
    g = cal.build(QConfig(act_method=method, act_symmetric=sym))
    x = _data(cal.calib[0].shape[1:], 32, seed=2)
    met = output_metrics(_ref(m, x), g.run(x.numpy())[0])
    assert met["cos_mean"] > 0.98, met
    # integer and fake-quant interpreters agree closely
    fq = g.float_outputs(g.run_float(x.numpy(), quant_acts="all", quant_weights="all"))[0]
    assert output_metrics(fq, g.run(x.numpy())[0])["cos_mean"] > 0.99


def test_cnn_structure(cnn):
    m, cal = cnn
    g = cal.build(QConfig(act_symmetric=False))
    ops = [n.op for n in g.nodes]
    assert ops.count("conv2d") == 2 and "maxpool2d" in ops and "mean" in ops and "lut" not in ops
    c1 = g.node("c1")
    assert c1.attrs["act_min"] == c1.attrs["zp_out"]          # ReLU fused as TFLite clamp
    assert c1.params["w"].shape == (3, 3, 1, 8)                 # HWIO
    assert c1.params["shift"].dtype == np.int8
    assert -31 <= int(c1.params["shift"].min()) and int(c1.params["shift"].max()) <= 30


def test_vit_structure(vit):
    m, cal = vit
    g = cal.build(QConfig(act_symmetric=False))
    ops = [n.op for n in g.nodes]
    for op in ("layernorm", "softmax", "matmul", "add", "lut", "const", "affine", "mean"):
        assert op in ops, op
    # LN1/LN2 affines folded into qkv/fc1; the final LN feeds mean() so it keeps an affine op
    assert ops.count("affine") == 1
    for n in g.nodes:
        if n.op == "matmul":
            assert all(g.tensors[i].zp == 0 for i in n.inputs)   # act x act operands symmetric
    qkv = g.tensors[g.node("qkv").output]
    assert not qkv.uniform and qkv.scale.shape[-1] == 96        # split q/k/v scales
    assert 1.0 / math.sqrt(16) == pytest.approx(g.node("matmul").meta["post_scale"])


def test_int4_and_mixed(vit):
    m, cal = vit
    x = _data((16, 8), 32, seed=3)
    ref = _ref(m, x)
    c8 = output_metrics(ref, cal.build(QConfig()).run(x.numpy())[0])["cos_mean"]
    g4 = cal.build(QConfig(weight_bits=4, int4_scheme="group_int", group_size=16, weight_method="mse"))
    c4 = output_metrics(ref, g4.run(x.numpy())[0])["cos_mean"]
    assert c4 < c8 and c4 > 0.9
    assert g4.weight_bytes() < 0.6 * cal.build(QConfig()).weight_bytes()
    g = cal.build(QConfig(weight_bits=4, layer_bits={"head": 8}))
    assert g.node("head").attrs["weight_bits"] == 8 and g.node("fc1").attrs["weight_bits"] == 4


def test_export_roundtrip(vit, tmp_path):
    m, cal = vit
    x = _data((16, 8), 8, seed=4).numpy()
    for cfg in (QConfig(act_symmetric=False), QConfig(weight_bits=4, group_size=16)):
        g = cal.build(cfg)
        export.save(g, tmp_path / "e")
        g2 = export.load(tmp_path / "e")
        a, b = g.run_int(x), g2.run_int(x)
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert np.allclose(g.run(x)[0], g2.run(x)[0])


def test_reports(vit):
    m, cal = vit
    g = cal.build(QConfig())
    x = _data((16, 8), 8, seed=5).numpy()
    rows = layer_report(g, x)
    assert len(rows) > 10 and all(np.isfinite(r["cos"]) for r in rows)
    s = act_sensitivity(g, x)
    assert s[0]["sqnr_db"] <= s[-1]["sqnr_db"]
    w = weight_sensitivity(g, x, bits=4, group_size=16)
    assert {r["name"] for r in w} == {n.name for n in g.weight_nodes()}


def test_unsupported_op_raises():
    class Bad(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(4, 4)
        def forward(self, x):
            return torch.exp(self.fc(x))
    cal = Calibrator(Bad(), torch.randn(8, 4))
    with pytest.raises(NotImplementedError, match="exp"):
        cal.build()
