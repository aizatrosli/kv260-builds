"""The jepa-tpu encoder re-quantised with tpuquant (reads jepa-tpu, never writes it)."""
import os
import json
import numpy as np
import pytest
import torch

from tpuquant import QConfig, export
from tpuquant import jepa as J
from tpuquant.analysis import output_metrics, layer_report, weight_sensitivity
from tpuquant.int4 import hardware_report, unpack_int4

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(J.JEPA_DIR, "ckpt.pt")),
                                reason="jepa-tpu checkpoint not available")
NEVAL = 64


@pytest.fixture(scope="module")
def setup():
    enc = J.load_encoder()
    xtr, _, xte, _ = J.load_mnist()
    cal = J.calibrate_encoder(enc, xtr[:256])
    x = xte[:NEVAL]
    with torch.no_grad():
        ref = enc(x).double().numpy()
    return enc, cal, x.numpy(), ref


def test_bit_exact_vs_jepa_reference(setup):
    """Given jepa-tpu's calibrated scales, the toolkit emits the identical integer program."""
    enc, cal, x, ref = setup
    P = J.load_jepa_qmodel()
    jq = J.jepa_quantize_module()
    g = cal.build(QConfig(fixed_qparams=J.jepa_fixed_qparams(cal, P)))
    for n in g.weight_nodes():
        key = "embed" if n.name == "embed" else n.name.split("_")[2] + n.name.split("_")[1]
        for a in ("w", "bias", "m0", "shift"):
            assert np.array_equal(np.asarray(n.params[a], np.int64), np.asarray(P[key][a], np.int64)), (n.name, a)
    yi = g.run_int(x[:16])[g.outputs[0]]
    want = np.stack([jq.int_forward(P, jq.quant_img(P, torch.from_numpy(x[i:i + 1]))[0]) for i in range(16)])
    assert np.array_equal(yi, want)
    # and the GELU / exp LUTs are identical
    luts = [n.params["lut"] for n in g.nodes if n.op == "lut"]
    for l, lut in enumerate(luts):
        assert np.array_equal(lut.astype(np.int64), P[f"gelu{l}"])


def test_int8_reproduces_jepa_accuracy(setup):
    enc, cal, x, ref = setup
    m = output_metrics(ref, cal.build(QConfig(act_method="percentile")).run(x)[0])
    assert m["cos_mean"] >= 0.998 - 2e-4, m          # jepa-tpu reports 0.998 (min 0.997)
    assert m["cos_min"] > 0.996


def test_calibrated_softmax_beats_jepa(setup):
    enc, cal, x, ref = setup
    base = output_metrics(ref, cal.build(QConfig(act_method="mse")).run(x)[0])
    best = output_metrics(ref, cal.build(QConfig(act_method="mse", softmax_mult="mse")).run(x)[0])
    assert best["cos_mean"] > 0.999 and best["cos_mean"] > base["cos_mean"]
    assert best["cos_min"] > 0.998


def test_int4(setup):
    enc, cal, x, ref = setup
    g8 = cal.build(QConfig(act_method="mse"))
    g4 = cal.build(QConfig(act_method="mse", weight_bits=4, int4_scheme="group_int", weight_method="mse"))
    m4 = output_metrics(ref, g4.run(x)[0])
    assert 0.99 < m4["cos_mean"] < output_metrics(ref, g8.run(x)[0])["cos_mean"]
    rep = hardware_report(g8, g4)
    assert rep["int4_weight_kib"] < 0.55 * rep["int8_weight_kib"]
    n = g4.node("blocks_0_fc1")
    assert np.array_equal(n.params["w"].astype(np.int64),
                          n.params["w4"].astype(np.int64) * np.repeat(n.params["gmul"], 32, 0))


def test_export_jepa(setup, tmp_path):
    enc, cal, x, ref = setup
    g = cal.build(QConfig(act_method="mse", softmax_mult="mse", weight_bits=4,
                          layer_bits={"embed": 8}))
    man = export.save(g, tmp_path / "jepa")
    g2 = export.load(tmp_path / "jepa")
    a, b = g.run_int(x[:8]), g2.run_int(x[:8])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    m = json.load(open(tmp_path / "jepa" / "manifest.json"))
    assert m["format"] == "tpuquant-graph" and m["version"] == 1
    ops = {nd["op"] for nd in m["nodes"]}
    assert {"quantize", "linear", "matmul", "softmax", "layernorm", "lut", "add", "const"} <= ops
    z = np.load(tmp_path / "jepa" / "weights.npz")
    nd = [nd for nd in m["nodes"] if nd["name"] == "blocks_0_qkv"][0]
    w4 = z[nd["params"]["w4"]["key"]]
    assert np.array_equal(unpack_int4(z[nd["params"]["w4_packed"]["key"]], w4.shape[-1]), w4)
    assert m["outputs"][0]["post"] is not None             # final LN affine applied on the host


def test_reports_jepa(setup):
    enc, cal, x, ref = setup
    g = cal.build(QConfig())
    rows = layer_report(g, x[:8])
    assert rows[-1]["cos"] > 0.99
    ws = weight_sensitivity(g, x[:4])
    assert len(ws) == 25 and ws[0]["sqnr_db"] < ws[-1]["sqnr_db"]
