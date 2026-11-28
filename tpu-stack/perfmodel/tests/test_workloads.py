import pytest

from tpuperf import workloads as W


def test_jepa_shape():
    j = W.jepa_tiny()
    assert j.macs == 81_920_000
    assert j.n_matmuls == 73
    assert j.param_bytes == pytest.approx(1.18e6, rel=0.02)


def test_mlp_benchmarks():
    assert W.mlp_wide().macs == 256 * 256 * 100
    assert W.mlp3().macs == 32 * (64 * 128 + 128 * 64 + 64 * 16)
    assert W.digits(128).macs == 128 * (64 * 128 + 128 * 64 + 64 * 10)


@pytest.mark.parametrize("name,params,gmacs", [
    ("vit_tiny", 5.7e6, 1.26), ("deit_small", 22e6, 4.6), ("ijepa_vits14", 21.5e6, 6.1),
    ("resnet18", 11.7e6, 1.82), ("mobilenet_v2", 3.5e6, 0.30),
])
def test_zoo_sizes(name, params, gmacs):
    w = W.get(name)
    assert w.param_bytes == pytest.approx(params, rel=0.06)
    assert w.macs / 1e9 == pytest.approx(gmacs, rel=0.06)


def test_batch_scales_macs():
    assert W.vit_tiny(batch=4).macs == pytest.approx(4 * W.vit_tiny().macs, rel=1e-9)


def test_conv_k_tiles():
    g = W.conv("c", 56, 64, 64, 3, 1)
    assert g.k_tiles(16, 16) == 9 * 4
    dw = W.conv("d", 56, 96, 96, 3, 1, depthwise=True)
    assert dw.k_tiles(16, 16) == 9
    assert dw.macs == 56 * 56 * 9 * 96
    stem = W.conv("s", 224, 3, 64, 7, 2, host_im2col=True)
    assert stem.k_tiles(16, 16) == 10
