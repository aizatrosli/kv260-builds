"""End-to-end: the JEPA encoder on the TPU backends is bit-exact with jepa-tpu's quantize.int_forward."""
import numpy as np
import pytest

from tpurt import get_backend, data
from tpurt.backends.raw import FakeFabric, FakeRegion, RawBackend
from tpurt.jepa import JepaExecutor


def _exact(y, golden, idx):
    return [bool((y[k] == golden[i]).all()) for k, i in enumerate(idx)]


def test_numpy_reference_equals_quantize(P, images, golden):
    for i in range(3):
        assert (data.reference_forward(P, images[i]) == golden[i]).all()


def test_quant_img_equals_quantize(P):
    import torch  # noqa: F401
    import quantize
    imgs, _ = data.load_images(3)
    t = torch.from_numpy(imgs.astype(np.float32))[:, None]
    for i in range(3):
        assert (data.quant_img(P, imgs[i:i + 1])[0] == quantize.quant_img(P, t[i:i + 1])[0]).all()


def test_sim_host_vectors_batch1(P, images, golden):
    ex = JepaExecutor(get_backend("sim"), P, batch=1)
    y = ex.forward_q(images[:4])
    assert _exact(y, golden, range(4)) == [True] * 4
    s = ex.prof.summary()
    assert s["runs_per_image"] == 31 and s["images"] == 4
    assert ex.consts.loads == 25                       # weights loaded once, reused for every image
    ex.close()


def test_sim_batched_pipelined_with_padding(P, images, golden):
    ex = JepaExecutor(get_backend("sim"), P, batch=3, contexts=2)
    y = ex.run_many(images[:7], pipeline=True)          # 3 + 3 + 1 (padded)
    assert y.shape == (7, 64, 128)
    assert _exact(y, golden, range(7)) == [True] * 7
    ex.close()


def test_sim_numpy_host_ops(P, images, golden):
    ex = JepaExecutor(get_backend("sim"), P, batch=2, ops="numpy")
    assert ex.ops.name == "numpy"
    assert _exact(ex.forward_q(images[2:4]), golden, [2, 3]) == [True, True]


@pytest.mark.parametrize("batch", [1, 4])
def test_sim_vpu_offload(P, images, golden, batch):
    ex = JepaExecutor(get_backend("sim"), P, batch=batch, vector="vpu")
    host_steps = [it.name for k, it in ex.items if k == "host"]
    assert all(n.startswith(("input", "pack_kv", "output")) for n in host_steps)   # only K/V packing left
    assert len(ex.programs) < 31
    assert _exact(ex.forward_q(images[:batch]), golden, range(batch)) == [True] * batch
    ex.close()


def test_vpu_rejected_without_vpu(P):
    with pytest.raises(RuntimeError):
        JepaExecutor(get_backend("sim", vpu=False), P, vector="vpu")


def test_mock_driver_path(P, images, golden):
    """Through tpuv1's tpu.Device + tpu_mock (ioctl ABI model, relocation, validation)."""
    ex = JepaExecutor(get_backend("mock"), P, batch=1)
    assert _exact(ex.forward_q(images[5:6]), golden, [5]) == [True]
    ex.close()


@pytest.mark.slow
def test_mock_vpu_batch2(P, images, golden):
    ex = JepaExecutor(get_backend("mock"), P, batch=2, vector="vpu")
    assert _exact(ex.forward_q(images[6:8]), golden, [6, 7]) == [True, True]
    ex.close()


def test_raw_fake_fabric(P, images, golden):
    reg = FakeRegion(32 << 20)
    ex = JepaExecutor(RawBackend(regs=FakeFabric(reg), region=reg), P, batch=2)
    assert _exact(ex.forward_q(images[:2]), golden, [0, 1]) == [True, True]
    assert reg.flushes > 0 and reg.invalidates > 0      # cache maintenance on the non-coherent path
    ex.close()


def test_programs_fit_and_profile(P):
    for vec in ("host", "vpu"):
        for b in (1, 16):
            ex = JepaExecutor(get_backend("sim"), P, batch=b, vector=vec)
            assert all(p.n_instr <= 2048 for p in ex.programs)
            assert ex.contexts[0].planned_bytes < ex.contexts[0].unshared_bytes
            ex.close()


@pytest.mark.board
def test_board_jepa(P, images, golden):
    ex = JepaExecutor(get_backend("board"), P, batch=1)
    assert _exact(ex.run_many(images[:4]), golden, range(4)) == [True] * 4
    print(ex.prof.report())
