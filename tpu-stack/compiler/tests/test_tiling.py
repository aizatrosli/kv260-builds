import pytest

from tpucc import tiling


def test_plan_small_layer_is_weight_stationary():
    p = tiling.plan_matmul(64, 128, 384)
    assert p.order == "nk" and p.KT == 8 and p.NT == 24
    assert p.mblocks == [(0, 64)]
    assert p.weight_tile_loads == 192 and p.weight_bytes == 192 * 256
    assert p.alternatives["kn"] is not None and p.alternatives["kn"] >= p.alternatives["nk"]


def test_plan_m_blocking_when_m_exceeds_acc_slot():
    p = tiling.plan_matmul(3000, 32, 32)
    assert p.mblocks == [(0, 1024), (1024, 1024), (2048, 952)]
    assert p.weight_tile_loads == 2 * 2 * 3                       # tiles re-streamed per block
    assert p.order == "nk"


def test_plan_kn_infeasible_for_wide_layers():
    p = tiling.plan_matmul(512, 64, 512)
    assert p.alternatives["kn"] is None and p.order == "nk"


def test_plan_conv_windows():
    p = tiling.plan_conv(40, 40, 3, 1, 1)
    assert sum(c for _, c in p.windows) == 40 and all(c * 40 <= 1024 for _, c in p.windows)
    q = tiling.plan_conv(40, 40, 3, 1, 1, pool="max")
    assert all(o % 2 == 0 and c % 2 == 0 for o, c in q.windows)
    assert tiling.plan_conv(8, 8, 3, 1, 2).windows == [(0, 8)]
