from tpuperf import config as C
from tpuperf import resources as RS
from tpuperf import workloads as W


def test_tpuv1_matches_measured_build():
    r = RS.estimate(C.tpuv1_board())
    assert abs(r.lut - 32_200) / 32_200 < 0.10
    assert abs(r.ff - 40_000) / 40_000 < 0.10
    assert r.dsp == 328
    assert r.uram == 32
    assert abs(r.bram36 - 44) <= 5
    assert r.feasible


def test_dsp_packing_halves_array_dsps():
    a = RS.estimate(C.proposed(32, 32, 200.0, dsp_packing=False, with_vpu=False))
    b = RS.estimate(C.proposed(32, 32, 200.0, dsp_packing=True, with_vpu=False))
    assert a.dsp - b.dsp == 512


def test_feasibility_flags():
    assert RS.estimate(C.proposed(32, 32, 200.0), W.jepa_tiny().param_bytes).feasible
    assert not RS.estimate(C.proposed(64, 64, 200.0)).feasible
    assert not RS.estimate(C.proposed(32, 32, 200.0, dsp_packing=False)).feasible  # VPU DSPs
    assert "INFEASIBLE" in RS.estimate(C.proposed(64, 64, 200.0)).status


def test_big_models_do_not_fit_on_chip():
    cfg = C.proposed(32, 32, 200.0)
    assert not RS.estimate(cfg, W.deit_small().param_bytes).feasible
    assert not RS.estimate(cfg, W.vit_tiny().param_bytes).feasible


def test_memory_primitives():
    assert RS.uram(65536, 128) == 32
    assert RS.bram36(1024, 128) == 4
    assert RS.bram36(2048, 512) == 29     # 2048 x 18 mode
