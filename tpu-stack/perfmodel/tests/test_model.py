import pytest

from tpuperf import config as C
from tpuperf import workloads as W
from tpuperf.model import simulate, gemm_timing, vpu_job_cycles


def one(M, K, N, **kw):
    return W.Workload("g", [W.Gemm("g", M, K, N, **kw)])


ROM = C.tpuv1_sim_rom()


def test_lower_bound_and_util():
    for w in (W.jepa_tiny(), W.mlp_wide(), W.vit_tiny()):
        r = simulate(w, ROM)
        assert r.matmul_cycles >= w.macs / ROM.macs_per_cycle
        assert 0 < r.util <= 1.0


def test_shadow_guard_limits_tile_period():
    # B = 16 rows < 3N+1 = 49: one tile per 49 cycles in steady state
    cfg = ROM.with_(cfg_mode="param_rom", act_blocks_cfg=False, sync_per_tile=False)
    g = W.Gemm("g", 16, 16 * 64, 16)          # 64 K tiles, one column tile
    t = gemm_timing(cfg, g)
    period = t.cycles / t.tiles
    assert 47 <= period <= 52
    # two shadow banks: a bank is rewritten only after the tile two back has swapped,
    # so the period is (guard + 1 + load) / 2 = 24.5
    t2 = gemm_timing(cfg.with_(weight_buffering="double"), g)
    assert t2.cycles / t2.tiles == pytest.approx(24.5, abs=1.5)


def test_double_buffer_irrelevant_when_rows_cover_guard():
    g = W.Gemm("g", 128, 256, 256)
    a = gemm_timing(ROM, g).cycles
    b = gemm_timing(ROM.with_(weight_buffering="double"), g).cycles
    assert a == pytest.approx(b, rel=0.01)


def test_streamlined_control_approaches_peak():
    cfg = ROM.with_(cfg_mode="param_rom", act_blocks_cfg=False, sync_per_tile=False,
                    weight_buffering="double")
    r = simulate(one(1024, 1024, 1024), cfg)
    assert r.util > 0.97


def test_ddr_latency_stalls_small_batches_prefetch_hides():
    w = W.mlp3()
    ddr = simulate(w, C.tpuv1_board())
    pre = simulate(w, C.tpuv1_board().with_(weight_source="ddr_prefetch"))
    assert ddr.weight_stall_cycles > pre.weight_stall_cycles
    assert pre.total_us < ddr.total_us


def test_axi_push_slower_than_rom():
    j = W.jepa_tiny()
    assert simulate(j, C.tpuv1_sim_axi()).matmul_us > simulate(j, ROM).matmul_us
    assert simulate(j, ROM).weight_stall_cycles == 0


def test_clock_scaling():
    w = W.vit_tiny()
    cfg = C.proposed(32, 32, 100.0, weight_source="ddr_prefetch")
    t100 = simulate(w, cfg).total_us
    t200 = simulate(w, cfg.with_(clock_mhz=200.0)).total_us
    assert 1.8 < t100 / t200 <= 2.05


def test_acc_capacity_chunks_m():
    g = W.Gemm("g", 4096, 64, 64)
    small = gemm_timing(ROM.with_(acc_rows=512), g)
    big = gemm_timing(ROM.with_(acc_rows=8192), g)
    assert small.tiles == 4096 // 256 * 4 * 4
    assert big.tiles == 16
    assert small.cycles > big.cycles


def test_ub_spill_costs_time():
    g = W.Gemm("g", 1024, 1024, 1024)      # 2 MiB working set
    cfg = C.tpuv1_board()
    assert gemm_timing(cfg.with_(ub_kib=256), g).spill_cycles > 0
    assert gemm_timing(cfg.with_(ub_kib=4096), g).spill_cycles == 0


def test_int4_halves_weight_traffic():
    w = W.deit_small()
    cfg = C.proposed(32, 32, 200.0, weight_source="ddr_prefetch", ub_kib=1024)
    a = simulate(w, cfg).weight_bytes_moved
    b = simulate(w, cfg.with_(weight_bits=4)).weight_bytes_moved
    assert b == pytest.approx(a / 2, rel=1e-6)


def test_vpu_fsm_cycles():
    assert vpu_job_cycles("add", 8) == 17
    assert vpu_job_cycles("gelu", 32) == 65
    assert vpu_job_cycles("ln", 8) == 1 + 16 + 22 + 30
    assert vpu_job_cycles("softmax", 4) == 1 + 8 + 4 + 30
    assert vpu_job_cycles("ln", 4, pipelined=True) == 52


def test_vpu_jepa_job_counts():
    j = W.jepa_tiny()
    jobs = {}
    rows = 0
    for v in j.vecops:
        jobs[v.kind] = jobs.get(v.kind, 0) + v.jobs
        rows += v.jobs * (v.elems // 16)
    assert jobs == {"add": 832, "ln": 832, "softmax": 1536, "gelu": 384}
    assert rows == 31_744          # jepa-tpu README


def test_gelu_in_act_and_overlap():
    j = W.jepa_tiny()
    base = simulate(j, ROM)
    fused = simulate(j, ROM.with_(gelu_in_act=True))
    assert fused.vpu_cycles < base.vpu_cycles
    ov = simulate(j, ROM.with_(vpu_overlap=0.8))
    assert ov.total_us < base.total_us
