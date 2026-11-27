"""End to end: the whole JEPA encoder compiled by tpucc, executed in tpuv1's ISA simulator
(sim.CoreSim, hazard checking on, DMA on) with host steps in between, compared bit for bit."""
import numpy as np
import pytest

from tpucc import codegen, runtime


def _check(run, lowered, ref, golden):
    out, log = golden
    assert np.array_equal(run.outputs["ln_out"], out)
    assert np.array_equal(run.outputs["ln_out"], ref["ln_out"])
    assert len(run.tpu) == 73
    for name, x, w, acc in log:                      # every encoder matmul + requant
        op = lowered.op_by_name(name)
        assert np.array_equal(run.tpu[name], ref[op.outputs[0]]), name


@pytest.mark.slow
def test_encoder_bit_exact_unmodified_tpuv1_sim(jepa_lowered, jepa_prog, jepa_img, jepa_ref, jepa_golden):
    run = runtime.run_program(jepa_prog, {"img": jepa_img}, fast=False, check_hazards=True)
    _check(run, jepa_lowered, jepa_ref, jepa_golden)
    assert run.kernels == len(jepa_prog.kernels) == 31
    assert run.instrs == jepa_prog.n_instrs
    assert run.sim.wpos == len(run.sim.wq)           # every weight word requested was consumed


def test_encoder_bit_exact_fast_sim(jepa_lowered, jepa_prog, jepa_img, jepa_ref, jepa_golden):
    run = runtime.run_program(jepa_prog, {"img": jepa_img}, fast=True)
    _check(run, jepa_lowered, jepa_ref, jepa_golden)


def test_host_ops_bit_exact(jepa_prog, jepa_img, jepa_ref):
    run = runtime.run_program(jepa_prog, {"img": jepa_img}, fast=True)
    for t in ("add0", "ln1_0", "sm3h2", "gelu5", "addb5"):
        assert np.array_equal(run.host[t], jepa_ref[t]), t


def test_encoder_without_dedupe_and_with_small_kernels(jepa_lowered, jepa_img, jepa_ref, jepa_golden):
    for kw in (dict(dedupe=False), dict(max_kernel=500), dict(reset_config_per_kernel=True)):
        prog = codegen.compile_graph(jepa_lowered, **kw)
        if "max_kernel" in kw:
            assert len(prog.kernels) > 31 and all(len(k.words) <= 500 for k in prog.kernels)
        run = runtime.run_program(prog, {"img": jepa_img}, fast=True)
        _check(run, jepa_lowered, jepa_ref, jepa_golden)


def test_second_image(P, jepa_lowered, jepa_prog):
    from tpucc import frontends, env
    img = frontends.synthetic_image(P, seed=7)
    out = env.jepa_quantize().int_forward(P, img)
    run = runtime.run_program(jepa_prog, {"img": img}, fast=True)
    assert np.array_equal(run.outputs["ln_out"], out)
