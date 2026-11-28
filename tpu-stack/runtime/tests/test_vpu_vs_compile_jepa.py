"""Cross-check tpurt.vpu against the RTL thread's reference (jepa-tpu/compile_jepa.py on branch
jepa-e2e-optimize): identical field table, identical words, and identical X_VEC/X_LUT/X_VCFG results
between its bit-exact `Machine` and tpurt.fastsim.

Point TPURT_COMPILE_JEPA at that file to run it, e.g.
  git show origin/jepa-e2e-optimize:jepa-tpu/compile_jepa.py > /tmp/compile_jepa.py
  TPURT_COMPILE_JEPA=/tmp/compile_jepa.py python -m pytest tests/test_vpu_vs_compile_jepa.py
"""
import importlib.util
import os
import random
import sys
import types

import numpy as np
import pytest

from tpurt import _paths, fastsim, vpu

SRC = os.environ.get("TPURT_COMPILE_JEPA")
pytestmark = pytest.mark.skipif(not SRC or not os.path.exists(SRC), reason="set TPURT_COMPILE_JEPA to compile_jepa.py")


@pytest.fixture(scope="module")
def cj():
    _paths.need_tpuv1_model()
    _paths.need_jepa()
    os.environ.setdefault("TPU", str(_paths.TPUV1))
    spec = importlib.util.spec_from_file_location("compile_jepa_ref", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_field_table_identical(cj):
    assert vpu.XF == cj.XF
    assert (vpu.X_VEC, vpu.X_LUT, vpu.X_VCFG) == (cj.X_VEC, cj.X_LUT, cj.X_VCFG)
    assert (vpu.V_ADD, vpu.V_LUT, vpu.V_LN, vpu.V_SM) == (cj.V_ADD, cj.V_LUT, cj.V_LN, cj.V_SM)


def test_words_identical(cj):
    rnd = random.Random(1)
    for opc, fields in vpu.XF.items():
        for _ in range(200):
            kw = {}
            for k, (_, w) in fields.items():
                kw[k] = rnd.randrange(-(1 << (w - 1)), 1 << (w - 1)) if k in vpu.SIGNED else rnd.randrange(1 << w)
            assert vpu.xenc(opc, **kw) == cj.xenc(opc, **kw)
            assert vpu.xdec(cj.xenc(opc, **kw)) == cj.xdec(cj.xenc(opc, **kw))


def test_machine_vs_fastsim(cj):
    import asm
    rng = np.random.default_rng(0)
    from tpurt.jepa import load_qmodel
    P = load_qmodel()
    depth = P["cfg"]["depth"]
    rom = vpu.jepa_lut_rom(P, depth)
    lrom = [int(x, 16) for x in vpu.lrom_hex(rom).split()]    # what the bitstream's lrom.hex must hold
    ub = [[int(v) for v in r] for r in rng.integers(-128, 128, (4096, 16))]
    ma, mb = P["adda0"]
    prog = [vpu.vcfg(int(ma), int(mb)), vpu.lut(vpu.LUT_EXP, vpu.lut_idx(vpu.LUT_EXP, 2, depth)),
            vpu.lut(vpu.LUT_GELU, vpu.lut_idx(vpu.LUT_GELU, 4, depth)),
            vpu.vec(vpu.V_ADD, 8, 0, 2000, 64, 64, b=512),
            vpu.vec(vpu.V_LN, 8, 64, 2600, 32, 128, 2),
            vpu.vcfg(int(ma), int(mb), 263), vpu.vec(vpu.V_SM, 4, 1024, 3200, 64, 64),
            vpu.vec(vpu.V_LUT, 5, 1300, 3500, 40, 80),
            asm.interrupt_host(1), asm.halt()]
    cp = types.SimpleNamespace(prog=prog, wrom=[], prom=[], lrom=lrom)
    ref = cj.Machine(cp, ub).run()
    m = fastsim.FastCoreSim(ub, vpu_enabled=True, lut_rom=rom).run(prog)
    assert not m.cfg_err and m.halted
    got = m.ubA.astype(np.int64)
    exp = np.array([ref.core.ub[r] for r in range(4096)], np.int64)
    assert (got[:4096] == exp).all()
