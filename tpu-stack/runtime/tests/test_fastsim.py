"""tpurt.fastsim.FastCoreSim must be indistinguishable from tpuv1's CoreSim (UB, ACC, cfg_err, hazards)."""
import random

import numpy as np
import pytest

from tpurt import _paths, fastsim, vpu
from tpurt.hostops import NumpyOps, tile, untile

_paths.need_tpuv1_model()
import asm                     # noqa: E402
import sim as isasim           # noqa: E402


def _both(prog, ub=None, ww=None, **kw):
    out = []
    for cls in (isasim.CoreSim, fastsim.FastCoreSim):
        try:
            m = cls(ub, ww, **kw).run(prog)
            out.append(("ok", m.ub, m.acc, m.cfg_err, m.cfg_err_op, m.halted, m.irq_count, m.irq_tag, m.wpos,
                        m.instr_count))
        except (isasim.HazardError, RuntimeError) as e:
            out.append((type(e).__name__, str(e).split(":")[0]))
    return out


def test_compiled_mlps_identical():
    import sys
    sys.path.insert(0, str(_paths.TPUV1 / 'model/mlp'))
    import compile_mlp as cm
    for seed, dims, acts, B in ((11, [16, 16], ["none"], 16), (33, [64, 128, 64, 16], ["relu", "relu", "none"], 32),
                                (55, [256, 256], ["relu6"], 100), (7, [48, 80, 20], ["relu", "none"], 5)):
        net = cm.Net(dims, acts, seed)
        rng = np.random.default_rng(seed)
        xq = rng.integers(-128, 128, (B, cm.pad16(dims[0])))
        prog, out_first, ub_end, _ = cm.emit_program(net, B)
        ub = [[0] * 16 for _ in range(ub_end)]
        for kt in range(cm.pad16(dims[0]) // 16):
            for b in range(B):
                ub[kt * B + b] = [int(v) for v in xq[b, 16 * kt:16 * kt + 16]]
        a, b = _both(prog, ub, cm.weight_words(net))
        assert a == b and a[0] == "ok"


def test_random_programs_identical_incl_hazards_and_cfg_err():
    rnd = random.Random(5)
    for trial in range(150):
        ub = [[rnd.randrange(-128, 128) for _ in range(16)] for _ in range(64)]
        ww = [rnd.getrandbits(128) for _ in range(16 * 6)]
        prog = [asm.matmul(0, 0, 48, acc=0, new_weights=1), asm.sync(asm.U_ALL)]    # defined ACC rows
        for _ in range(rnd.randrange(3, 25)):
            k = rnd.random()
            if k < 0.3:
                prog.append(asm.matmul(rnd.randrange(0, 48), rnd.randrange(0, 40), rnd.randrange(1, 16),
                                       acc=rnd.randrange(2), new_weights=rnd.randrange(2)))
            elif k < 0.5:
                prog.append(asm.activate(rnd.randrange(0, 40), rnd.randrange(0, 60), rnd.randrange(1, 8),
                                         rnd.randrange(4), rnd.randrange(3)))
            elif k < 0.75:
                q, c = rnd.randrange(4), rnd.randrange(16)
                prog.append(rnd.choice([asm.set_config(0x1000 | q << 4 | c, rnd.getrandbits(32) | ((rnd.randrange(-31, 31) & 0xFF) << 32)),
                                        asm.set_config(0x2000 | q << 4 | c, rnd.getrandbits(32)),
                                        asm.set_config(0x3000 | q, rnd.getrandbits(24))]))
            elif k < 0.95:
                prog.append(asm.sync(rnd.randrange(16)))
            else:
                prog.append(rnd.choice([asm.encode_raw(0x7F), asm.matmul(0, 2000, 100), asm.halt()]))
        prog += [asm.interrupt_host(trial), asm.halt()]
        for hz in (True, False):
            a, b = _both(prog, ub, ww, check_hazards=hz)
            assert a == b, (trial, hz, a[0], b[0])


def test_vector_opcode_semantics(P):
    """X_VEC / X_LUT / X_VCFG == intops on gathered vectors (all four functions + LUT-ROM loads)."""
    import intops as I
    rng = np.random.default_rng(0)
    M = 64
    X = rng.integers(-128, 128, (M, 128))
    Y = rng.integers(-128, 128, (M, 128))
    S = rng.integers(-128, 128, (M, 64))
    G = rng.integers(-128, 128, (M, 512))
    rows = lambda A: [list(r) for r in tile(A).reshape(-1, 16)]
    ub = rows(X) + rows(Y) + rows(S) + rows(G)                     # X@0 Y@512 S@1024 G@1280
    rom = {3: vpu.rom_entry(vpu.LUT_EXP, P["exp0"]), 9: vpu.rom_entry(vpu.LUT_GELU, P["gelu0"])}
    ma, mb = P["adda0"]
    prog = [vpu.vcfg(ma, mb), vpu.lut(vpu.LUT_EXP, 3), vpu.lut(vpu.LUT_GELU, 9),
            vpu.vec(vpu.V_ADD, 8, 0, 4000, M, M, b=512),
            vpu.vec(vpu.V_LN, 8, 0, 5000, M, M),
            vpu.vec(vpu.V_SM, 4, 1024, 6000, M, M),
            vpu.vec(vpu.V_LUT, 32, 1280, 7000, M, M),
            asm.interrupt_host(1), asm.halt()]
    m = fastsim.FastCoreSim(ub, vpu_enabled=True, lut_rom=rom).run(prog)
    assert not m.cfg_err and m.halted
    get = lambda r0, T: untile(m.ubA[r0:r0 + T * M].reshape(T, M, 16))
    assert (get(4000, 8) == I.ew_add(X, Y, ma, mb)).all()
    assert (get(5000, 8) == I.layernorm(X)).all()
    assert (get(6000, 4) == I.softmax(S, P["exp0"])).all()
    assert (get(7000, 32) == I.lut8(G, P["gelu0"])).all()
    # without the VPU the extension opcodes are illegal, like on today's bitstream
    m2 = fastsim.FastCoreSim(ub).run(prog[3:])
    assert m2.cfg_err and m2.cfg_err_op == vpu.X_VEC
    # bad fields, a LUT index missing from the ROM, and unsupported extension ops are cfg_err too
    for w in (vpu.vec(vpu.V_LN, 8, 65500, 0, 64, 64), vpu.lut(vpu.LUT_EXP, 7), vpu.xenc(vpu.X_WROM, count=1),
              vpu.vec(vpu.V_LN, 8, 0, 0, 64, 64) | (1 << 120)):
        assert fastsim.FastCoreSim(ub, vpu_enabled=True, lut_rom=rom).run([w]).cfg_err


def test_vpu_encoding_roundtrip():
    w = vpu.vec(vpu.V_SM, 4, 123, 456, 64, 1024, 3, b=789)
    assert w & 0xFF == 0x43
    op, f = vpu.xdec(w)
    assert (op, f) == (vpu.X_VEC, dict(op=vpu.V_SM, rows=4, a=123, b=789, o=456, count=64, rstride=1024, vstride=3))
    assert vpu.xdec(vpu.vcfg(-5, 131071)) == (vpu.X_VCFG, dict(ma=-5, mb=131071))
    assert vpu.xdec(vpu.lut(1, 200)) == (vpu.X_LUT, dict(sel=1, idx=200))
    with pytest.raises(ValueError):
        vpu.vcfg(1 << 17, 0)
    with pytest.raises(ValueError):
        vpu.vec(vpu.V_LN, 64, 0, 0, 1, 1)                 # rows is 6 bits
    t = np.arange(-128, 128)
    assert (vpu.rom_table(vpu.LUT_GELU, vpu.rom_entry(vpu.LUT_GELU, t)) == t).all()
