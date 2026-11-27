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
    """Provisional VECTOR instruction == intops on gathered vectors (all four functions + LUT loads)."""
    import intops as I
    rng = np.random.default_rng(0)
    M = 64
    X = rng.integers(-128, 128, (M, 128))
    Y = rng.integers(-128, 128, (M, 128))
    S = rng.integers(-128, 128, (M, 64))
    G = rng.integers(-128, 128, (M, 512))
    rows = lambda A: [list(r) for r in tile(A).reshape(-1, 16)]
    ub = rows(X) + rows(Y) + rows(S) + rows(G)                     # X@0 Y@512 S@1024 G@1280
    lut_e, lut_g = vpu.lut_rows(P["exp0"]), vpu.lut_rows(P["gelu0"])
    base_l = len(ub)
    ub += [list(r) for r in lut_e] + [list(r) for r in lut_g]
    ma, mb = P["adda0"]
    prog = [asm.set_config(vpu.VPU_MULT, vpu.mult_value(ma, mb)),
            vpu.vector(vpu.F_LDE, base_l), vpu.vector(vpu.F_LDG, base_l + 32),
            vpu.vector(vpu.F_ADD, 0, 4000, M, 8, M, ub_b=512),
            vpu.vector(vpu.F_LN, 0, 5000, M, 8, M),
            vpu.vector(vpu.F_SM, 1024, 6000, M, 4, M),
            vpu.vector(vpu.F_LUT, 1280, 7000, M, 32, M),
            asm.interrupt_host(1), asm.halt()]
    m = fastsim.FastCoreSim(ub, vpu_enabled=True).run(prog)
    assert not m.cfg_err and m.halted
    get = lambda r0, T: untile(m.ubA[r0:r0 + T * M].reshape(T, M, 16))
    assert (get(4000, 8) == I.ew_add(X, Y, ma, mb)).all()
    assert (get(5000, 8) == I.layernorm(X)).all()
    assert (get(6000, 4) == I.softmax(S, P["exp0"])).all()
    assert (get(7000, 32) == I.lut8(G, P["gelu0"])).all()
    # without the VPU the opcode is illegal, like on today's bitstream
    m2 = fastsim.FastCoreSim(ub).run(prog[1:])
    assert m2.cfg_err and m2.cfg_err_op == vpu.OP_VECTOR
    # bad VECTOR fields are cfg_err too
    m3 = fastsim.FastCoreSim(ub, vpu_enabled=True).run([vpu.vector(vpu.F_LN, 65500, 0, 64, 8, 64)])
    assert m3.cfg_err


def test_vpu_encoding_roundtrip():
    w = vpu.vector(vpu.F_SM, 123, 456, 64, 4, 1024, 789)
    f = vpu.decode(w)
    assert (f["func"], f["ub_a"], f["ub_out"], f["count"], f["chunks"], f["stride"], f["ub_b"], f["reserved"]) == \
        (vpu.F_SM, 123, 456, 64, 4, 1024, 789, 0)
    assert vpu.mult_decode(vpu.mult_value(-5, 131071)) == (-5, 131071)
    with pytest.raises(ValueError):
        vpu.mult_value(1 << 17, 0)
    t = np.arange(-128, 128)
    assert (vpu.lut_from_rows(vpu.lut_rows(t), signed=True) == t).all()
