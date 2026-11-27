import errno
import sys

import numpy as np
import pytest

from tpurt import _paths, isa, get_backend
from tpurt.backends.base import relocate, Job
from tpurt.program import Gemm, Tensor, pack_weight_tiles, schedule, ProgramTooLong

_paths.need_tpuv1_model()
sys.path.insert(0, str(_paths.TPUV1 / "model/mlp"))


def test_weight_tiles_match_tpuv1_compiler():
    import compile_mlp as cm
    import sim as isasim
    net = cm.Net([48, 80], ["none"], 3)
    W = np.zeros((48, 80), np.int64)
    W[:, :] = net.layers[0]["wq"]
    words = cm.weight_words(net)
    ref = b"".join(w.to_bytes(16, "little") for w in words)
    assert pack_weight_tiles(W) == ref


def _gemm(name, x, out, KT, NT, M, w):
    n = 16 * NT
    return Gemm(name, x, 0, M, KT, M, out, 0, M, NT, w, 0, np.full(n, 1 << 30), np.full(n, -8), np.zeros(n, int))


def test_schedule_splits_and_spills():
    M = 64
    x = Tensor("x", 32 * M)
    w = Tensor("w", 32 * 32 * 16, "const")
    h = Tensor("h", 32 * M)
    y = Tensor("y", 32 * M, "out")
    steps = [_gemm("a", x, h, 32, 32, M, w), _gemm("b", h, y, 32, 32, M, w)]
    with pytest.raises(ProgramTooLong):
        schedule(steps)                                 # one 32x32-tile GEMM needs > 2048 instructions
    from tpurt.program import split_gemm_nt
    st = [g for s in steps for g in split_gemm_nt(s, 8)]
    items = schedule(st)
    progs = [p for k, p in items if k == "dev"]
    assert len(progs) > 1 and all(p.n_instr <= isa.MAX_INSTR for p in progs)
    # h is produced in one program and consumed in later ones -> stored then re-loaded
    assert any(h in p.stores for p in progs) and any(h in p.loads for p in progs[1:])
    assert all(t is not y for p in progs for t in p.loads)


def test_qset_cache_avoids_set_config():
    M = 64
    x, w, o = Tensor("x", 2 * M), Tensor("w", 8 * 16, "const"), Tensor("o", 8 * M, "out")
    st = [Gemm(f"g{i}", x, 0, M, 2, M, o, 0, M, 4, w, 0, np.full(64, 1 << 30), np.full(64, -7), np.zeros(64, int))
          for i in range(3)]
    prog = schedule(st)[0][1]
    n_cfg = sum(1 for wd in prog.words if wd & 0xFF == isa.OP_SET_CONFIG)
    assert n_cfg == 33                                  # one qset load, then cache hits


def test_relocate_rules():
    be = get_backend("sim")
    a, b = be.alloc(4096), be.alloc(4096)
    size = lambda buf: buf.alloc_size
    prog = isa.to_bytes([isa.read_host_mem(0, 0, 4), isa.read_weights(0, 1), isa.matmul(0, 0, 4, 0, 1),
                         isa.write_host_mem(0, 0, 4), isa.halt()])
    words, used = relocate(prog, [(0, a, 0), (1, a, 256), (3, b, 64)], lambda x: x.phys, size)
    assert isa.dma_fields(words[0])[1] == a.phys and isa.dma_fields(words[3])[1] == b.phys + 64
    bad = [([(0, a, 0), (1, a, 256)], "missing"), ([(0, a, 0), (0, a, 0), (1, a, 0), (3, b, 0)], "dup"),
           ([(2, a, 0), (0, a, 0), (1, a, 0), (3, b, 0)], "non-DMA"), ([(0, a, 8), (1, a, 0), (3, b, 0)], "align"),
           ([(0, a, 4096 - 32), (1, a, 0), (3, b, 0)], "span")]
    for rel, why in bad:
        with pytest.raises(OSError) as e:
            relocate(prog, rel, lambda x: x.phys, size)
        assert e.value.errno == errno.EINVAL, why
    under = isa.to_bytes([isa.matmul(0, 0, 4, 0, 1), isa.read_weights(0, 1), isa.halt()])
    with pytest.raises(OSError):
        relocate(under, [(1, a, 0)], lambda x: x.phys, size)
    # the mock driver applies the same rules (EINVAL from the ioctl)
    mk = get_backend("mock")
    ma = mk.alloc(4096)
    with pytest.raises(OSError) as e:
        mk.run(prog, [(0, ma, 8), (1, ma, 0), (3, ma, 0)])
    assert e.value.errno == errno.EINVAL
