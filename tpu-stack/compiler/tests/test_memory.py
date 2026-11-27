import numpy as np

from tpucc import memory, codegen, passes
from tpucc.ir import Graph


def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def test_live_buffers_never_share_rows(jepa_prog):
    m = jepa_prog.mem
    roots = list(m.ub_base)
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            if _overlap(m.intervals[a], m.intervals[b]):
                ra = (m.ub_base[a], m.ub_base[a] + m.rows[a] - 1)
                rb = (m.ub_base[b], m.ub_base[b] + m.rows[b] - 1)
                assert not _overlap(ra, rb), (a, b)


def test_reuse_shrinks_footprint(jepa_lowered, jepa_prog):
    m = jepa_prog.mem
    assert m.live_peak_rows <= m.peak_rows <= m.noreuse_rows
    assert m.peak_rows * 16 <= 64 * 1024                 # README: "about 64 KiB of activations"
    assert m.noreuse_rows > 10 * m.peak_rows
    nr = codegen.compile_graph(jepa_lowered, reuse=False).mem
    assert nr.peak_rows == nr.noreuse_rows


def test_views_alias_their_root(jepa_lowered, jepa_prog):
    m = jepa_prog.mem
    assert m.alias["q0h1"] == ("qkv0", 2 * 64)          # columns 32..63 = k-tiles 2,3
    assert m.alias["k0h0"] == ("qkv0", 8 * 64)
    assert m.alias["v0h3"] == ("qkv0", 22 * 64)
    assert m.alias["pv0h2"] == ("o0", 4 * 64)           # heads written in place into the concat
    assert "q0h1" not in m.ub_base and "pv0h2" not in m.ub_base


def test_host_boundary_dma_plan(jepa_lowered, jepa_prog):
    m = jepa_prog.mem
    loaded = {r for rs in m.rhm.values() for r in rs}
    written = {t for ts in m.whm.values() for t in ts}
    assert "img" in loaded and "ln1_0" in loaded and "gelu0" in loaded and "sm0h0" in loaded
    assert {"k0h0", "v0h0", "qk0h0", "embed", "proj0", "fc10", "fc20"} <= written
    assert "q0h0" not in written and "o0" not in written and "qkv0" not in written   # stay in UB
    assert "adda0" not in m.ub_base                     # residual stream is host-only (DDR)


def test_first_fit_reuses_freed_rows():
    g = Graph()
    x = g.input("x", (16, 16))
    a = g.op("layernorm", "h0", [x], (16, 16))
    names = [a]
    for i in range(4):                                 # chain of TPU ops: only 2 live at a time
        w = g.const(f"w{i}", np.eye(16, dtype=np.int64))
        acc = g.op("matmul", f"m{i}", [names[-1], w], (16, 16), "int32", out_name=f"m{i}.acc")
        names.append(g.op("requant", f"r{i}", [acc], (16, 16), out_name=f"y{i}", bias=np.zeros(16),
                          m0=np.full(16, 1 << 30), shift=np.zeros(16), zp_out=0, act_min=-128, act_max=127))
    g.mark_output(names[-1])
    prog = codegen.compile_graph(passes.lower(g))
    assert prog.mem.peak_rows == 32 and prog.mem.noreuse_rows == 5 * 16
