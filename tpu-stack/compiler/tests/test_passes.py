import numpy as np
import pytest

from tpucc import passes, memory, frontends, refexec
from tpucc.ir import Graph


def test_fuse_jepa(jepa_graph, jepa_img, jepa_ref):
    f = passes.fuse(jepa_graph)
    assert not [o for o in f.ops if o.kind in ("matmul", "requant", "clamp")]
    assert sum(o.kind == "tpu_matmul" for o in f.ops) == 73
    assert len(f.ops) == len(jepa_graph.ops) - 73
    v = refexec.run(f, {"img": jepa_img})
    assert np.array_equal(v["ln_out"], jepa_ref["ln_out"])
    assert len(jepa_graph.ops) == 280            # input graph untouched


def test_fuse_clamp_and_pool():
    import compile_cnn as CC
    spec, seed = CC.CASES["cnn_tiny"]
    g = frontends.from_cnn(CC.Net(spec, seed))
    f = passes.fuse(g)
    convs = [o for o in f.ops if o.kind == "tpu_conv"]
    assert len(convs) == 2 and all(o.attrs["pool"] == "max" and o.attrs["act"] == "relu" for o in convs)
    lo = {o.name.split(".")[0]: o.attrs["lo"] for o in g.ops if o.kind == "clamp"}
    assert all(o.attrs["act_min"] == lo[o.name] for o in convs)


def test_legalize_inserts_weight_packing(jepa_graph):
    lg = passes.legalize(passes.fuse(jepa_graph))
    packs = [o for o in lg.ops if o.kind == "pack_weights"]
    assert len(packs) == 48
    assert sum(o.attrs["transpose"] for o in packs) == 24          # K^T for every Q.K^T
    dyn = [o for o in lg.ops if o.attrs.get("dynamic_weights")]
    assert len(dyn) == 48 and all(not o.attrs["transpose_b"] for o in dyn)


def test_legalize_rejects_unfused():
    g = Graph()
    x = g.input("x", (4, 16))
    w = g.const("w", np.ones((16, 16)))
    g.op("matmul", "m", [x, w], (4, 16), "int32")
    with pytest.raises(ValueError):
        passes.legalize(g)


def test_schedule_groups_devices(jepa_lowered):
    g = jepa_lowered
    done = set(g.inputs) | {n for n, t in g.tensors.items() if t.is_const}
    for o in g.ops:
        assert all(i in done for i in o.inputs)
        done |= set(o.outputs)
    steps, _, _ = memory.segment(g)
    kernels = [s for s in steps if s.kind == "kernel"]
    # per block: qkv | 4x qk | 4x pv + proj | fc1 | fc2   (+ embed)
    assert len(kernels) == 1 + 6 * 5
    assert any(s.ops == ["pv0h0", "pv0h1", "pv0h2", "pv0h3", "proj0"] for s in kernels)
    host = [s for s in steps if s.kind == "host"]
    assert any(set(s.ops) == {f"pack_qk0h{h}" for h in range(4)} | {f"pack_pv0h{h}" for h in range(4)} for s in host)


def test_partition_plans(jepa_lowered):
    cur = passes.device_plan(jepa_lowered, "current")
    fut = passes.device_plan(jepa_lowered, "future")
    assert cur["ln1_0"] == "host" and fut["ln1_0"] == "vpu"
    assert cur["gelu0"] == "host" and fut["gelu0"] == "act_lut"
    assert cur["pack_qk0h0"] == "host" and fut["pack_qk0h0"] == "ub2w"
    assert cur["fc10"] == fut["fc10"] == "tpu"
