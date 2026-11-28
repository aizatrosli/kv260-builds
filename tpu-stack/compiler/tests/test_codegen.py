import numpy as np

from tpucc import codegen, encoding
from tpucc.env import asm, IFIFO_WORDS, WFIFO_TILES
from tpucc.baseline import baseline_size
import compile_mlp as CM
import compile_cnn as CC


def _words(b):
    return [int.from_bytes(b[i:i + 16], "little") for i in range(0, len(b), 16)]


def test_matmul_tile_image_matches_tpuv1():
    W = np.random.default_rng(0).integers(-127, 128, (40, 50))

    class Net:
        layers = [dict(wq=W)]
    assert _words(codegen.tile_bytes(codegen.tiles_matmul(W))) == CM.weight_words(Net)


def test_conv_tile_image_matches_tpuv1():
    spec, seed = CC.CASES["conv3x3_ct2_co32"]
    net = CC.Net(spec, seed)
    W = net.layers[0]["wq"]
    assert _words(codegen.tile_bytes(codegen.tiles_conv(W))) == CC.weight_words(net)


def test_host_layout_roundtrip():
    x = np.random.default_rng(1).integers(-128, 128, (5, 37))
    b = codegen.host_layout(x)
    assert len(b) == 3 * 5 * 16
    assert np.array_equal(codegen.from_host_layout(b, x.shape), x)


def test_kernels_well_formed(jepa_prog):
    for k in jepa_prog.kernels:
        assert len(k.words) <= IFIFO_WORDS
        assert [w & 0xFF for w in k.words[-3:]] == [asm.SYNC, asm.INTERRUPT_HOST, asm.HALT]
        for w in k.words:
            op, f, rsv, known = asm.decode(w)
            assert known and not rsv


def test_set_config_only_on_change(jepa_prog):
    regs = {}
    for k in jepa_prog.kernels:
        for w in k.words:
            op, f, _, _ = asm.decode(w)
            if op == asm.SET_CONFIG:
                assert regs.get(f["target"]) != f["value"], hex(f["target"])
                regs[f["target"]] = f["value"]


def test_weight_fifo_never_overfilled(jepa_prog):
    outstanding = requested = consumed = 0
    for k in jepa_prog.kernels:
        for w in k.words:
            op, f, _, _ = asm.decode(w)
            if op == asm.READ_WEIGHTS:
                assert outstanding == 0                # requested lazily, when the FIFO is empty
                outstanding += f["tiles"]
                requested += f["tiles"]
                assert outstanding <= WFIFO_TILES
            elif op in (asm.MATMUL, asm.CONVOLVE) and f["new_weights"]:
                outstanding -= 1
                consumed += 1
                assert outstanding >= 0
    assert requested == consumed == sum(s.weight_tiles for s in jepa_prog.stats.values())


def test_program_size_vs_baseline(jepa_lowered, jepa_prog):
    base = baseline_size(jepa_lowered)
    assert base["programs"] == 73
    assert 400 * 1024 < base["bytes"] < 410 * 1024              # the README's 404 KiB
    nd = codegen.compile_graph(jepa_lowered, dedupe=False)
    assert abs(nd.n_instrs - base["instrs"]) < 0.01 * base["instrs"]
    assert jepa_prog.n_bytes < 0.85 * base["bytes"]
    assert jepa_prog.opcode_counts()["SET_CONFIG"] < 0.8 * base["set_config"]
    rc = codegen.compile_graph(jepa_lowered, reset_config_per_kernel=True)
    assert jepa_prog.n_instrs <= rc.n_instrs < nd.n_instrs


def test_baseline_matches_rtl_check(P, jepa_golden, jepa_lowered):
    """Rebuild the baseline exactly as jepa-tpu rtl_check.py does (its OneLayer + tpuv1 compiler)."""
    import re
    import sys
    from tpucc import env
    sys.path.insert(0, str(env.JEPA_ROOT))
    import rtl_check
    import intops as I
    _, log = jepa_golden
    total = 0
    for name, x, w, acc in log:
        if re.match(r"^(qk|pv)(\d+)h\d+$", name):
            m0, sh = P[name[:2] + re.match(r"^..(\d+)", name).group(1)]
            net = rtl_check.OneLayer(np.asarray(w, np.int64), 0, m0, sh)
        else:
            p = P[name]
            net = rtl_check.OneLayer(p["w"], p["bias"], p["m0"], p["shift"])
        total += len(CM.emit_program(net, x.shape[0])[0])
    assert total == baseline_size(jepa_lowered)["instrs"]


def test_encoding_estimate(jepa_prog):
    e = encoding.estimate(jepa_prog)
    assert e["qparam_table"]["distinct_configs"] == 440 + 12
    assert e["qparam_table"]["instrs"] < e["current"]["instrs"] / 2
    assert e["table_plus_loop"]["instrs"] < IFIFO_WORDS          # whole encoder in one IFIFO load
    assert e["table_plus_loop"]["program_bytes"] < 16 * 1024
