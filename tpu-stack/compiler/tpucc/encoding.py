"""Size estimates for the compact program encodings proposed in docs/compact_encoding.md.

Computed from an actual compiled Program (not from formulas), so the numbers track the compiler.

A. LOAD_QPARAMS qset, row  - one instruction replaces each run of requant SET_CONFIGs; the
   parameters live in a requant-parameter table (one 160-byte row per distinct 16-channel
   config: 16 x {M0 int32, bias int32, shift int8} + zp/min/max, padded to 10 x 16 B).
B. A + LOOP/ENDLOOP with affine field strides - every TPU op becomes a fixed-size loop nest
   (matmul: 7 instructions, conv: 10, +1 per extra M block), DMA / control stays as is.
"""
from .codegen import requant_cfg

ROW_BYTES = 160
LOOP_MATMUL = 7     # READ_WEIGHTS(all) LOOP(nt) LOAD_QPARAMS LOOP(kt) MATMUL SYNC ACTIVATE
LOOP_CONV = 10      # + LOOP(window) and kh/kw/ct as three nested loops around CONVOLVE


def _is_qcfg(w):
    return (w & 0xFF) == 0x30 and ((w >> 16) & 0xFFFF) >> 12 in (1, 2, 3)


def estimate(prog):
    g = prog.graph
    configs = set()
    for o in g.ops:
        if o.kind in ("tpu_matmul", "tpu_conv"):
            n = len(o.attrs["m0"])
            for nt in range(-(-n // 16)):
                configs.add(requant_cfg(o.attrs, nt))
    table = len(configs) * ROW_BYTES
    runs = qcfg = 0
    for k in prog.kernels:
        prev = False
        for w in k.words:
            q = _is_qcfg(w)
            qcfg += q
            runs += q and not prev
            prev = q
    a_instrs = prog.n_instrs - qcfg + runs
    loop = 0
    for name, s in prog.stats.items():
        if name.startswith("<"):
            loop += s.instrs                              # DMA + kernel control, unchanged
        else:
            blocks = int(s.plan.split("x")[-1]) if s.kind.startswith("matmul") else 1
            loop += (LOOP_MATMUL if s.kind.startswith("matmul") else LOOP_CONV) + (blocks > 1)
    return dict(
        current=dict(instrs=prog.n_instrs, program_bytes=prog.n_bytes, table_bytes=0),
        qparam_table=dict(instrs=a_instrs, program_bytes=16 * a_instrs, table_bytes=table,
                          distinct_configs=len(configs), qcfg_runs=runs),
        table_plus_loop=dict(instrs=loop, program_bytes=16 * loop, table_bytes=table),
    )
