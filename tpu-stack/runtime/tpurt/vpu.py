"""Vector-unit (VPU) instructions: the sequencer extension of jepa_top (opcodes 0x40-0x47).

Source of truth: jepa-tpu/compile_jepa.py on branch `jepa-e2e-optimize` (XF table, xenc/xdec, and
the bit-exact `Machine`), RTL in jepa-tpu/rtl/jepa_top.sv. A sequencer in front of the unmodified
tpuv1 core executes these words; tpuv1 opcodes pass straight through. This runtime only emits the
three vector ops below; the field table XF mirrors all eight so programs can be disassembled.

  X_VEC   0x43  op[9:8] rows[15:10] a[31:16] b[47:32] o[63:48] count[79:64] rstride[95:80] vstride[103:96]
                `count` jobs; job v reads the vector concat_r UB[base + v*vstride + r*rstride], r < rows
                (16 lanes per row). op: 0 ADD  out = sat8((a*MA + b*MB + 2^15) >> 16)   (residual)
                                        1 LUT  out = LUT_GELU[a + 128]                   (GELU)
                                        2 LN   integer LayerNorm (intops.layernorm)
                                        3 SM   softmax via LUT_EXP (0..127)
  X_LUT   0x44  sel[8] idx[23:16]   load VPU LUT `sel` (0 exp, 1 gelu) from LUT-ROM table `idx`
  X_VCFG  0x45  ma[25:8] mb[49:32] sq[63:50]   residual multipliers (signed 18 bit) for later ADD
                jobs, and the softmax numerator q (127..8191): SM jobs compute
                inv = (q << 22) // Z, p = min((e*inv + 2^21) >> 22, 127), i.e. probs at scale 1/q

The LUT ROM is part of the bitstream (lrom.hex, 256 x 16-bit entries per table). compile_jepa's
layout is exp table of layer l at idx l and GELU table of layer l at idx depth + l; GELU entries are
stored as 8 bits and sign-extended. On a simulated backend the runtime supplies the ROM contents
itself (`backend.lut_rom`, see `jepa_lut_rom`).

The sequencer waits for the core to go idle before each X_VEC/X_LUT (the reference Machine does
`_sync(U_ALL)`), so no extra SYNC is needed; the runtime emits SYNC(U_ALL) anyway for safety.
"""
import numpy as np

X_WROM, X_WUB, X_LOADQ, X_VEC, X_LUT, X_VCFG, X_UBROM, X_MM = range(0x40, 0x48)
XF = {
    X_WROM:  {"count": (8, 24)},
    X_WUB:   {"tr": (8, 1), "base": (16, 16), "stride": (32, 16), "kt": (48, 8), "nt": (56, 8)},
    X_LOADQ: {"q": (8, 2), "entry": (16, 16)},
    X_VEC:   {"op": (8, 2), "rows": (10, 6), "a": (16, 16), "b": (32, 16), "o": (48, 16),
              "count": (64, 16), "rstride": (80, 16), "vstride": (96, 8)},
    X_LUT:   {"sel": (8, 1), "idx": (16, 8)},
    X_VCFG:  {"ma": (8, 18), "mb": (32, 18), "sq": (50, 14)},
    X_UBROM: {"base": (16, 16), "count": (32, 16)},
    X_MM:    {"q": (8, 2), "sync": (10, 1), "kt": (11, 6), "in_base": (17, 16), "acc_row": (33, 11),
              "out_row": (44, 16), "entry": (60, 16), "B": (76, 16)},
}
XNAMES = {X_WROM: "X_WROM", X_WUB: "X_WUB", X_LOADQ: "X_LOADQ", X_VEC: "X_VEC", X_LUT: "X_LUT",
          X_VCFG: "X_VCFG", X_UBROM: "X_UBROM", X_MM: "X_MM"}
SIGNED = ("ma", "mb")

V_ADD, V_LUT, V_LN, V_SM = range(4)
OP_NAMES = {V_ADD: "add", V_LUT: "gelu", V_LN: "layernorm", V_SM: "softmax"}
LUT_EXP, LUT_GELU = 0, 1                # X_LUT sel
LUT_OF_OP = {V_SM: LUT_EXP, V_LUT: LUT_GELU}


def xenc(opc, **kw):
    """Encode an extension word (same contract as compile_jepa.xenc, but raises instead of assert)."""
    w = opc
    for k, v in kw.items():
        lsb, width = XF[opc][k]
        v = int(v)
        if k in SIGNED:
            if not -(1 << (width - 1)) <= v < (1 << (width - 1)):
                raise ValueError(f"{XNAMES[opc]}.{k}={v} does not fit {width} signed bits")
            v &= (1 << width) - 1
        elif not 0 <= v < (1 << width):
            raise ValueError(f"{XNAMES[opc]}.{k}={v} does not fit {width} bits")
        w |= v << lsb
    return w


def xdec(w):
    op = w & 0xFF
    f = {k: (w >> l) & ((1 << n) - 1) for k, (l, n) in XF[op].items()}
    for k in SIGNED:
        if k in f and f[k] & (1 << (XF[op][k][1] - 1)):
            f[k] -= 1 << XF[op][k][1]
    return op, f


def is_ext(w):
    return (w & 0xFF) in XF


def used_bits(op):
    m = 0xFF
    for lsb, width in XF[op].values():
        m |= ((1 << width) - 1) << lsb
    return m


def disasm(w):
    op, f = xdec(w)
    return XNAMES[op] + " " + " ".join(f"{k}={v}" for k, v in f.items())


def vec(op, rows, a, o, count, rstride, vstride=1, b=0):
    return xenc(X_VEC, op=op, rows=rows, a=a, b=b, o=o, count=count, rstride=rstride, vstride=vstride)


def lut(sel, idx):
    return xenc(X_LUT, sel=sel, idx=idx)


SQ_DEFAULT, SQ_MAX = 127, 8191


def vcfg(ma, mb, sq=SQ_DEFAULT):
    if not SQ_DEFAULT <= int(sq) <= SQ_MAX:
        raise ValueError(f"X_VCFG.sq={sq} outside {SQ_DEFAULT}..{SQ_MAX}")
    return xenc(X_VCFG, ma=ma, mb=mb, sq=sq)


# ---- LUT ROM -------------------------------------------------------------------------------------
def lut_idx(sel, layer, depth):
    """compile_jepa's LUT-ROM layout: exp tables at idx l, GELU tables at idx depth + l."""
    return layer if sel == LUT_EXP else depth + layer


def rom_entry(sel, table):
    """256-entry table -> LUT-ROM words as stored in lrom.hex (exp: u16, gelu: 8-bit two's complement)."""
    t = np.asarray(table, np.int64)
    if t.shape != (256,):
        raise ValueError("LUT must have 256 entries")
    if sel == LUT_EXP and (t.min() < 0 or t.max() > 0xFFFF):
        raise ValueError("exp LUT entries must fit u16")
    if sel == LUT_GELU and (t.min() < -128 or t.max() > 127):
        raise ValueError("GELU LUT entries must fit int8")
    return t & (0xFFFF if sel == LUT_EXP else 0xFF)


def rom_table(sel, words):
    """LUT-ROM words -> the table the VPU uses (mirrors Machine's X_LUT)."""
    t = np.asarray(words, np.int64)
    return t if sel == LUT_EXP else np.where(t > 127, t - 256, t)


def jepa_lut_rom(P, depth):
    """{idx: rom words} for a jepa qmodel, in compile_jepa's layout."""
    rom = {}
    for l in range(depth):
        rom[lut_idx(LUT_EXP, l, depth)] = rom_entry(LUT_EXP, P[f"exp{l}"])
        rom[lut_idx(LUT_GELU, l, depth)] = rom_entry(LUT_GELU, P[f"gelu{l}"])
    return rom


def lrom_hex(rom):
    """LUT ROM as $readmemh text (one 16-bit word per line), for comparing with a bitstream's lrom.hex."""
    n = (max(rom) + 1) * 256 if rom else 0
    words = np.zeros(n, np.int64)
    for i, t in rom.items():
        words[i * 256:(i + 1) * 256] = t
    return "".join(f"{int(v):04x}\n" for v in words)
