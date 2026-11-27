"""PROVISIONAL encoding of the future VECTOR unit (VPU) opcode.

The RTL (another engineer: rtl/vpu.sv integration + a VECTOR opcode in decoder.sv) is not final.
Everything this runtime assumes about it lives in this file, so adopting the real encoding is a
one-file change; the semantics (= jepa-tpu/intops.py) are fixed and tested.

  VECTOR (0x21)    one instruction = `count` vectors, each `chunks` UB rows of 16 lanes, chunk c of
                   vector v at row base + c*stride + v (the TPU's tiled layout: stride = M rows).
    [7:0]   0x21
    [11:8]  func     0 ADD   out = sat8((a*MA + b*MB + 2^15) >> 16)        (residual)
                     1 LUT   out = LUT_GELU[a + 128]                        (GELU)
                     2 LN    integer LayerNorm of each vector               (scale 1/32)
                     3 SM    softmax of each vector via LUT_EXP             (0..127)
                     4 LDE   load LUT_EXP  (256 x u16) from 32 UB rows at ub_a
                     5 LDG   load LUT_GELU (256 x s16) from 32 UB rows at ub_a
    [31:16] ub_a     [47:32] ub_out     [63:48] ub_b
    [75:64] count    [81:76] chunks     [97:82] stride
  SET_CONFIG target 0x5000 (VPU_MULT): value = MA[17:0] | MB[17:0] << 18 (signed 18-bit each).

The decoder is assumed to wait for the VPU to finish before issuing the next instruction (the
runtime also emits SYNC(U_ALL) before each VECTOR), so no new SYNC mask bit is needed.
LUT tables are 256 16-bit little-endian entries = 512 bytes = 32 UB rows; entry i of LUT_GELU is the
value for input i - 128, entry d of LUT_EXP is exp for (max - s) = d.
"""
import numpy as np

OP_VECTOR = 0x21
F_ADD, F_LUT, F_LN, F_SM, F_LDE, F_LDG = range(6)
FUNC_NAMES = {F_ADD: "add", F_LUT: "gelu", F_LN: "layernorm", F_SM: "softmax", F_LDE: "load_exp", F_LDG: "load_gelu"}
VPU_MULT = 0x5000
LUT_ROWS = 32


def vector(func, ub_a, ub_out=0, count=1, chunks=1, stride=0, ub_b=0):
    for v, w, n in ((ub_a, 16, "ub_a"), (ub_out, 16, "ub_out"), (ub_b, 16, "ub_b"), (count, 12, "count"),
                    (chunks, 6, "chunks"), (stride, 16, "stride"), (func, 4, "func")):
        if not 0 <= v < (1 << w):
            raise ValueError(f"VECTOR.{n}={v} does not fit {w} bits")
    return (OP_VECTOR | (func << 8) | (ub_a << 16) | (ub_out << 32) | (ub_b << 48) | (count << 64)
            | (chunks << 76) | (stride << 82))


def decode(word):
    return dict(func=(word >> 8) & 0xF, ub_a=(word >> 16) & 0xFFFF, ub_out=(word >> 32) & 0xFFFF,
                ub_b=(word >> 48) & 0xFFFF, count=(word >> 64) & 0xFFF, chunks=(word >> 76) & 0x3F,
                stride=(word >> 82) & 0xFFFF, reserved=word >> 98)


def mult_value(ma, mb):
    for m in (ma, mb):
        if not -(1 << 17) <= m < (1 << 17):
            raise ValueError(f"residual multiplier {m} does not fit 18 signed bits")
    return (ma & 0x3FFFF) | ((mb & 0x3FFFF) << 18)


def mult_decode(value):
    s18 = lambda x: x - (1 << 18) if x & (1 << 17) else x
    return s18(value & 0x3FFFF), s18((value >> 18) & 0x3FFFF)


def lut_rows(table):
    """256-entry table -> 32 UB rows (int8 [32, 16]) of little-endian 16-bit entries."""
    t = np.asarray(table, np.int64)
    if t.shape != (256,) or t.min() < -32768 or t.max() > 65535:
        raise ValueError("LUT must be 256 entries in 16 bits")
    return (t & 0xFFFF).astype("<u2").view(np.int8).reshape(LUT_ROWS, 16)


def lut_from_rows(rows, signed):
    u = np.ascontiguousarray(rows, dtype=np.int8).view("<u2").reshape(256).astype(np.int64)
    return np.where(u >= 32768, u - 65536, u) if signed else u
