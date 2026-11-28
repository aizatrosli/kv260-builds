"""Bit-exact integer operators for the TPU-KV260 transformer extension (vectorised numpy).

Every function here is the *specification* the RTL must match bit for bit.
  requant      : TFLite double-rounding requant, identical to tpuv1 model/quantization/tflite_int8.py
  ew_add       : residual add, out = sat8(rnd((a*Ma + b*Mb) / 2^SH))
  layernorm    : integer LayerNorm (no affine), int8 in -> int8 out at scale 1/LN_Q
  softmax      : row softmax over int8 logits via 256-entry exp LUT -> int8 probs, scale 1/q (q per layer)
  lut8         : int8 -> int8 table lookup (GELU)
"""
import math
import numpy as np

I32 = np.int64
LN_Q = 32          # LayerNorm output scale 1/32 (covers +-4 sigma)
LN_K = 24          # reciprocal precision
SM_K = 22
EXP_ONE = 1 << 15  # exp LUT value for diff = 0
EW_SH = 16


def wrap32(x):
    x = np.asarray(x, dtype=I32) & 0xFFFFFFFF
    return np.where(x & 0x80000000, x - (1 << 32), x)


def srdhm(a, b):
    a = np.asarray(a, I32); b = np.asarray(b, I32)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    num = ab + nudge
    q = np.where(num >= 0, num >> 31, -((-num) >> 31))      # truncate toward zero
    q = wrap32(q)
    both_min = (a == -(1 << 31)) & (b == -(1 << 31))
    return np.where(both_min, (1 << 31) - 1, q)


def rdbpot(x, e):
    x = np.asarray(x, I32); e = np.asarray(e, I32)
    mask = (I32(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def requant(acc, bias, m0, shift, lo=-128, hi=127):
    """acc [..., C] int32; bias/m0/shift per channel [C]."""
    x = wrap32(np.asarray(acc, I32) + bias)
    shift = np.asarray(shift, I32)
    left = np.maximum(shift, 0); right = np.maximum(-shift, 0)
    y = rdbpot(srdhm(wrap32(x << left), m0), right)
    return np.clip(y, lo, hi).astype(I32)


def quantize_multiplier(real):
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    v = q * (1 << 31)
    qf = int(math.copysign(math.floor(abs(v) + 0.5), v))
    if qf == (1 << 31):
        qf //= 2; shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        shift, qf = 30, (1 << 31) - 1
    return qf, shift


def qmul_vec(reals):
    m = [quantize_multiplier(float(r)) for r in np.broadcast_to(reals, np.shape(reals))]
    return np.array([a for a, _ in m], I32), np.array([b for _, b in m], I32)


def ew_mult(s_in, s_out):
    return int(round(s_in / s_out * (1 << EW_SH)))


def ew_add(a, b, ma, mb):
    v = np.asarray(a, I32) * ma + np.asarray(b, I32) * mb
    return np.clip((v + (1 << (EW_SH - 1))) >> EW_SH, -128, 127)


def isqrt(v):
    v = np.asarray(v, I32)
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(I32)
    r = np.where(r * r > v, r - 1, r)
    r = np.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def layernorm(x):
    """x [..., D] int8 -> int8 at scale 1/LN_Q (gamma/beta folded into the next matmul)."""
    x = np.asarray(x, I32); D = x.shape[-1]
    S = x.sum(-1, keepdims=True)
    d = D * x - S
    Q = (d * d).sum(-1, keepdims=True)
    sig = np.maximum(isqrt(Q // D), 1)
    inv = (I32(1) << LN_K) // sig
    v = d * LN_Q * inv
    return np.clip((v + (1 << (LN_K - 1))) >> LN_K, -128, 127)


def exp_lut(s_logit):
    d = np.arange(256)
    return np.round(EXP_ONE * np.exp(-d * s_logit)).astype(I32)


SM_QMAX = 8191     # largest per-layer softmax numerator (keeps the quotient < 2^20)


def softmax(s, lut, q=127):
    """s [..., T] int8 logits -> int8 probs 0..127 at scale 1/q (q: per-layer numerator,
    127 = exact full range; larger q gives small probabilities more resolution, saturating at 127)."""
    s = np.asarray(s, I32)
    m = s.max(-1, keepdims=True)
    e = lut[m - s]
    Z = e.sum(-1, keepdims=True)
    inv = (I32(q) << SM_K) // Z
    return np.minimum((e * inv + (1 << (SM_K - 1))) >> SM_K, 127)


def gelu_lut(s_in, s_out):
    x = np.arange(-128, 128) * s_in
    g = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return np.clip(np.round(g / s_out), -128, 127).astype(I32)   # index = x + 128


def lut8(x, lut):
    return lut[np.asarray(x, I32) + 128]


def matmul(x, w):
    return np.asarray(x, I32) @ np.asarray(w, I32)
