"""Vectorised integer math shared by the fast simulator and the reference checks.

requant() is tpuv1's ACTIVATE requantisation (model/quantization/tflite_int8.py: TFLite double
rounding + zp_out + clamp) on numpy arrays; tests/test_intmath.py checks it against the scalar
original on random and corner cases.
"""
import numpy as np

I64 = np.int64


def wrap32(x):
    x = np.asarray(x, dtype=I64) & 0xFFFFFFFF
    return np.where(x & 0x80000000, x - (1 << 32), x)


def srdhm(a, b):
    a = np.asarray(a, I64)
    b = np.asarray(b, I64)
    ab = a * b                                           # |a*b| <= 2^62: exact in int64
    num = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    q = np.where(num >= 0, num >> 31, -((-num) >> 31))   # C division: truncate toward zero
    q = wrap32(q)
    return np.where((a == -(1 << 31)) & (b == -(1 << 31)), (1 << 31) - 1, q)


def rdbpot(x, e):
    x = np.asarray(x, I64)
    e = np.asarray(e, I64)
    mask = (I64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def requant(acc, bias, m0, shift, zp_out=0, act_min=-128, act_max=127):
    """acc [..., C] int32 values; bias/m0/shift broadcast per channel. Returns int64."""
    x = wrap32(np.asarray(acc, I64) + np.asarray(bias, I64))
    shift = np.asarray(shift, I64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    y = rdbpot(srdhm(wrap32(x << left), np.asarray(m0, I64)), right)
    z = wrap32(y + zp_out)
    return np.maximum(act_min, np.minimum(act_max, z))     # tpuv1 order (act_min wins if min > max)
