"""Bit-exact integer operators (vectorised numpy, int64 containers).

This is the arithmetic contract of the TPU-KV260 stack. It reproduces, bit for bit:
  * tpuv1 ``model/quantization/tflite_int8.py``  (requant: TFLite double-rounding
    MultiplyByQuantizedMultiplier, QuantizeMultiplier, CalculateActivationRangeQuantized)
  * jepa-tpu ``intops.py``                       (ew_add, layernorm, softmax, LUTs)
and generalises the jepa vector ops to non-zero zero points (with zp == 0 they are identical).
The tests in ``tests/test_intops.py`` cross-check both references on random vectors.
"""
import math
import numpy as np

I64 = np.int64
LN_Q = 32          # LayerNorm output scale 1/LN_Q
LN_K = 24          # LayerNorm reciprocal precision (bits)
SM_K = 22          # softmax reciprocal precision (bits)
SM_OUT = 127       # softmax output: probs in 0..127 at scale 1/127
EXP_ONE = 1 << 15  # exp LUT value for logit difference 0
EW_SH = 16         # element-wise multiplier precision (Q16)
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1

CONSTANTS = dict(LN_Q=LN_Q, LN_K=LN_K, SM_K=SM_K, SM_OUT=SM_OUT, EXP_ONE=EXP_ONE, EW_SH=EW_SH)


# --------------------------------------------------------------------------------- requant ---
def wrap32(x):
    return ((np.asarray(x, dtype=I64) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(int32, int32) -> int32."""
    a = np.asarray(a, I64); b = np.asarray(b, I64)
    ab = a * b
    neg = ab < 0
    num = ab + (1 << 30) - neg * ((1 << 31) - 1)                    # nudge: 2^30 or 1 - 2^30
    q = wrap32((num + (num < 0) * ((1 << 31) - 1)) >> 31)           # C division: toward zero
    return np.where((a == INT32_MIN) & (b == INT32_MIN), INT32_MAX, q)


def rdbpot(x, e):
    """RoundingDivideByPOT(int32 x, 0 <= e <= 31): round half away from zero."""
    x = np.asarray(x, I64); e = np.asarray(e, I64)
    mask = (I64(1) << e) - 1
    return (x >> e) + ((x & mask) > ((mask >> 1) + (x < 0)))


def mbqm(x, m0, shift):
    """MultiplyByQuantizedMultiplier(int32 x, int32 m0, int shift); shift > 0 is a left shift."""
    shift = np.asarray(shift, I64)
    left = np.maximum(shift, 0); right = np.maximum(-shift, 0)
    return rdbpot(srdhm(wrap32(np.asarray(x, I64) << left), m0), right)


def requant(acc, bias, m0, shift, zp_out=0, act_min=-128, act_max=127):
    """tpuv1 activation unit: clamp(wrap32(mbqm(wrap32(acc + bias)) + zp_out)).
    acc [..., C] int32; bias/m0/shift broadcast over the channel (last) axis."""
    x = wrap32(np.asarray(acc, I64) + np.asarray(bias, I64))
    y = wrap32(mbqm(x, m0, shift) + np.asarray(zp_out, I64))
    return np.clip(y, act_min, act_max).astype(I64)


def quantize_multiplier(real):
    """TFLite QuantizeMultiplier(double) -> (m0, shift), m0 in [2^30, 2^31) (or 0)."""
    real = float(real)
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
    reals = np.atleast_1d(np.asarray(reals, dtype=np.float64))
    m = [quantize_multiplier(r) for r in reals.ravel()]
    return (np.array([a for a, _ in m], I64).reshape(reals.shape),
            np.array([b for _, b in m], I64).reshape(reals.shape))


def act_range(kind, scale, zp):
    """TFLite CalculateActivationRangeQuantized for int8 output."""
    if kind in (None, "none"):
        return -128, 127
    if kind == "relu":
        return max(-128, int(zp)), 127
    if kind == "relu6":
        return max(-128, int(zp)), min(127, int(zp) + int(round(6.0 / scale)))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- vector ops ---
def ew_mult(s_in, s_out):
    """Q16 element-wise multiplier (scalar or array)."""
    r = np.round(np.asarray(s_in, np.float64) / np.asarray(s_out, np.float64) * (1 << EW_SH))
    return int(r) if np.ndim(r) == 0 else r.astype(I64)


def ew_add(a, b, ma, mb, za=0, zb=0, zo=0):
    """sat8(((a-za)*ma + (b-zb)*mb + 2^15) >> 16 + zo).  With zero points 0 == jepa intops.ew_add."""
    v = (np.asarray(a, I64) - za) * ma + (np.asarray(b, I64) - zb) * mb
    return np.clip(((v + (1 << (EW_SH - 1))) >> EW_SH) + zo, -128, 127).astype(I64)


def rescale(a, m, za=0, zo=0):
    """Single-input element-wise requant (ew_add with mb = 0)."""
    v = (np.asarray(a, I64) - za) * np.asarray(m, I64)
    return np.clip(((v + (1 << (EW_SH - 1))) >> EW_SH) + zo, -128, 127).astype(I64)


def isqrt(v):
    v = np.asarray(v, I64)
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(I64)
    r = np.where(r * r > v, r - 1, r)
    return np.where((r + 1) * (r + 1) <= v, r + 1, r)


def layernorm(x):
    """x [..., D] int8 (any zero point: it cancels) -> int8 at scale 1/LN_Q, zp 0. No affine."""
    x = np.asarray(x, I64); D = x.shape[-1]
    S = x.sum(-1, keepdims=True)
    d = D * x - S
    Q = (d * d).sum(-1, keepdims=True)
    sig = np.maximum(isqrt(Q // D), 1)
    inv = (I64(1) << LN_K) // sig
    v = d * LN_Q * inv
    return np.clip((v + (1 << (LN_K - 1))) >> LN_K, -128, 127)


def exp_lut(s_logit):
    d = np.arange(256)
    return np.round(EXP_ONE * np.exp(-d * float(s_logit))).astype(I64)


def softmax(s, lut, out_mult=SM_OUT):
    """Row softmax over the last axis. int8 logits (any zp) -> int8 probs, scale 1/out_mult,
    clamped to 0..127. out_mult = 127 (default) is exactly jepa-tpu's softmax; a larger out_mult
    (calibrated) spends the 7 bits on the small probabilities and saturates p > 127/out_mult."""
    s = np.asarray(s, I64); lut = np.asarray(lut, I64)
    e = lut[s.max(-1, keepdims=True) - s]
    Z = e.sum(-1, keepdims=True)
    inv = (I64(out_mult) << SM_K) // Z
    return np.minimum((e * inv + (1 << (SM_K - 1))) >> SM_K, 127)


def _gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


LUT_FUNCS = {
    "gelu": _gelu,
    "gelu_tanh": lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))),
    "relu": lambda x: np.maximum(x, 0.0),
    "relu6": lambda x: np.clip(x, 0.0, 6.0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "silu": lambda x: x / (1 + np.exp(-x)),
    "identity": lambda x: x,
}


def _erf_fast(x):
    import torch
    return torch.erf(torch.from_numpy(np.asarray(x, np.float64))).numpy()


# fast float versions for the float / fake-quant interpreter (LUT building keeps math.erf so the
# tables stay bit-identical to jepa-tpu's gelu_lut)
FLOAT_FUNCS = dict(LUT_FUNCS, gelu=lambda x: 0.5 * x * (1 + _erf_fast(x / math.sqrt(2))))


def make_lut(fn, s_in, s_out, zp_in=0, zp_out=0):
    """256-entry int8 -> int8 table, index = q_in + 128.  ('gelu', zp 0) == jepa intops.gelu_lut."""
    f = LUT_FUNCS[fn] if isinstance(fn, str) else fn
    x = (np.arange(-128, 128) - zp_in) * float(s_in)
    return np.clip(np.round(f(x) / float(s_out)) + zp_out, -128, 127).astype(I64)


def lut8(x, lut):
    return np.asarray(lut, I64)[np.asarray(x, I64) + 128]


def matmul(x, w):
    """Exact int8 x int8 -> int32 matmul. Float64 BLAS is exact while |sum| < 2^53."""
    x = np.asarray(x); w = np.asarray(w)
    return np.rint(np.matmul(np.ascontiguousarray(x, np.float64),
                             np.ascontiguousarray(w, np.float64))).astype(I64)
