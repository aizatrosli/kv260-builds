"""Quantisation schemes: activation qparams and INT8 / INT4 weight quantisation."""
import math
import numpy as np


def act_qparams(lo, hi, symmetric=True):
    """Float clip range -> (scale, zero_point) for int8.
    symmetric : scale = amax/127, zp = 0                       (jepa-tpu convention)
    asymmetric: scale = (hi-lo)/255, zp = round(-128 - lo/s)   (tpuv1 compile_mlp.quant_act)"""
    if symmetric:
        a = max(abs(lo), abs(hi), 1e-12)
        return a / 127.0, 0
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi <= lo:
        return 1.0, 0
    s = (hi - lo) / 255.0
    zp = int(np.clip(np.round(-128 - lo / s), -128, 127))
    return s, zp


def quantize_np(x, scale, zp=0, lo=-128, hi=127):
    """float -> int8 (np.round = round half to even, as jepa-tpu and tpuv1 compile_mlp)."""
    return np.clip(np.round(np.asarray(x, np.float64) / scale) + zp, lo, hi).astype(np.int64)


def dequantize_np(q, scale, zp=0):
    return (np.asarray(q, np.float64) - zp) * scale


# ------------------------------------------------------------------------------- weights -----
INT4_SCHEMES = ("channel", "group_int", "group_pow2", "group_float")


def _chan_scale(W, qmax, method):
    """Per output channel (axis 1 of W [K, N]) scale; method minmax | mse (clip-ratio search)."""
    amax = np.maximum(np.abs(W).max(0), 1e-8)
    if method == "minmax":
        return amax / qmax
    best_err = np.full(W.shape[1], np.inf); best = amax / qmax
    for r in np.linspace(0.3, 1.0, 71):
        s = r * amax / qmax
        err = ((W - np.clip(np.round(W / s), -qmax, qmax) * s) ** 2).sum(0)
        m = err < best_err
        best_err[m] = err[m]; best[m] = s[m]
    return best


def quantize_weight(W, bits=8, scheme="channel", group_size=32, method="minmax"):
    """W [K, N] float (K = reduction axis, N = output channels).

    Returns dict:
      w        int64 [K, N] values for the int8 MAC array (None if the scheme is not
               integer-representable, i.e. 'group_float'), symmetric, zp 0
      scale    float [N] per-channel scale: W ~= w * scale
      w_deq    float [K, N] dequantised weights (for fake-quant analysis)
      q4       int4 values [K, N] (bits == 4 only), gmul [K/G, N] group multipliers (group_*)
    INT4 schemes:
      channel     w4 in [-7, 7], one scale per channel                  -> w = w4
      group_int   per group G an integer multiplier M in 1..16 (4 bits): w = w4 * M  (|w| <= 112)
      group_pow2  as group_int with M in {1,2,4,8,16} (weight loader does a shift)
      group_float arbitrary float scale per group (needs per-group accumulator rescale in HW;
                  analysed in fake-quant only)
    """
    W = np.asarray(W, np.float64)
    K, N = W.shape
    if bits == 8:
        s = _chan_scale(W, 127, method)
        w = np.clip(np.round(W / s), -127, 127).astype(np.int64)
        return dict(w=w, scale=s, w_deq=w * s, bits=8, scheme="channel")
    if bits != 4:
        raise ValueError("bits must be 8 or 4")
    if scheme == "channel":
        s = _chan_scale(W, 7, method)
        q = np.clip(np.round(W / s), -7, 7).astype(np.int64)
        return dict(w=q, q4=q, scale=s, w_deq=q * s, bits=4, scheme=scheme)
    if scheme not in INT4_SCHEMES:
        raise ValueError(scheme)
    G = int(group_size)
    ng = math.ceil(K / G)
    Wp = np.zeros((ng * G, N)); Wp[:K] = W
    Wg = Wp.reshape(ng, G, N)
    amax_g = np.maximum(np.abs(Wg).max(1), 1e-8)                       # [ng, N]
    if scheme == "group_float":
        if method == "mse":
            s = np.stack([_chan_scale(Wg[g], 7, "mse") for g in range(ng)])
        else:
            s = amax_g / 7
        q = np.clip(np.round(Wg / s[:, None]), -7, 7)
        wd = (q * s[:, None]).reshape(-1, N)[:K]
        return dict(w=None, q4=q.reshape(-1, N)[:K].astype(np.int64), gscale=s, scale=s.max(0),
                    w_deq=wd, bits=4, scheme=scheme, group_size=G)
    s_c = np.maximum(np.abs(W).max(0), 1e-8) / (7 * 16)                 # [N]
    cands = np.arange(1, 17) if scheme == "group_int" else np.array([1, 2, 4, 8, 16])
    if method == "minmax":
        need = amax_g / (7 * s_c)                                       # M must be >= need
        idx = np.searchsorted(cands, need - 1e-9)
        M = cands[np.minimum(idx, len(cands) - 1)]
    else:
        errs = []
        for m in cands:
            sc = m * s_c
            errs.append(((Wg - np.clip(np.round(Wg / sc), -7, 7) * sc) ** 2).sum(1))
        M = cands[np.argmin(np.stack(errs), 0)]
    q = np.clip(np.round(Wg / (M[:, None] * s_c)), -7, 7).astype(np.int64)
    w = (q * M[:, None]).reshape(-1, N)[:K]
    return dict(w=w, q4=q.reshape(-1, N)[:K], gmul=M.astype(np.int64), scale=s_c, w_deq=w * s_c,
                bits=4, scheme=scheme, group_size=G)
