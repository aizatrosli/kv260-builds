"""Numpy reference interpreter for the IR (bit-exact integer semantics).

requant is TFLite double-rounding MultiplyByQuantizedMultiplier (same as tpuv1
model/quantization/tflite_int8.py and jepa-tpu intops.requant); host ops call jepa-tpu's
intops directly so the reference *is* the jepa integer spec.
"""
import numpy as np

from .env import jepa_modules

I64 = np.int64


def wrap32(x):
    x = np.asarray(x, I64) & 0xFFFFFFFF
    return np.where(x & 0x80000000, x - (1 << 32), x)


def _srdhm(a, b):
    ab = np.asarray(a, I64) * np.asarray(b, I64)
    num = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    q = wrap32(np.where(num >= 0, num >> 31, -((-num) >> 31)))
    both = (np.asarray(a) == -(1 << 31)) & (np.asarray(b) == -(1 << 31))
    return np.where(both, (1 << 31) - 1, q)


def _rdbpot(x, e):
    x = np.asarray(x, I64)
    e = np.asarray(e, I64)
    mask = (I64(1) << e) - 1
    return (x >> e) + ((x & mask) > ((mask >> 1) + (x < 0)))


def requant(acc, bias, m0, shift, zp_out=0, act_min=-128, act_max=127):
    """acc [..., C] int32 -> int8 (per-channel bias/m0/shift, scalar zp / clamp)."""
    x = wrap32(np.asarray(acc, I64) + np.asarray(bias, I64))
    shift = np.asarray(shift, I64)
    y = _rdbpot(_srdhm(wrap32(x << np.maximum(shift, 0)), np.asarray(m0, I64)), np.maximum(-shift, 0))
    return np.clip(wrap32(y + zp_out), act_min, act_max).astype(I64)


def conv2d(x, w, hw, stride, pad, pad_val):
    """x [H*W, Cin], w [k, k, Cin, Cout] -> [OH*OW, Cout] raw int sums; padded taps = pad_val."""
    H, W = hw
    k = w.shape[0]
    xi = np.asarray(x, I64).reshape(H, W, -1)[:, :, :w.shape[2]]
    xp = np.pad(xi, ((pad, pad), (pad, pad), (0, 0)), constant_values=pad_val)
    OH, OW = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    y = np.zeros((OH, OW, w.shape[3]), I64)
    for dy in range(k):
        for dx in range(k):
            y += xp[dy:dy + stride * OH:stride, dx:dx + stride * OW:stride, :] @ np.asarray(w[dy, dx], I64)
    return y.reshape(OH * OW, -1)


def pool2(x, hw, mode):
    H, W = hw
    t = np.asarray(x, I64).reshape(H // 2, 2, W // 2, 2, -1)
    if mode == "max":
        r = t.max(axis=(1, 3))
    else:                                   # TFLite int8 avg: round half away from zero
        s = t.sum(axis=(1, 3))
        r = np.where(s >= 0, (s + 2) // 4, -((-s + 2) // 4))
    return r.reshape((H // 2) * (W // 2), -1)


def flatten_planar(x):
    P, C = x.shape
    CT = (C + 15) // 16
    xp = np.zeros((P, CT * 16), I64)
    xp[:, :C] = x
    return xp.reshape(P, CT, 16).transpose(1, 0, 2).reshape(1, -1)


def _rq(a, acc):
    return requant(acc, a["bias"], a["m0"], a["shift"], a.get("zp_out", 0),
                   a.get("act_min", -128), a.get("act_max", 127))


def eval_op(op, vals, g):
    I = jepa_modules()
    a = op.attrs
    x = [vals[i] if i in vals else g.T(i).const for i in op.inputs]
    k = op.kind
    if k in ("matmul", "tpu_matmul"):
        w = x[1].T if a.get("transpose_b") else x[1]
        acc = np.asarray(x[0], I64) @ np.asarray(w, I64)
        return [acc if k == "matmul" else _rq(a, acc)]
    if k in ("conv2d", "tpu_conv"):
        acc = conv2d(x[0], x[1], a["hw"], a["stride"], a["pad"], a["pad_val"])
        if k == "conv2d":
            return [acc]
        y = _rq(a, acc)
        if a.get("pool"):
            y = pool2(y, a["out_hw"], a["pool"])
        return [y]
    if k == "requant":
        return [_rq(a, x[0])]
    if k == "clamp":
        return [np.clip(x[0], a["lo"], a["hi"])]
    if k == "pool2":
        return [pool2(x[0], a["hw"], a["mode"])]
    if k == "layernorm":
        return [I.layernorm(x[0])]
    if k == "softmax":
        return [I.softmax(x[0], a["lut"])]
    if k == "lut":
        return [I.lut8(x[0], a["table"])]
    if k == "add":
        return [I.ew_add(x[0], x[1], a["ma"], a["mb"])]
    if k == "pack_weights":
        return [x[0].T if a.get("transpose") else x[0]]
    if k == "slice":
        return [x[0][:, a["c0"]:a["c1"]]]
    if k == "concat":
        return [np.concatenate(x, axis=1)]
    if k == "flatten_planar":
        return [flatten_planar(x[0])]
    raise NotImplementedError(k)


def run(g, feeds):
    """Execute the graph; returns dict tensor name -> numpy int64 array (every tensor)."""
    vals = {k: np.asarray(v, I64) for k, v in feeds.items()}
    for op in g.ops:
        outs = eval_op(op, vals, g)
        for name, v in zip(op.outputs, outs):
            v = np.asarray(v, I64)
            if v.shape != g.T(name).shape:
                raise AssertionError(f"{op.name}: ref shape {v.shape} != IR {g.T(name).shape}")
            vals[name] = v
    return vals
