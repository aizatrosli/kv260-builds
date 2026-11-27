"""INT4 weight helpers: nibble packing and a hardware-cost analysis for tpuv1."""
import math
import numpy as np


def pack_int4(q, axis=-1):
    """Pack int4 values (-8..7) two per byte along `axis` (padded to even length).
    Byte j holds element 2j in bits [3:0] and element 2j+1 in bits [7:4] (two's complement)."""
    q = np.asarray(q, np.int64)
    if q.min(initial=0) < -8 or q.max(initial=0) > 7:
        raise ValueError("int4 values must be in [-8, 7]")
    q = np.moveaxis(q, axis, -1)
    if q.shape[-1] % 2:
        q = np.concatenate([q, np.zeros(q.shape[:-1] + (1,), np.int64)], -1)
    u = (q & 0xF).astype(np.uint8)
    out = u[..., 0::2] | (u[..., 1::2] << 4)
    return np.moveaxis(out, -1, axis)


def unpack_int4(p, n, axis=-1):
    """Inverse of pack_int4; n = original length along axis."""
    p = np.moveaxis(np.asarray(p, np.uint8), axis, -1)
    lo = (p & 0xF).astype(np.int64); hi = (p >> 4).astype(np.int64)
    q = np.stack([lo, hi], -1).reshape(p.shape[:-1] + (-1,))[..., :n]
    q = np.where(q >= 8, q - 16, q)
    return np.moveaxis(q, -1, axis)


def pack_uint4(u, axis=-1):
    """Pack unsigned 0..15 values (e.g. group multiplier M-1) two per byte."""
    u = np.asarray(u, np.int64)
    if u.min(initial=0) < 0 or u.max(initial=0) > 15:
        raise ValueError("uint4 values must be in [0, 15]")
    return pack_int4(np.where(u >= 8, u - 16, u), axis)


def unpack_uint4(p, n, axis=-1):
    return unpack_int4(p, n, axis) & 0xF


def expand_group_weights(q4, gmul, group_size):
    """Weight-loader model: int8 word = int4 value * group multiplier (what the MAC array sees)."""
    K = q4.shape[0]
    M = np.repeat(np.asarray(gmul, np.int64), group_size, axis=0)[:K]
    return np.asarray(q4, np.int64) * M


KV260_FREE_BRAM_KIB = (23.6 - 10.8) * 1024 / 8   # free on-chip RAM next to tpuv1 (jepa-tpu README)


def hardware_report(graph_int8, graph_int4=None):
    """Memory footprint of the weights at INT8 vs INT4 and what tpuv1 needs to support INT4."""
    def wb(g):
        b = 0; meta = 0
        for n in g.weight_nodes():
            bits = n.attrs.get("weight_bits", 8)
            b += math.ceil(n.params["w"].size * bits / 8)
            if "gmul" in n.params:
                meta += math.ceil(n.params["gmul"].size / 2)
            meta += n.params["m0"].size * 4 + n.params["shift"].size + n.params["bias"].size * 4
        return b, meta
    w8, m8 = wb(graph_int8)
    rep = dict(int8_weight_kib=w8 / 1024, requant_param_kib=m8 / 1024,
               kv260_free_onchip_kib=KV260_FREE_BRAM_KIB)
    if graph_int4 is not None:
        w4, m4 = wb(graph_int4)
        rep.update(int4_weight_kib=w4 / 1024, int4_meta_kib=m4 / 1024,
                   saving_kib=(w8 - w4) / 1024, ratio=(w4 + m4) / max(w8 + m8, 1))
    rep["hardware_changes"] = HW_NOTES
    return rep


HW_NOTES = {
    "mac_array": "unchanged: int4 weights are expanded to int8 (q4 * M) before the 16x16 array",
    "weight_fifo": "a 16-column weight row becomes a 64-bit word (16 nibbles) instead of 128 bits; "
                   "add a 4->8 sign-extend (+ x M_g multiply or << e shift) stage in the weight loader",
    "group_int": "per (group, column) 4-bit multiplier M in 1..16: a 4x5-bit multiplier per lane; "
                 "group boundary must align with the 16-row K tiles (G multiple of 16)",
    "group_pow2": "M in {1,2,4,8,16}: a barrel shift per lane (cheapest)",
    "group_float": "true per-group float scales need per-group partial sums: either a requant per "
                   "K-group into a wider accumulator (extra ACC traffic + 16 multipliers), or "
                   "dequant to int16 weights (doubles the array width). Not supported in the int graph.",
    "requant": "unchanged: per-channel m0/shift absorb the channel scale",
    "throughput": "weight-stationary loads are ~1/4 of matmul cycles for this model; halving weight "
                  "bandwidth mainly removes the AXI weight stalls seen without the ROM path",
}
