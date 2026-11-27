"""Host implementations of the JEPA non-matmul ops, bit-exact with jepa-tpu/intops.py.

All ops work on the TPU's tiled activation layout x[T, M, 16] (== token-major X[M, 16T]), which is
exactly what WRITE_HOST_MEM produces and READ_HOST_MEM consumes, so results go straight back into
DMA buffers with no transposes.  Two implementations with one interface:

  NumpyOps  vectorised numpy (always available)
  COps      csrc/tpurt_host.c via ctypes (faster for small tensors / on the Cortex-A53; the GIL is
            released during the call so it overlaps with device jobs)

get_ops("auto") returns COps when the shared library is (or can be) built, else NumpyOps.
"""
import ctypes
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

LN_Q, LN_K, SM_K, EW_SH = 32, 24, 22, 16
LIB = Path(__file__).resolve().parent / "_lib" / f"libtpurt_host-{platform.machine()}.so"
I8, I32, I64 = np.int8, np.int32, np.int64


def isqrt(v):
    v = np.asarray(v, I64)
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(I64)
    r = np.where(r * r > v, r - 1, r)
    return np.where((r + 1) * (r + 1) <= v, r + 1, r)


def tile(X):
    """token-major [M, 16T] -> tiled [T, M, 16] (copy)."""
    X = np.asarray(X)
    M, D = X.shape
    return np.ascontiguousarray(X.reshape(M, D // 16, 16).transpose(1, 0, 2))


def untile(x):
    """tiled [T, M, 16] -> token-major [M, 16T] (copy)."""
    T, M, _ = x.shape
    return np.ascontiguousarray(x.transpose(1, 0, 2).reshape(M, 16 * T))


def _out(out, shape):
    return np.empty(shape, I8) if out is None else out


class NumpyOps:
    name = "numpy"

    def layernorm(self, x, out=None):
        T, M, _ = x.shape
        D = 16 * T
        xi = x.astype(I32)
        S = xi.sum(axis=(0, 2), dtype=I32)
        d = D * xi - S[None, :, None]                       # |d| <= 2^15: int32
        d64 = d.astype(I64)
        Q = np.einsum("tml,tml->m", d64, d64)
        sig = np.maximum(isqrt(Q // D), 1)
        inv = (I64(1) << LN_K) // sig
        v = d64 * (LN_Q * inv)[None, :, None]
        y = np.clip((v + (1 << (LN_K - 1))) >> LN_K, -128, 127)
        o = _out(out, x.shape)
        o[...] = y
        return o

    def softmax(self, s, lut, out=None):
        """s [G, T, M, 16]: softmax over the T*16 keys of each (g, m) row."""
        lut = np.asarray(lut, I64)
        s32 = s.astype(I32)
        m = s32.max(axis=(1, 3), keepdims=True)
        idx = m - s32
        small = int(lut.max()) * ((127 << SM_K) // max(int(lut[0]), 1)) + (1 << SM_K) < (1 << 31)
        e = lut.astype(I32 if small else I64)[idx]
        Z = e.sum(axis=(1, 3), keepdims=True, dtype=I64)
        inv = ((I64(127) << SM_K) // Z).astype(e.dtype)
        p = (e * inv + (1 << (SM_K - 1))) >> SM_K
        o = _out(out, s.shape)
        o[...] = p
        return o

    def lut8(self, x, lut, out=None):
        lut8 = np.asarray(lut, I64).astype(I8)
        o = _out(out, x.shape)
        o[...] = lut8[x.astype(np.intp) + 128]
        return o

    def ew_add(self, a, b, ma, mb, out=None):
        ma, mb = int(ma), int(mb)
        dt = I32 if (abs(ma) + abs(mb)) * 128 + (1 << EW_SH) < (1 << 31) else I64
        v = a.astype(dt) * dt(ma) + np.asarray(b).astype(dt) * dt(mb)
        y = np.clip((v + (1 << (EW_SH - 1))) >> EW_SH, -128, 127)
        o = _out(out, np.broadcast_shapes(a.shape, np.shape(b)))
        o[...] = y
        return o

    def pack_kv(self, qkv, B, H, kT_out=None, v_out=None):
        """qkv tiled [3*Td, M, 16] -> (kT tiles [B*H, 4, KTd, 16, 16], v tiles [B*H, KTd, 4, 16, 16]),
        tpuv1 weight-tile byte order (word w = W_tile[15 - w])."""
        T3, M, _ = qkv.shape
        Td = T3 // 3
        KTd = Td // H
        k = qkv[Td:2 * Td].reshape(H, KTd, B, 4, 16, 16)       # h, kt(dim), b, nt(token), n(token), k(dim)
        v = qkv[2 * Td:].reshape(H, KTd, B, 4, 16, 16)         # h, nt(dim), b, kt(token), k(token), n(dim)
        # kT tile (nt, kt)[k][n] = k[token 16nt+n][dim 16kt+k]; stored rows reversed
        kT = k.transpose(2, 0, 3, 1, 5, 4)[..., ::-1, :]       # b, h, nt, kt, k, n
        vv = v.transpose(2, 0, 1, 3, 4, 5)[..., ::-1, :]       # b, h, nt, kt, k, n
        ko = np.empty((B * H, 4, KTd, 16, 16), I8) if kT_out is None else kT_out
        vo = np.empty((B * H, KTd, 4, 16, 16), I8) if v_out is None else v_out
        ko[...] = kT.reshape(B * H, 4, KTd, 16, 16)
        vo[...] = vv.reshape(B * H, KTd, 4, 16, 16)
        return ko, vo


class COps(NumpyOps):
    """ctypes wrapper of libtpurt_host.so. Inputs must be C-contiguous int8 (DMA views are)."""
    name = "c"

    def __init__(self, path=LIB):
        self.lib = ctypes.CDLL(str(path))
        P, I, Z, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_int64
        sig = dict(tpurt_layernorm=[P, P, I, I], tpurt_softmax=[P, P, I, I, I, P], tpurt_lut8=[P, P, Z, P],
                   tpurt_ew_add=[P, P, P, Z, L, L], tpurt_ew_add_bcast=[P, P, P, Z, Z, L, L],
                   tpurt_pack_kv=[P, P, P, I, I, I, I])
        for fn, args in sig.items():
            f = getattr(self.lib, fn)
            f.argtypes, f.restype = args, None
        self._luts = {}

    @staticmethod
    def _c(a):
        if a.dtype != I8 or not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=I8)
        return a

    def _lut(self, lut, dtype):
        key = (id(lut), dtype)
        hit = self._luts.get(key)
        if hit is None or hit[0] is not lut:
            hit = (lut, np.ascontiguousarray(np.asarray(lut, I64).astype(dtype)))
            self._luts[key] = hit
        return hit[1]

    def layernorm(self, x, out=None):
        x = self._c(x)
        o = _out(out, x.shape)
        self.lib.tpurt_layernorm(x.ctypes.data, o.ctypes.data, x.shape[0], x.shape[1])
        return o

    def softmax(self, s, lut, out=None):
        s = self._c(s)
        o = _out(out, s.shape)
        G, T, M, _ = s.shape
        self.lib.tpurt_softmax(s.ctypes.data, o.ctypes.data, G, T, M, self._lut(lut, I32).ctypes.data)
        return o

    def lut8(self, x, lut, out=None):
        x = self._c(x)
        o = _out(out, x.shape)
        self.lib.tpurt_lut8(x.ctypes.data, o.ctypes.data, x.size, self._lut(lut, I8).ctypes.data)
        return o

    def ew_add(self, a, b, ma, mb, out=None):
        a = self._c(a)
        b = self._c(np.asarray(b))
        o = _out(out, a.shape)
        if b.shape == a.shape:
            self.lib.tpurt_ew_add(a.ctypes.data, b.ctypes.data, o.ctypes.data, a.size, int(ma), int(mb))
        elif 0 < b.ndim <= a.ndim and a.shape[a.ndim - b.ndim:] == b.shape:
            self.lib.tpurt_ew_add_bcast(a.ctypes.data, b.ctypes.data, o.ctypes.data, a.size, b.size, int(ma), int(mb))
        else:
            return NumpyOps.ew_add(self, a, b, ma, mb, out=o)
        return o

    def pack_kv(self, qkv, B, H, kT_out=None, v_out=None):
        qkv = self._c(qkv)
        T3, M, _ = qkv.shape
        Td = T3 // 3
        KTd = Td // H
        ko = np.empty((B * H, 4, KTd, 16, 16), I8) if kT_out is None else kT_out
        vo = np.empty((B * H, KTd, 4, 16, 16), I8) if v_out is None else v_out
        assert ko.flags.c_contiguous and vo.flags.c_contiguous
        self.lib.tpurt_pack_kv(qkv.ctypes.data, ko.ctypes.data, vo.ctypes.data, B, H, M, 16 * Td)
        return ko, vo


def build_c(quiet=True):
    """Compile the C kernels with `make` (needs gcc). Returns True on success."""
    root = Path(__file__).resolve().parents[1]
    try:
        r = subprocess.run(["make", "-C", str(root)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if r.returncode and not quiet:
        print(r.stdout, r.stderr)
    return r.returncode == 0 and LIB.exists()


def get_ops(kind="auto"):
    """'numpy', 'c' (build if needed, raise if impossible) or 'auto' (c if possible)."""
    if kind == "numpy" or os.environ.get("TPURT_NO_C"):
        if kind == "c":
            raise RuntimeError("TPURT_NO_C is set")
        return NumpyOps()
    if not LIB.exists() or LIB.stat().st_mtime < (Path(__file__).resolve().parents[1] / "csrc/tpurt_host.c").stat().st_mtime:
        build_c()
    try:
        return COps()
    except OSError:
        if kind == "c":
            raise RuntimeError(f"C host kernels unavailable ({LIB}); run `make` in {LIB.parents[2]}")
        return NumpyOps()
