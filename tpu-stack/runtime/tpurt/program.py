"""Tiny graph compiler: a list of steps (GEMM on the systolic array, VECTOR on the future VPU, or a
host function) -> a schedule of TPU programs (<= 2048 instructions each) and host calls.

Tensors live in the TPU's tiled activation layout (rows of 16 int8 lanes; a [M, 16T] activation is
T tiles of M rows).  Inside a program, tensors stay in the Unified Buffer; the builder loads (RHM) a
tensor the first time a step needs rows it does not hold, and stores (WHM) at the end of the
program every row produced there that a later program or a host step reads.  Programs are split
automatically when the next step would exceed the instruction FIFO or the UB.

Hazards follow tpuv1's rules (model/isa/sim.py checks them on every simulated run):
  SYNC(U_MATRIX) before each ACTIVATE; SYNC(U_ACT) before a MATMUL reads rows an unsynced ACTIVATE
  writes, before reusing an ACC half or a qset an unsynced ACTIVATE uses; SYNC(U_DMA) after loads;
  SYNC(U_ALL) before stores and before each VECTOR.
SET_CONFIG traffic is minimised by caching the 4 requant parameter sets (qsets) per program.

Instructions carry DDR address 0 and a symbolic relocation (index, tensor, byte offset); a context
resolves tensors to DMA blocks, so the same compiled program runs on any double-buffer context.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import isa, vpu

N = 16
TILE_BYTES = 256


# ---------------------------------------------------------------------------------------------
class Tensor:
    """kind: 'act' (per context), 'const' (shared, written once: weights, LUTs, pos-embedding),
    'out' (per context, always stored)."""

    def __init__(self, name, rows, kind="act", data=None):
        self.name, self.rows, self.kind, self.data = name, int(rows), kind, data

    @property
    def nbytes(self):
        return self.rows * N

    def __repr__(self):
        return f"Tensor({self.name}, rows={self.rows}, {self.kind})"


@dataclass(eq=False)
class Gemm:
    name: str
    x: Tensor
    x_off: int
    x_stride: int
    KT: int
    m: int
    out: Tensor
    o_off: int
    o_stride: int
    NT: int
    w: Tensor                 # weight tiles, consumption order (nt outer, kt inner), 256 B each
    w_off: int                # byte offset of the first tile
    m0: np.ndarray            # [16*NT]
    shift: np.ndarray
    bias: np.ndarray
    x_load: Optional[tuple] = None     # rows (lo, hi) of x to load if not resident (default: all)
    op: str = ""              # profiling label
    act: tuple = (0, -128, 127)        # (zp_out, act_min, act_max): ReLU = (0, 0, 127)

    def reads(self):
        return [self.x, self.w]

    def writes(self):
        return [self.out]


@dataclass(eq=False)
class Vec:
    name: str
    func: int                 # vpu.F_ADD / F_LUT / F_LN / F_SM
    a: Tensor
    a_off: int
    out: Tensor
    o_off: int
    count: int
    chunks: int
    stride: int
    b: Optional[Tensor] = None
    b_off: int = 0
    mult: tuple = (0, 0)
    lut: Optional[Tensor] = None
    op: str = ""

    def reads(self):
        return [t for t in (self.a, self.b, self.lut) if t is not None]

    def writes(self):
        return [self.out]


@dataclass(eq=False)
class Host:
    name: str
    fn: Callable
    reads_: list = field(default_factory=list)
    writes_: list = field(default_factory=list)
    op: str = ""

    def reads(self):
        return self.reads_

    def writes(self):
        return self.writes_


class ProgramTooLong(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
def pack_weight_tiles(W):
    """W [K, N] int (K, N multiples of 16) -> bytes: tiles nt outer, kt inner; word t = W_tile[15-t]."""
    W = np.asarray(W)
    K, Nn = W.shape
    if K % N or Nn % N:
        raise ValueError(f"weight shape {W.shape} must be multiples of 16")
    if W.min() < -128 or W.max() > 127:
        raise ValueError("weights must be int8")
    t = W.astype(np.int8).reshape(K // N, N, Nn // N, N).transpose(2, 0, 1, 3)[:, :, ::-1, :]
    return np.ascontiguousarray(t).tobytes()


def _runs(mask, base=0):
    """Contiguous True runs of a bool array -> [(lo, hi)] (offset by base)."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return [(int(a) + base, int(b) + base) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


@dataclass
class Program:
    words: list
    relocs: list              # (instr_index, Tensor, byte_offset)
    steps: list               # step names
    loads: list               # Tensors read from DDR
    stores: list              # Tensors written to DDR
    ops: list                 # profiling labels of the steps

    @property
    def n_instr(self):
        return len(self.words)

    def to_bytes(self):
        return isa.to_bytes(self.words)

    def label(self):
        lab = []
        for o in self.ops:
            if not lab or lab[-1] != o:
                lab.append(o)
        return "+".join(lab)

    def disasm(self):
        from . import _paths
        _paths.need_tpuv1_model()
        import asm
        out = []
        for i, w in enumerate(self.words):
            out.append(f"{i:5d} {vpu.FUNC_NAMES[(w >> 8) & 15].upper() + ' (VECTOR)' if w & 255 == vpu.OP_VECTOR else asm.disasm(w)}")
        return "\n".join(out)


class ProgramBuilder:
    def __init__(self, limit=isa.MAX_INSTR, rw_group=False, tag=0x7A0B0001):
        self.limit, self.rw_group, self.tag = limit, rw_group, tag
        self.words, self.relocs, self.steps, self.ops = [], [], [], []
        self.ub_base, self.ub_next = {}, 0
        self.valid, self.dirty = {}, {}
        self.loaded = []
        self.qsets = [None] * 4           # cached (m0 tuple, shift tuple, bias tuple, zp/min/max)
        self.qrr = 0
        self.qset_busy, self.half_busy = set(), [False, False]
        self.act_written = []             # UB intervals written by unsynced ACTIVATEs
        self.tile_idx = 0
        self.vpu_mult = None
        self.lut_loaded = {}
        self.last_step = -1

    # ---- snapshot / rollback --------------------------------------------------------------
    def _snap(self):
        return (len(self.words), len(self.relocs), len(self.steps), len(self.ops), dict(self.ub_base), self.ub_next,
                {k: v.copy() for k, v in self.valid.items()}, {k: v.copy() for k, v in self.dirty.items()},
                list(self.loaded), list(self.qsets), self.qrr, set(self.qset_busy), list(self.half_busy),
                list(self.act_written), self.tile_idx, self.vpu_mult, dict(self.lut_loaded), self.last_step)

    def _restore(self, s):
        (nw, nr, ns, no, self.ub_base, self.ub_next, self.valid, self.dirty, self.loaded, self.qsets, self.qrr,
         self.qset_busy, self.half_busy, self.act_written, self.tile_idx, self.vpu_mult, self.lut_loaded,
         self.last_step) = s
        del self.words[nw:], self.relocs[nr:], self.steps[ns:], self.ops[no:]

    # ---- primitives -----------------------------------------------------------------------
    def _emit(self, w, reloc=None):
        if reloc is not None:
            self.relocs.append((len(self.words), reloc[0], reloc[1]))
        self.words.append(w)

    def _sync(self, mask):
        self._emit(isa.sync(mask))
        if mask & isa.U_ACT:
            self.qset_busy.clear()
            self.half_busy = [False, False]
            self.act_written = []

    def _alloc(self, t):
        if t not in self.ub_base:
            if self.ub_next + t.rows > isa.UB_ROWS:
                raise ProgramTooLong(f"UB full ({self.ub_next} + {t.rows} rows for {t.name})")
            self.ub_base[t] = self.ub_next
            self.ub_next += t.rows
            self.valid[t] = np.zeros(t.rows, bool)
            self.dirty[t] = np.zeros(t.rows, bool)
        return self.ub_base[t]

    def _ensure(self, needs):
        """needs: [(tensor, lo, hi)] rows that must be valid in UB; loads missing ones."""
        loads = []
        for t, lo, hi in needs:
            base = self._alloc(t)
            loads += [(t, a, b) for a, b in _runs(~self.valid[t][lo:hi], lo)]
        if not loads:
            return
        if len(self.words):
            self._sync(isa.U_ALL)
        for t, a, b in loads:
            base = self.ub_base[t]
            for r in range(a, b, isa.MAX_DMA_ROWS):
                n = min(isa.MAX_DMA_ROWS, b - r)
                self._emit(isa.read_host_mem(base + r, 0, n), (t, r * N))
            self.valid[t][a:b] = True
            if t not in self.loaded:
                self.loaded.append(t)
        self._emit(isa.sync(isa.U_DMA))

    def _produce(self, t, rows):
        self._alloc(t)
        self.valid[t][rows] = True
        self.dirty[t][rows] = True

    # ---- GEMM -----------------------------------------------------------------------------
    def _qset_for(self, key):
        for q, k in enumerate(self.qsets):
            if k is not None and k == key:
                return q, []
        q = self.qrr % 4
        self.qrr += 1
        if q in self.qset_busy:
            self._sync(isa.U_ACT)
        old = self.qsets[q]
        cfg = []
        for c in range(N):
            if old is None or (old[0][c], old[1][c]) != (key[0][c], key[1][c]):
                cfg.append(isa.set_config(0x1000 | (q << 4) | c, (key[0][c] & 0xFFFFFFFF) | ((key[1][c] & 0xFF) << 32)))
            if old is None or old[2][c] != key[2][c]:
                cfg.append(isa.set_config(0x2000 | (q << 4) | c, key[2][c] & 0xFFFFFFFF))
        if old is None or old[3] != key[3]:
            zp, lo, hi = key[3]
            cfg.append(isa.set_config(0x3000 | q, (zp & 0xFF) | ((lo & 0xFF) << 8) | ((hi & 0xFF) << 16)))
        self.qsets[q] = key
        return q, cfg

    def _gemm(self, g):
        if not 1 <= g.m <= isa.ACC_HALF:
            raise ValueError(f"{g.name}: m={g.m} outside 1..{isa.ACC_HALF}")
        x_lo, x_hi = g.x_load or (g.x_off, g.x_off + (g.KT - 1) * g.x_stride + g.m)
        self._ensure([(g.x, x_lo, x_hi)])
        xb = self.ub_base[g.x]
        ob = self._alloc(g.out)
        x_rows = [(xb + g.x_off + kt * g.x_stride, xb + g.x_off + kt * g.x_stride + g.m) for kt in range(g.KT)]
        if any(a < hi and lo < b for a, b in self.act_written for lo, hi in x_rows):
            self._sync(isa.U_ACT)
        for nt in range(g.NT):
            ch = slice(N * nt, N * nt + N)
            key = (tuple(int(v) for v in g.m0[ch]), tuple(int(v) for v in g.shift[ch]),
                   tuple(int(v) for v in g.bias[ch]), tuple(g.act))
            q, cfg = self._qset_for(key)
            for w in cfg:
                self._emit(w)
            half = self.tile_idx % 2
            if self.half_busy[half]:
                self._sync(isa.U_ACT)
            a = half * isa.ACC_HALF
            tile0 = g.w_off + nt * g.KT * TILE_BYTES
            if self.rw_group:
                self._emit(isa.read_weights(0, g.KT), (g.w, tile0))
            for kt in range(g.KT):
                if not self.rw_group:
                    self._emit(isa.read_weights(0, 1), (g.w, tile0 + kt * TILE_BYTES))
                self._emit(isa.matmul(x_rows[kt][0], a, g.m, acc=1 if kt else 0, new_weights=1))
            self._emit(isa.sync(isa.U_MATRIX))
            o0 = g.o_off + nt * g.o_stride
            self._emit(isa.activate(a, ob + o0, g.m, q))
            self._produce(g.out, slice(o0, o0 + g.m))
            self.qset_busy.add(q)
            self.half_busy[half] = True
            self.act_written.append((ob + o0, ob + o0 + g.m))
            self.tile_idx += 1

    # ---- VECTOR (provisional VPU) --------------------------------------------------------------
    def _vec(self, v):
        span = (v.chunks - 1) * v.stride + v.count
        needs = [(v.a, v.a_off, v.a_off + span)]
        if v.func == vpu.F_ADD:
            needs.append((v.b, v.b_off, v.b_off + span))
        if v.func in (vpu.F_SM, vpu.F_LUT):
            ld = vpu.F_LDE if v.func == vpu.F_SM else vpu.F_LDG
            if self.lut_loaded.get(ld) is not v.lut:
                self._ensure([(v.lut, 0, v.lut.rows)])
                self._sync(isa.U_ALL)
                self._emit(vpu.vector(ld, self.ub_base[v.lut]))
                self.lut_loaded[ld] = v.lut
        self._ensure(needs)
        self._sync(isa.U_ALL)
        if v.func == vpu.F_ADD and self.vpu_mult != tuple(v.mult):
            self._emit(isa.set_config(vpu.VPU_MULT, vpu.mult_value(*v.mult)))
            self.vpu_mult = tuple(v.mult)
        ob = self._alloc(v.out)
        self._emit(vpu.vector(v.func, self.ub_base[v.a] + v.a_off, ob + v.o_off, v.count, v.chunks, v.stride,
                              self.ub_base[v.b] + v.b_off if v.b is not None else 0))
        rows = (v.o_off + np.arange(v.chunks)[:, None] * v.stride + np.arange(v.count)[None, :]).ravel()
        self._produce(v.out, rows)

    # ---- program assembly -----------------------------------------------------------------------
    def _store_plan(self, keep):
        plan = []
        for t, d in self.dirty.items():
            if keep(t):
                plan += [(t, a, b) for a, b in _runs(d)]
        return plan

    def _close_cost(self, keep):
        return sum((b - a + isa.MAX_DMA_ROWS - 1) // isa.MAX_DMA_ROWS for _, a, b in self._store_plan(keep)) + 4

    def try_add(self, step, index, keep):
        snap = self._snap()
        try:
            (self._gemm if isinstance(step, Gemm) else self._vec)(step)
            self.steps.append(step.name)
            self.ops.append(step.op or step.name)
            self.last_step = index
            if len(self.words) + self._close_cost(keep) > self.limit:
                raise ProgramTooLong("instruction FIFO")
            return True
        except ProgramTooLong:
            self._restore(snap)
            return False

    def close(self, keep):
        self._sync(isa.U_ALL)
        stores = []
        for t, a, b in self._store_plan(keep):
            base = self.ub_base[t]
            for r in range(a, b, isa.MAX_DMA_ROWS):
                n = min(isa.MAX_DMA_ROWS, b - r)
                self._emit(isa.write_host_mem(base + r, 0, n), (t, r * N))
            if t not in stores:
                stores.append(t)
        self._emit(isa.sync(isa.U_DMA))
        self._emit(isa.interrupt_host(self.tag))
        self._emit(isa.halt())
        assert len(self.words) <= self.limit
        return Program(self.words, self.relocs, self.steps, self.loaded, stores, self.ops)


def schedule(steps, limit=isa.MAX_INSTR, rw_group=False):
    """steps -> [('dev', Program) | ('host', Host)] in execution order."""
    readers = {}
    for i, st in enumerate(steps):
        for t in st.reads():
            readers.setdefault(t, []).append(i)
    items, pb = [], None

    def close():
        nonlocal pb
        if pb is not None:
            last = pb.last_step
            keep = lambda t: t.kind == "out" or any(r > last for r in readers.get(t, []))
            items.append(("dev", pb.close(keep)))
            pb = None

    for i, st in enumerate(steps):
        if isinstance(st, Host):
            close()
            items.append(("host", st))
            continue
        keep_now = lambda t, i=i: t.kind == "out" or any(r > i for r in readers.get(t, []))
        if pb is None:
            pb = ProgramBuilder(limit, rw_group)
        if not pb.try_add(st, i, keep_now):
            close()
            pb = ProgramBuilder(limit, rw_group)
            if not pb.try_add(st, i, keep_now):
                raise ProgramTooLong(f"step {st.name} alone does not fit one program ({limit} instructions / UB)")
    close()
    return items


def split_gemm_nt(g, max_nt):
    """Split a GEMM into column chunks of <= max_nt tiles (each fits one program)."""
    if g.NT <= max_nt:
        return [g]
    out = []
    for n0 in range(0, g.NT, max_nt):
        n1 = min(g.NT, n0 + max_nt)
        ch = slice(16 * n0, 16 * n1)
        out.append(Gemm(f"{g.name}[{n0}:{n1}]", g.x, g.x_off, g.x_stride, g.KT, g.m, g.out,
                        g.o_off + n0 * g.o_stride, g.o_stride, n1 - n0, g.w, g.w_off + n0 * g.KT * TILE_BYTES,
                        g.m0[ch], g.shift[ch], g.bias[ch], g.x_load, g.op, g.act))
    return out
