"""Instruction generation: scheduled IR graph -> tpuv1 ISA kernels (via tpuv1's asm.py).

A compiled program is a sequence of steps: TPU *kernels* (each <= 2048 instructions, the
Instruction FIFO size, ending SYNC / INTERRUPT_HOST / HALT) and *host* steps (ops tpuv1 cannot
run). Data crosses the boundary through DDR: WRITE_HOST_MEM at the end of a kernel for tensors
the host needs, READ_HOST_MEM at the start of a kernel for host results the TPU needs.
Everything else stays in the Unified Buffer.

Per output tile the emitter:
* loads the 33 requantisation registers (16 x M0/shift, 16 x bias, zp/min/max) into one of the
  four parameter sets - **only the registers whose value differs** from what that set already
  holds (and nothing if some set already holds the exact config);
* requests weight tiles with READ_WEIGHTS lazily, never more than the 64-tile Weight FIFO;
* emits MATMUL/CONVOLVE then ACTIVATE (requant + activation (+ pool) straight into the UB);
* inserts SYNCs from an exact model of the core's hazards (same rules as tpuv1 sim.py).
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .env import asm, IFIFO_WORDS, ACC_ROWS, NSETS, WFIFO_TILES
from . import memory as M
from . import tiling
from .passes import ACT_FUNC, device_plan

U = asm
DMA_MAX_ROWS = 4095


# ---------------------------------------------------------------------------------------------
# weight tiles
# ---------------------------------------------------------------------------------------------
def tiles_matmul(W):
    """W [K, N] -> tile array [NT*KT, 16, 16] in stream order (nt outer, kt inner)."""
    K, N = W.shape
    KT, NT = -(-K // 16), -(-N // 16)
    Wp = np.zeros((KT * 16, NT * 16), np.int64)
    Wp[:K, :N] = W
    return Wp.reshape(KT, 16, NT, 16).transpose(2, 0, 1, 3).reshape(NT * KT, 16, 16)


def tiles_conv(W):
    """W [k, k, Cin, Cout] -> tiles in stream order (nt, kh, kw, ct)."""
    k, _, cin, cout = W.shape
    CT, NT = -(-cin // 16), -(-cout // 16)
    Wp = np.zeros((k, k, CT * 16, NT * 16), np.int64)
    Wp[:, :, :cin, :cout] = W
    t = Wp.reshape(k, k, CT, 16, NT, 16).transpose(4, 0, 1, 2, 3, 5)
    return t.reshape(NT * k * k * CT, 16, 16)


def tile_bytes(tiles):
    """Weight FIFO image: word t of a tile = W_tile[15 - t], byte n = column n."""
    return np.ascontiguousarray(tiles[:, ::-1, :]).astype(np.int8).tobytes()


# ---------------------------------------------------------------------------------------------
@dataclass
class OpStats:
    name: str
    kind: str
    shape: str
    plan: str
    kernel: int = -1
    counts: Counter = field(default_factory=Counter)
    weight_tiles: int = 0
    macs: int = 0
    array_cycles: int = 0
    act_rows: int = 0
    dma_bytes: int = 0

    @property
    def instrs(self):
        return sum(self.counts.values())

    @property
    def weight_bytes(self):
        return self.weight_tiles * 256


@dataclass
class Kernel:
    index: int
    step: int
    words: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    done: dict = field(default_factory=dict)       # op -> instr index after which its output is final


@dataclass
class Program:
    graph: object
    steps: list                                     # [("host", [op names]) | ("kernel", Kernel)]
    mem: object
    ddr_init: dict                                  # address -> bytes (weights, host constants)
    wregion: dict                                   # tensor -> DDR address of its weight tiles
    stats: dict                                     # op name -> OpStats (kernel ops + "<kernel>" overhead)
    options: dict

    @property
    def kernels(self):
        return [k for kind, k in self.steps if kind == "kernel"]

    @property
    def n_instrs(self):
        return sum(len(k.words) for k in self.kernels)

    @property
    def n_bytes(self):
        return 16 * self.n_instrs

    def opcode_counts(self):
        c = Counter()
        for k in self.kernels:
            for w in k.words:
                c[U.NAMES[w & 0xFF]] += 1
        return c


# ---------------------------------------------------------------------------------------------
class Emitter:
    """Tracks core state (parameter sets, conv geometry, hazards, ACC ring, weight FIFO)."""

    def __init__(self, dedupe=True):
        self.dedupe = dedupe
        self.qregs = [dict() for _ in range(NSETS)]
        self.qlru = [0] * NSETS
        self.tick = 0
        self.rr = 0
        self.geo = {}
        self.acc_cursor = 0
        self.words = None
        self.stats = None
        self.prev_stream = 0
        self.reset_hazards()

    def reset_hazards(self):
        self.mat_ub_rd, self.mat_acc_wr = set(), set()
        self.act_acc_rd, self.act_ub_wr, self.act_q = set(), set(), set()
        self.conv_busy = self.pool_busy = False
        self.prev_stream = 0

    def emit(self, w):
        self.words.append(w)
        self.stats.counts[U.NAMES[w & 0xFF]] += 1

    def sync(self, mask):
        self.emit(U.sync(mask))
        if mask & U.U_MATRIX:
            self.mat_ub_rd.clear(); self.mat_acc_wr.clear(); self.conv_busy = False
            self.stats.array_cycles += tiling.DRAIN
            self.prev_stream = 0
        if mask & U.U_ACT:
            self.act_acc_rd.clear(); self.act_ub_wr.clear(); self.act_q.clear(); self.pool_busy = False

    # ---- array ----------------------------------------------------------------------------
    def _array(self, rows_in, rows_out, B, new_weights):
        if not self.act_ub_wr.isdisjoint(rows_in) or not self.act_acc_rd.isdisjoint(rows_out):
            self.sync(U.U_ACT)
        if new_weights:
            self.stats.weight_tiles += 1
            self.stats.array_cycles += max(0, tiling.W_LOAD - self.prev_stream)
        self.stats.array_cycles += B
        self.prev_stream = B
        self.mat_ub_rd.update(rows_in)
        self.mat_acc_wr.update(rows_out)

    def matmul(self, ub_row, B, acc_row, acc, nw):
        self._array(range(ub_row, ub_row + B), range(acc_row, acc_row + B), B, nw)
        self.emit(U.matmul(ub_row, acc_row, B, acc=acc, new_weights=nw))

    def convolve(self, plane_row, plane_rows, acc_row, B, kh, kw, oh0, cnt, acc, nw):
        self._array(range(plane_row, plane_row + plane_rows), range(acc_row, acc_row + B), B, nw)
        self.emit(U.convolve(plane_row, acc_row, kh, kw, oh0, cnt, acc=acc, new_weights=nw))
        self.conv_busy = True

    def activate(self, acc_row, ub_row, rows, q, func, pool=0):
        out = rows // 4 if pool else rows
        rin, rout = range(acc_row, acc_row + rows), range(ub_row, ub_row + out)
        if not self.mat_acc_wr.isdisjoint(rin) or not self.mat_ub_rd.isdisjoint(rout):
            self.sync(U.U_MATRIX)
        self.emit(U.activate(acc_row, ub_row, rows, q, func=func, pool=pool))
        self.act_acc_rd.update(rin)
        self.act_ub_wr.update(rout)
        self.act_q.add(q)
        self.stats.act_rows += rows
        if pool:
            self.pool_busy = True

    def acc_alloc(self, rows):
        if self.acc_cursor + rows > ACC_ROWS:
            self.acc_cursor = 0
        base = self.acc_cursor
        self.acc_cursor += rows
        return base

    # ---- configuration --------------------------------------------------------------------
    @staticmethod
    def _target(i, q):
        return (0x1000 | (q << 4) | i) if i < 16 else (0x2000 | (q << 4) | (i - 16)) if i < 32 else (0x3000 | q)

    def load_qset(self, cfg):
        """cfg: tuple of 33 register values. Returns the parameter set index holding it."""
        self.tick += 1
        if not self.dedupe:
            q = self.rr
            self.rr = (self.rr + 1) % NSETS
            diff = list(range(33))
        else:
            best = None
            for q in range(NSETS):
                diff = [i for i in range(33) if self.qregs[q].get(i) != cfg[i]]
                key = (len(diff) + (3 if q in self.act_q else 0), self.qlru[q])
                if best is None or key < best[0]:
                    best = (key, q, diff)
                if not diff:
                    best = ((0, 0), q, diff)
                    break
            _, q, diff = best
        if diff and q in self.act_q:
            self.sync(U.U_ACT)
        for i in diff:
            self.emit(U.set_config(self._target(i, q), cfg[i]))
            self.qregs[q][i] = cfg[i]
        self.qlru[q] = self.tick
        return q

    def set_geo(self, target, value):
        if self.dedupe and self.geo.get(target) == value:
            return
        if target == U.POOL_W and self.pool_busy:
            self.sync(U.U_ACT)
        if target != U.POOL_W and self.conv_busy:
            self.sync(U.U_MATRIX)
        self.emit(U.set_config(target, value))
        self.geo[target] = value

    # ---- DMA ------------------------------------------------------------------------------
    def rhm(self, ub_row, addr, rows):
        for r in range(0, rows, DMA_MAX_ROWS):
            n = min(DMA_MAX_ROWS, rows - r)
            self.emit(U.read_host_mem(ub_row + r, addr + 16 * r, n))
            self.stats.dma_bytes += 16 * n
        # hardware interlock drains ACTIVATE and the matrix unit before a UB write
        self.act_acc_rd.clear(); self.act_ub_wr.clear(); self.act_q.clear(); self.pool_busy = False
        self.mat_ub_rd.clear(); self.mat_acc_wr.clear(); self.conv_busy = False

    def whm(self, ub_row, addr, rows):
        for r in range(0, rows, DMA_MAX_ROWS):
            n = min(DMA_MAX_ROWS, rows - r)
            self.emit(U.write_host_mem(ub_row + r, addr + 16 * r, n))
            self.stats.dma_bytes += 16 * n
        self.act_acc_rd.clear(); self.act_ub_wr.clear(); self.act_q.clear(); self.pool_busy = False


def requant_cfg(a, nt):
    m0, sh, bias = np.asarray(a["m0"]), np.asarray(a["shift"]), np.asarray(a["bias"])
    n = len(m0)
    regs = []
    for c in range(16):
        ch = 16 * nt + c
        regs.append((int(m0[ch]) & 0xFFFFFFFF) | ((int(sh[ch]) & 0xFF) << 32) if ch < n else 0)
    for c in range(16):
        ch = 16 * nt + c
        regs.append(int(bias[ch]) & 0xFFFFFFFF if ch < n else 0)
    regs.append((a["zp_out"] & 0xFF) | ((a["act_min"] & 0xFF) << 8) | ((a["act_max"] & 0xFF) << 16))
    return tuple(regs)


class WeightStream:
    """Lazy READ_WEIGHTS for one op's tile stream; never more than the FIFO holds."""

    def __init__(self, em, addr):
        self.em, self.addr, self.cpos, self.upto = em, addr, 0, 0

    def need(self, idx, run_end):
        """idx: DDR tile index about to be consumed; run_end: end of its contiguous run."""
        if self.cpos < self.upto and idx == self.cpos:
            self.cpos += 1
            return
        if self.cpos != self.upto:
            raise AssertionError("weight stream out of order")
        n = min(WFIFO_TILES, run_end - idx)
        self.em.emit(U.read_weights(self.addr + 256 * idx, n))
        self.cpos, self.upto = idx + 1, idx + n


# ---------------------------------------------------------------------------------------------
def compile_graph(g, dedupe=True, reuse=True, tag_base=0xC0DE0000, max_kernel=IFIFO_WORDS,
                  reset_config_per_kernel=False):
    """g: a lowered (fused, legalized, scheduled) graph. Returns a Program.

    dedupe: emit a SET_CONFIG only when the register value changes (and reuse a parameter set
    that already holds a tile's config). reuse: liveness-based UB buffer reuse.
    reset_config_per_kernel: assume parameter / geometry registers do not survive a kernel
    relaunch (conservative; by default they persist, as in sim.py and the RTL register file)."""
    dev = device_plan(g)
    steps, pos, step_of = M.segment(g)
    mem = M.plan_memory(g, steps, pos, step_of, reuse=reuse)
    em = Emitter(dedupe)
    stats, ddr_init, wregion = {}, {}, {}
    # static weights -> DDR (one contiguous tile image per op, stream order)
    a = M.DDR_W
    for o in g.ops:
        if o.kind in ("tpu_matmul", "tpu_conv") and g.T(o.inputs[1]).is_const:
            W = g.T(o.inputs[1]).const
            img = tile_bytes(tiles_matmul(W) if o.kind == "tpu_matmul" else tiles_conv(W))
            wregion[o.inputs[1]] = a
            ddr_init[a] = img
            a += len(img)
    wdyn = M.DDR_WDYN
    for o in g.ops:
        if o.kind == "pack_weights":
            K, N = g.T(o.outputs[0]).shape
            wregion[o.outputs[0]] = wdyn
            wdyn += (-(-K // 16)) * (-(-N // 16)) * 256
    # host constants (e.g. the position embedding) -> DDR
    for t, addr in mem.ddr.items():
        if g.T(t).is_const:
            ddr_init[addr] = host_layout(g.T(t).const)
    out_steps = []
    kernels = []

    def new_kernel(step_idx):
        k = Kernel(len(kernels), step_idx)
        kernels.append(k)
        if reset_config_per_kernel:
            em.qregs = [dict() for _ in range(NSETS)]
            em.geo = {}
        out_steps.append(("kernel", k))
        em.words = k.words
        return k

    def close_kernel(k):
        em.sync(U.U_ALL)
        em.emit(U.interrupt_host(tag_base | k.index))
        em.emit(U.halt())
        em.reset_hazards()

    for st in steps:
        if st.kind == "host":
            out_steps.append(("host", list(st.ops)))
            continue
        kst = stats.setdefault(f"<kernel{st.index}>", OpStats(f"<step {st.index} dma/ctl>", "overhead", "", ""))
        em.stats = kst
        k = new_kernel(st.index)
        loads = mem.rhm.get(st.index, [])
        for root in loads:
            em.rhm(mem.ub_base[root], mem.ddr[root], g.T(root).ub_rows)
        if loads:
            em.sync(U.U_DMA)
        whms = mem.whm.get(st.index, [])
        reserve = 4 + sum(-(-g.T(t).ub_rows // DMA_MAX_ROWS) for t in whms)

        def maybe_split(need):
            nonlocal k
            if len(k.words) + need + reserve > max_kernel:
                saved = em.stats
                em.stats = kst
                close_kernel(k)
                k = new_kernel(st.index)
                em.stats = saved

        for name in st.ops:
            o = g.op_by_name(name)
            s = em.stats = stats[name] = _new_stats(g, o)
            s.kernel = k.index
            if o.kind == "tpu_matmul":
                _emit_matmul(g, o, em, mem, wregion, maybe_split)
            else:
                _emit_conv(g, o, em, mem, wregion, maybe_split)
            k = kernels[-1]
            k.ops.append(name)
            k.done[name] = len(k.words) - 1
        em.stats = kst
        for t in whms:
            em.whm(mem.ub_row(g, t), mem.ddr[t], g.T(t).ub_rows)
        close_kernel(k)
    for k in kernels:
        if len(k.words) > max_kernel:
            raise ValueError(f"kernel {k.index} has {len(k.words)} > {max_kernel} instructions")
    return Program(g, out_steps, mem, ddr_init, wregion, stats,
                   dict(dedupe=dedupe, reuse=reuse, max_kernel=max_kernel,
                        reset_config_per_kernel=reset_config_per_kernel))


def host_layout(x):
    """[rows, cols] int -> k-tile-major 16-byte rows (the UB layout), as bytes."""
    x = np.asarray(x, np.int64)
    R, C = x.shape
    CT = -(-C // 16)
    xp = np.zeros((R, CT * 16), np.int64)
    xp[:, :C] = x
    return xp.reshape(R, CT, 16).transpose(1, 0, 2).astype(np.int8).tobytes()


def from_host_layout(b, shape):
    R, C = shape
    CT = -(-C // 16)
    a = np.frombuffer(bytes(b), np.int8).astype(np.int64).reshape(CT, R, 16)
    return a.transpose(1, 0, 2).reshape(R, CT * 16)[:, :C]


def _new_stats(g, o):
    if o.kind == "tpu_matmul":
        x, out = g.T(o.inputs[0]), g.T(o.outputs[0])
        Mr, K = x.shape
        N = out.shape[1]
        p = tiling.plan_matmul(Mr, K, N)
        s = OpStats(o.name, "matmul" + ("(dyn W)" if o.attrs.get("dynamic_weights") else ""),
                    f"{Mr}x{K}x{N}", f"{p.order} KT={p.KT} NT={p.NT} B={p.mblocks[0][1]}x{len(p.mblocks)}")
        s.macs = Mr * K * N
    else:
        H, W = o.attrs["hw"]
        OH, OW = o.attrs["out_hw"]
        w = g.T(o.inputs[1]).const
        s = OpStats(o.name, "conv" + (f"+{o.attrs['pool']}pool" if o.attrs.get("pool") else ""),
                    f"{H}x{W}x{w.shape[2]}->{OH}x{OW}x{w.shape[3]} k{w.shape[0]}s{o.attrs['stride']}", "")
        s.macs = OH * OW * w.shape[0] ** 2 * w.shape[2] * w.shape[3]
    return s


def _emit_matmul(g, o, em, mem, wregion, maybe_split):
    x, out = g.T(o.inputs[0]), g.T(o.outputs[0])
    Mr, K = x.shape
    N = out.shape[1]
    p = tiling.plan_matmul(Mr, K, N)
    xr, orow = mem.ub_row(g, x.name), mem.ub_row(g, out.name)
    ws = WeightStream(em, wregion[o.inputs[1]])
    func = ACT_FUNC.get(o.attrs.get("act"), 0)
    for m0, mb in p.mblocks:
        for nt in range(p.NT):
            maybe_split(p.KT + 40)
            q = em.load_qset(requant_cfg(o.attrs, nt))
            acc = em.acc_alloc(mb)
            for kt in range(p.KT):
                ws.need(nt * p.KT + kt, p.NT * p.KT)
                em.matmul(xr + kt * Mr + m0, mb, acc, 1 if kt else 0, 1)
            em.activate(acc, orow + nt * Mr + m0, mb, q, func)


def _emit_conv(g, o, em, mem, wregion, maybe_split):
    x, out = g.T(o.inputs[0]), g.T(o.outputs[0])
    W = g.T(o.inputs[1]).const
    k, cin, cout = W.shape[0], W.shape[2], W.shape[3]
    H, Wd = o.attrs["hw"]
    OH, OW = o.attrs["out_hw"]
    CT, NT = -(-cin // 16), -(-cout // 16)
    pool = o.attrs.get("pool")
    p = tiling.plan_conv(OH, OW, k, CT, NT, pool)
    em.stats.plan = f"windows={len(p.windows)} taps={k * k} CT={CT} NT={NT}"
    s, pad = o.attrs["stride"], o.attrs["pad"]
    em.set_geo(U.CONV_IN, Wd | (H << 16))
    em.set_geo(U.CONV_OUT, OW | (OH << 16))
    em.set_geo(U.CONV_MISC, s | (s << 4) | (pad << 8) | (pad << 16) | ((o.attrs["pad_val"] & 0xFF) << 24))
    if pool:
        em.set_geo(U.POOL_W, OW)
    xr, orow = mem.ub_row(g, x.name), mem.ub_row(g, out.name)
    ws = WeightStream(em, wregion[o.inputs[1]])
    func = ACT_FUNC.get(o.attrs.get("act"), 0)
    taps = k * k * CT
    PH, PW = OH // 2, OW // 2
    for oh0, cnt in p.windows:
        for nt in range(NT):
            maybe_split(taps + 40)
            q = em.load_qset(requant_cfg(o.attrs, nt))
            acc = em.acc_alloc(cnt * OW)
            t = 0
            for kh in range(k):
                for kw in range(k):
                    for ct in range(CT):
                        ws.need(nt * taps + t, NT * taps)
                        em.convolve(xr + ct * H * Wd, H * Wd, acc, cnt * OW, kh, kw, oh0, cnt, 1 if t else 0, 1)
                        t += 1
            row = orow + (nt * PH * PW + (oh0 // 2) * PW if pool else nt * OH * OW + oh0 * OW)
            em.activate(acc, row, cnt * OW, q, func, {None: 0, "max": 1, "avg": 2}[pool])
