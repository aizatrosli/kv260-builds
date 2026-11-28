"""Execute a compiled Program on tpuv1's ISA simulator, with the host steps in Python.

The TPU side is tpuv1's own ``sim.CoreSim`` (DMA enabled, hazard checking on): every kernel is
stepped instruction by instruction. Between kernels the host reads its inputs from the
simulated DDR, runs the integer op (jepa-tpu intops), and writes results back to DDR (weight
packing writes Weight-FIFO tile images). ``FastCoreSim`` is the same simulator with the 16x16
MAC loop and the requant loop vectorised in numpy (identical results, checked by the tests).
"""
import numpy as np

from .env import isasim, asm, N
from . import refexec
from .codegen import host_layout, from_host_layout, tiles_matmul, tile_bytes
from .passes import device_plan


class DDR:
    """Sparse byte-addressable memory with the dict interface sim.CoreSim uses."""
    PAGE = 1 << 16

    def __init__(self):
        self.pages = {}

    def _page(self, a):
        p = self.pages.get(a >> 16)
        if p is None:
            p = self.pages[a >> 16] = bytearray(self.PAGE)
        return p

    def __getitem__(self, a):
        return self._page(a)[a & 0xFFFF]

    def update(self, items):
        for a, v in items:
            self._page(a)[a & 0xFFFF] = v

    def write(self, addr, data):
        data = memoryview(bytes(data))
        while data:
            off = addr & 0xFFFF
            n = min(len(data), self.PAGE - off)
            self._page(addr)[off:off + n] = data[:n]
            addr += n
            data = data[n:]

    def read(self, addr, n):
        out = bytearray()
        while n:
            off = addr & 0xFFFF
            m = min(n, self.PAGE - off)
            out += self._page(addr)[off:off + m]
            addr += m
            n -= m
        return bytes(out)


class FastCoreSim(isasim.CoreSim):
    """tpuv1 CoreSim with numpy MAC / requant (same state, hazards and error rules)."""

    def _mac(self, xs, acc_row, acc, what):
        Y = np.asarray(xs, np.int64) @ np.asarray(self.W, np.int64)
        for b in range(len(xs)):
            r = acc_row + b
            y = Y[b]
            if acc:
                old = self.acc.get(r)
                if old is None:
                    raise RuntimeError(f"{what} accumulates into uninitialised ACC row {r}")
                y = np.asarray(old, np.int64) + y
            self.acc[r] = refexec.wrap32(y).tolist()

    def _activate(self, acc_row, ub_row, rows, q, pool=0):
        out_rows = rows // 4 if pool else rows
        rows_in = {acc_row + r for r in range(rows)}
        rows_out = {ub_row + r for r in range(out_rows)}
        self._hz(rows_in & self.mat_acc_wr, "ACTIVATE reads ACC rows an unsynced MATMUL writes (RAW)")
        self._hz(rows_out & self.mat_ub_rd, "ACTIVATE writes UB rows an unsynced MATMUL reads (WAR)")
        p = self.params[q]
        A = []
        for r in range(rows):
            a = self.acc.get(acc_row + r)
            if a is None:
                raise RuntimeError(f"ACTIVATE reads uninitialised ACC row {acc_row + r}")
            A.append(a)
        vals = refexec.requant(np.asarray(A, np.int64), p["bias"], p["m0"], p["shift"], p["zp_out"],
                               p["act_min"], p["act_max"]).tolist()
        if pool:
            vals = isasim.cr.pool_raster(vals, self.pool_w, "max" if pool == 1 else "avg")
            self.pool_busy = True
        for r, row in enumerate(vals):
            self.ub[ub_row + r] = row
        self.act_acc_rd |= rows_in
        self.act_ub_wr |= rows_out
        self.act_qsets.add(q)


class RunResult:
    def __init__(self):
        self.outputs = {}       # graph output -> array (read from DDR)
        self.tpu = {}           # TPU op output (UB snapshot right after the op finished)
        self.host = {}          # host op outputs
        self.instrs = 0
        self.kernels = 0
        self.dma = 0
        self.sim = None


def _ub_read(sim, row, shape):
    R, C = shape
    CT = -(-C // N)
    a = np.array([sim.ub[row + i] for i in range(CT * R)], np.int64).reshape(CT, R, N)
    return a.transpose(1, 0, 2).reshape(R, CT * N)[:, :C]


def run_program(prog, feeds, fast=True, check_hazards=True, snapshot=True):
    g, mem = prog.graph, prog.mem
    dev = device_plan(g)
    ddr = DDR()
    for addr, b in prog.ddr_init.items():
        ddr.write(addr, b)
    for t in g.inputs:
        ddr.write(mem.ddr[t], host_layout(feeds[t]))
    sim = (FastCoreSim if fast else isasim.CoreSim)(check_hazards=check_hazards, enable_dma=True)
    sim.ddr = ddr
    res = RunResult()
    res.sim = sim

    def read_host(t):
        T = g.T(t)
        if T.is_const:
            return T.const
        if t not in mem.ddr:
            raise RuntimeError(f"host needs {t} but it has no DDR copy")
        return from_host_layout(ddr.read(mem.ddr[t], T.ub_rows * 16), T.shape)

    for kind, item in prog.steps:
        if kind == "host":
            for name in item:
                op = g.op_by_name(name)
                vals = {i: read_host(i) for i in op.inputs}
                outs = refexec.eval_op(op, vals, g)
                for t, v in zip(op.outputs, outs):
                    res.host[t] = v
                    if g.T(t).attrs.get("kind") == "weight_stream":
                        ddr.write(prog.wregion[t], tile_bytes(tiles_matmul(v)))
                    else:
                        ddr.write(mem.ddr[t], host_layout(v))
            continue
        done_at = {i: n for n, i in item.done.items()}
        sim.halted = False
        irq0 = sim.irq_count
        for i, w in enumerate(item.words):
            sim.step(w)
            if sim.cfg_err:
                raise RuntimeError(f"kernel {item.index} instr {i} {asm.disasm(w)}: cfg_err {sim.cfg_err_why}")
            if snapshot and i in done_at:
                op = g.op_by_name(done_at[i])
                t = op.outputs[0]
                res.tpu[op.name] = _ub_read(sim, mem.ub_row(g, t), g.T(t).shape)
        if not sim.halted or sim.irq_count != irq0 + 1:
            raise RuntimeError(f"kernel {item.index} did not finish with IRQ+HALT")
        res.instrs += len(item.words)
        res.kernels += 1
    res.dma = sim.dma_count
    for t in g.outputs:
        res.outputs[t] = read_host(t) if t in mem.ddr else res.tpu[g.producer(t).name]
    return res
