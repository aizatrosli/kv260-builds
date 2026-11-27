"""`sim` backend: in-process device on tpuv1's ISA simulator (numpy-accelerated subclass).

Buffers are numpy arrays at fake DDR-high addresses; run() applies the same validation/relocation
rules as the kernel driver (base.relocate) and executes on tpurt.fastsim.FastCoreSim (hazard
checking on).  Timing (wall_ns, perf) is an estimate from a simple cycle model at 100 MHz.
golden=True runs the unmodified tpuv1 CoreSim instead (slow; for cross-checks on small programs).
"""
import errno

import numpy as np

from .base import Backend, DevBuffer, RunResult, relocate, perf_from_cycles, RUN_OK, RUN_CFG_ERR, RUN_TIMEOUT
from .. import fastsim

FAKE_BASE = 0x8_1000_0000
SLOT = 0x400_0000                   # 64 MiB per buffer slot
CLOCK_HZ = 100e6


class _Mem:
    def __init__(self, be):
        self.be = be

    def _loc(self, addr, n):
        slot, off = divmod(addr - FAKE_BASE, SLOT)
        buf = self.be._slots.get(slot)
        if addr < FAKE_BASE or buf is None or off + n > buf.size:
            raise RuntimeError(f"simulated DMA outside any buffer: {addr:#x} +{n}")
        return buf.mem, off

    def read(self, addr, n):
        a, off = self._loc(addr, n)
        return a[off:off + n].copy()

    def write(self, addr, data):
        a, off = self._loc(addr, len(data))
        a[off:off + len(data)] = np.frombuffer(data, np.uint8)


class SimBackend(Backend):
    name = "sim"
    estimated_timing = True

    def __init__(self, vpu=True, check_hazards=True, golden=False, max_buf_bytes=64 << 20):
        self.has_vpu = vpu and not golden
        self.check_hazards, self.golden = check_hazards, golden
        self.max_buf_bytes = max_buf_bytes
        self._slots, self._next = {}, 1
        self.jobs_ok = self.jobs_failed = 0
        self.last_sim = None

    def alloc(self, size):
        if size <= 0 or size > self.max_buf_bytes:
            raise OSError(errno.EINVAL if size <= 0 else errno.ENOSPC, "bad buffer size")
        asize = (size + 4095) & ~4095
        slot = next(i for i in range(len(self._slots) + 1) if i not in self._slots)
        buf = DevBuffer(self, self._next, size, np.zeros(asize, np.uint8), phys=FAKE_BASE + slot * SLOT)
        buf.alloc_size, buf.slot = asize, slot
        self._next += 1
        self._slots[slot] = buf
        return buf

    def free(self, buf):
        if not buf.freed:
            buf.freed = True
            self._slots.pop(buf.slot, None)

    def run_job(self, job, timeout_ms=0):
        words, _ = relocate(job.program, job.relocs, lambda b: b.phys,
                            lambda b: b.alloc_size if self._slots.get(getattr(b, "slot", -1)) is b else None)
        if self.golden:
            ddr = {}
            for b in self._slots.values():
                ddr.update({b.phys + i: int(v) for i, v in enumerate(b.mem)})
            m = fastsim.isasim.CoreSim(enable_dma=True, ddr=ddr, check_hazards=self.check_hazards).run(words)
            for b in self._slots.values():
                b.mem[:] = np.frombuffer(bytes(m.ddr.get(b.phys + i, 0) for i in range(b.alloc_size)), np.uint8)
            cycles = {"other": m.instr_count}
        else:
            m = fastsim.FastCoreSim(enable_dma=True, mem=_Mem(self), check_hazards=self.check_hazards,
                                    vpu_enabled=self.has_vpu).run(words)
            cycles = m.cycles
        self.last_sim = m
        status = RUN_CFG_ERR if m.cfg_err else (RUN_OK if m.halted and m.irq_count >= 1 else RUN_TIMEOUT)
        if status == RUN_OK:
            self.jobs_ok += 1
        else:
            self.jobs_failed += 1
        perf = perf_from_cycles(cycles, m.instr_count)
        return RunResult(status=status, cfg_err_op=(m.cfg_err_op or 0) if m.cfg_err else 0,
                         core_status=0x1A if status == RUN_OK else 0x4, hw_status=0, irq_tag=m.irq_tag or 0,
                         perf=perf, wall_ns=int(perf["total"] / CLOCK_HZ * 1e9))

    def info(self):
        return dict(backend=self.name, hw_version=0x400, build_id=0, vpu=self.has_vpu, golden=self.golden,
                    buffers=len(self._slots), jobs_ok=self.jobs_ok, jobs_failed=self.jobs_failed,
                    timing="estimated (cycle model, 100 MHz)")

    def close(self):
        for b in list(self._slots.values()):
            self.free(b)


class NullBackend(SimBackend):
    """Jobs 'complete' instantly without computing (outputs are whatever the buffers hold): measures
    the pure host-side cost of an executor (orchestration, host ops, copies, relocation checks)."""
    name = "null"
    estimated_timing = False

    def __init__(self, **kw):
        super().__init__(vpu=True, **kw)

    def run_job(self, job, timeout_ms=0):
        self.jobs_ok += 1
        return RunResult(status=RUN_OK, cfg_err_op=0, core_status=0x1A, hw_status=0, irq_tag=0,
                         perf={k: 0 for k in ("total", "array_active", "w_stall", "w_shift", "non_matrix", "sync",
                                              "act", "retired")}, wall_ns=0)
