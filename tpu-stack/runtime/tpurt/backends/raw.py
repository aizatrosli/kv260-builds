"""`raw` backend: drive the TPU from user space without the kernel driver, the way the M5 board
gate (tpuv1 software/linux/idma_runtime/m5_board_test.py) does, but with the job sequence of the
kernel driver (tpu_kv260.c: tpu_run_locked / tpu_push_program / tpu_drain_locked):

  registers   UIO "fabric" node at 0xA000_0000 (64 KiB, aligned 32-bit accesses only), or
              pynq.MMIO when `pynq` is importable (PYNQ-style path)
  DMA memory  u-dma-buf (/dev/udmabufN, CACHED mapping + explicit sync_for_device/sync_for_cpu via
              sysfs), or pynq.allocate() buffers (.flush() / .invalidate())
  relocation  in user space (base.relocate: same checks as the kernel)

Buffers are carved out of the one physically contiguous region by a first-fit allocator.  Use this
path when the PL image is loaded but /dev/tpu0 is not (e.g. bring-up), or to compare overheads.
Completion is polled (CORE_STATUS); there is no IRQ wait.
"""
import glob
import mmap
import os
import time
from pathlib import Path

import numpy as np

from .base import (Backend, BackendUnavailable, DevBuffer, RunResult, relocate, RUN_OK, RUN_CFG_ERR,
                   RUN_DMA_FAULT, RUN_TIMEOUT, RUN_PUSH_FAIL)
from ..buffers import FirstFit

# tpu_regs v0.3/0.4 (kv260_top.sv) -- identical to the kernel driver's defines
R_VERSION, R_SCRATCH, R_BUILD_ID, R_STATUS, R_CTRL, R_LEVEL, R_CSTATUS, R_IRQ_TAG = 0x00, 0x04, 0x08, 0x10, 0x14, 0x18, 0x1C, 0x20
R_PERF0 = 0x24
CTRL_ERRCLR, CTRL_UB_OWN, CTRL_ACC_OWN, CTRL_WF_FLUSH, CTRL_IF_FLUSH = 1, 2, 4, 0x20, 0x40
CTRL_RUN, CTRL_IRQ_CLR, CTRL_PERF_CLR = 0x80, 0x100, 0x200
CS_BUSY, CS_HALTED, CS_CFG_ERR, CS_IRQ, CS_DMA_ERR = 1, 2, 4, 8, 0x10
ST_ERR_MASK = 0x1FE
IDMA = 0x1000
D_CONF, D_NEXT_ID, D_DONE_ID, D_DST, D_SRC, D_LEN = IDMA + 0x000, IDMA + 0x044, IDMA + 0x084, IDMA + 0x0D0, IDMA + 0x0D8, IDMA + 0x0E0
D_DSTRIDE, D_SSTRIDE, D_REPS = IDMA + 0x100, IDMA + 0x108, IDMA + 0x110
IFIFO_PUSH_ADDR = 0xA040_0000
FABRIC_BASE, FABRIC_SIZE = 0xA000_0000, 0x10000
STAGE_BYTES = 2048 * 16
CLEAR_BITS = CTRL_ERRCLR | CTRL_WF_FLUSH | CTRL_IF_FLUSH | CTRL_IRQ_CLR | CTRL_PERF_CLR


def _sysfs(p):
    return Path(p).read_text().strip()


# ---------------------------------------------------------------------------------------------- regs
class UioRegs:
    def __init__(self):
        nodes = [n for n in glob.glob("/sys/class/uio/uio*")
                 if _sysfs(n + "/name") == "fabric" and int(_sysfs(n + "/maps/map0/addr"), 0) == FABRIC_BASE]
        if len(nodes) != 1:
            raise BackendUnavailable("raw: no unique UIO 'fabric' node at 0xA000_0000 (is the TPU bound to "
                                     "tpu_kv260 instead? then use --backend board)")
        self.path = "/dev/" + os.path.basename(nodes[0])
        fd = os.open(self.path, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(fd, FABRIC_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.w = memoryview(self._mm).cast("I")       # aligned 32-bit accesses only (L1)

    def rd(self, off):
        return self.w[off >> 2]

    def wr(self, off, v):
        self.w[off >> 2] = v & 0xFFFFFFFF


class PynqRegs:
    def __init__(self):
        import pynq
        self.m = pynq.MMIO(FABRIC_BASE, FABRIC_SIZE)

    def rd(self, off):
        return self.m.read(off)

    def wr(self, off, v):
        self.m.write(off, v & 0xFFFFFFFF)


# ---------------------------------------------------------------------------------------------- memory
class UdmabufRegion:
    """One u-dma-buf: cached mapping; sync_* via sysfs (HP0 is not coherent)."""

    def __init__(self, name="udmabuf0"):
        self.sys = f"/sys/class/u-dma-buf/{name}/"
        if not os.path.exists(self.sys):
            raise BackendUnavailable(f"raw: /dev/{name} not found (load u-dma-buf: modprobe u-dma-buf udmabuf0=16777216)")
        self.phys = int(_sysfs(self.sys + "phys_addr"), 0)
        self.size = int(_sysfs(self.sys + "size"))
        fd = os.open(f"/dev/{name}", os.O_RDWR)
        self._mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.mem = np.frombuffer(self._mm, np.uint8)
        self.coherent = _sysfs(self.sys + "dma_coherent") == "1" if os.path.exists(self.sys + "dma_coherent") else False

    def _sync(self, what, off, n):
        for k, v in (("sync_offset", off), ("sync_size", n), ("sync_direction", 0)):
            Path(self.sys + k).write_text(str(v))
        Path(self.sys + what).write_text("1")

    def flush(self, off, n):
        if not self.coherent:
            self._sync("sync_for_device", off, n)

    def invalidate(self, off, n):
        if not self.coherent:
            self._sync("sync_for_cpu", off, n)


class PynqRegion:
    def __init__(self, size=32 << 20):
        import pynq
        self.buf = pynq.allocate(shape=(size,), dtype=np.uint8)
        self.phys, self.size, self.mem = int(self.buf.physical_address), size, np.asarray(self.buf)
        self.coherent = False

    def flush(self, off, n):
        self.buf.flush()

    def invalidate(self, off, n):
        self.buf.invalidate()


# ---------------------------------------------------------------------------------------------- backend
class RawBackend(Backend):
    name = "raw"

    def __init__(self, regs=None, region=None, flavor="auto", poll_sleep=0.0):
        if regs is None or region is None:
            pynq_ok = False
            if flavor in ("auto", "pynq"):
                try:
                    import pynq  # noqa: F401
                    pynq_ok = True
                except ImportError:
                    if flavor == "pynq":
                        raise BackendUnavailable("raw/pynq: `pynq` is not importable") from None
            if not Path("/sys/class/fpga_manager/fpga0").exists():
                raise BackendUnavailable("raw: not on a Zynq MPSoC (no fpga_manager): use --backend sim/mock")
            if _sysfs("/sys/class/fpga_manager/fpga0/state") != "operating":
                raise BackendUnavailable("raw: PL not loaded (xmutil loadapp tpu_kv260); never touch an unloaded PL (L5)")
            regs = regs or (PynqRegs() if pynq_ok else UioRegs())
            region = region or (PynqRegion() if pynq_ok else UdmabufRegion())
            self.name = "raw-pynq" if pynq_ok else "raw-uio"
        self.regs, self.region, self.poll_sleep = regs, region, poll_sleep
        self.coherent = getattr(region, "coherent", False)
        self.ctrl = 0
        hv = regs.rd(R_VERSION)
        if (hv >> 8) not in (4, 5, 6):
            raise BackendUnavailable(f"raw: unsupported/missing TPU (VERSION=0x{hv:08x})")
        self.hw_version, self.build_id = hv, regs.rd(R_BUILD_ID)
        self.has_vpu = (hv >> 8) >= 6
        self.alloc_ = FirstFit(region.size, align=4096)
        self.stage_off = self.alloc_.alloc(STAGE_BYTES)
        self.max_buf_bytes = region.size - STAGE_BYTES
        self._bufs, self._next = {}, 1
        self.jobs_ok = self.jobs_failed = 0
        self._ctrl_set(0)
        if regs.rd(R_CSTATUS) & CS_BUSY:
            self._drain()
        self._pulse(CLEAR_BITS)

    # -- memory ---------------------------------------------------------------------------------
    def alloc(self, size):
        off = self.alloc_.alloc(size)
        if off is None:
            raise OSError(28, "raw: DMA region exhausted")
        asize = (size + 4095) & ~4095
        mem = self.region.mem[off:off + asize]
        mem[:] = 0
        b = DevBuffer(self, self._next, size, mem, phys=self.region.phys + off)
        b.alloc_size, b.off = asize, off
        self._bufs[b.handle] = b
        self._next += 1
        return b

    def free(self, buf):
        if not buf.freed:
            buf.freed = True
            self._bufs.pop(buf.handle, None)
            self.alloc_.free(buf.off)

    def sync_for_device(self, buf, offset=0, length=None):
        self.region.flush(buf.off + offset, buf.alloc_size - offset if length is None else length)

    def sync_for_cpu(self, buf, offset=0, length=None):
        self.region.invalidate(buf.off + offset, buf.alloc_size - offset if length is None else length)

    # -- registers --------------------------------------------------------------------------------
    def _ctrl_set(self, v):
        self.ctrl = v
        self.regs.wr(R_CTRL, v)

    def _pulse(self, bits):
        self.regs.wr(R_CTRL, self.ctrl | bits)
        self.regs.rd(R_VERSION)
        self.regs.wr(R_CTRL, self.ctrl)
        self.regs.rd(R_VERSION)

    def _wait_idle(self, timeout_s):
        end = time.monotonic() + timeout_s
        while self.regs.rd(R_CSTATUS) & CS_BUSY:
            if time.monotonic() > end:
                return False
        return True

    def _drain(self):
        self._ctrl_set(CTRL_UB_OWN | CTRL_ACC_OWN)
        if not self._wait_idle(0.5):
            raise OSError(5, f"raw: core did not drain (CORE_STATUS={self.regs.rd(R_CSTATUS):08x}); reload the PL")
        self._pulse(CLEAR_BITS)
        self._ctrl_set(0)
        if (self.regs.rd(R_STATUS) & ST_ERR_MASK) or (self.regs.rd(R_CSTATUS) & (CS_BUSY | CS_CFG_ERR | CS_DMA_ERR)):
            raise OSError(5, "raw: errors persist after recovery")

    def _wr64(self, off, v):
        self.regs.wr(off, v & 0xFFFFFFFF)
        self.regs.wr(off + 4, v >> 32)

    def _push(self, n):
        r = self.regs
        self._wr64(D_SRC, self.region.phys + self.stage_off)
        self._wr64(D_DST, IFIFO_PUSH_ADDR)
        self._wr64(D_LEN, n * 16)
        self._wr64(D_SSTRIDE, 0)
        self._wr64(D_DSTRIDE, 0)
        self._wr64(D_REPS, 1)
        r.wr(D_CONF, 0)
        tid = r.rd(D_NEXT_ID)                       # side effect: launches (read exactly once)
        if not tid:
            return False
        end = time.monotonic() + 0.1
        while ((r.rd(D_DONE_ID) - tid) & 0xFFFFFFFF) >= 0x80000000:
            if time.monotonic() > end:
                return False
        return not (r.rd(R_STATUS) & 0x1C) and (r.rd(R_LEVEL) >> 16) == n

    # -- run --------------------------------------------------------------------------------------
    def run_job(self, job, timeout_ms=0):
        r = self.regs
        words, used = relocate(job.program, job.relocs, lambda b: b.phys,
                               lambda b: b.alloc_size if self._bufs.get(b.handle) is b else None)
        n = len(words)
        stage = self.region.mem[self.stage_off:self.stage_off + n * 16]
        stage[:] = np.frombuffer(b"".join(w.to_bytes(16, "little") for w in words), np.uint8)
        self.region.flush(self.stage_off, n * 16)
        for b in used:                                  # host writes -> device
            self.sync_for_device(b)
        if r.rd(R_CSTATUS) & CS_BUSY:
            self._drain()
        self._ctrl_set(0)
        self._pulse(CLEAR_BITS)
        if not self._push(n):
            cs, hw = r.rd(R_CSTATUS), r.rd(R_STATUS)
            self.jobs_failed += 1
            self._drain()
            return RunResult(RUN_PUSH_FAIL, 0, cs, hw, 0, {}, 0)
        self._ctrl_set(CTRL_UB_OWN | CTRL_ACC_OWN | CTRL_PERF_CLR)
        t0 = time.perf_counter_ns()
        self._ctrl_set(CTRL_UB_OWN | CTRL_ACC_OWN | CTRL_RUN)
        deadline = time.monotonic() + (timeout_ms or 2000) / 1000
        timeout = False
        while True:
            cs = r.rd(R_CSTATUS)
            if cs & (CS_CFG_ERR | CS_DMA_ERR | CS_IRQ):  # as m5_board_test: error or INTERRUPT_HOST
                break
            if time.monotonic() > deadline:
                timeout = True
                break
            if self.poll_sleep:
                time.sleep(self.poll_sleep)
        t1 = time.perf_counter_ns()
        perf = {k: r.rd(R_PERF0 + 4 * i) for i, k in enumerate(("total", "array_active", "w_stall", "w_shift",
                                                                 "non_matrix", "sync", "act", "retired"))}
        tag = r.rd(R_IRQ_TAG)
        end = time.monotonic() + 0.02
        while (cs := r.rd(R_CSTATUS)) & CS_BUSY and not cs & (CS_HALTED | CS_CFG_ERR | CS_DMA_ERR):
            if time.monotonic() > end:
                break
        perf["retired"] = r.rd(R_PERF0 + 28)
        hw = r.rd(R_STATUS)
        if timeout:
            status = RUN_TIMEOUT
        elif (cs & CS_DMA_ERR) or (hw & ST_ERR_MASK):
            status = RUN_DMA_FAULT
        elif cs & CS_CFG_ERR:
            status = RUN_CFG_ERR
        elif not (cs & CS_HALTED) and (cs & CS_BUSY):
            status = RUN_TIMEOUT
        else:
            status = RUN_OK
        res = RunResult(status, (cs >> 8) & 0xFF if cs & CS_CFG_ERR else 0, cs, hw, tag, perf, t1 - t0)
        self._drain()
        for b in used:                                  # device writes -> host
            self.sync_for_cpu(b)
        if status == RUN_OK:
            self.jobs_ok += 1
        else:
            self.jobs_failed += 1
        return res

    def info(self):
        return dict(backend=self.name, hw_version=self.hw_version, build_id=self.build_id, vpu=self.has_vpu,
                    region_phys=hex(self.region.phys), region_bytes=self.region.size, coherent=self.coherent,
                    jobs_ok=self.jobs_ok, jobs_failed=self.jobs_failed, timing="measured (user-space poll)")


# ---------------------------------------------------------------------------------------------- fake
class FakeRegion:
    """numpy 'physical' memory for FakeFabric tests; counts cache operations."""

    def __init__(self, size=16 << 20, phys=0x6000_0000):
        self.phys, self.size, self.mem = phys, size, np.zeros(size, np.uint8)
        self.coherent = False
        self.flushes = self.invalidates = 0

    def flush(self, off, n):
        self.flushes += 1

    def invalidate(self, off, n):
        self.invalidates += 1


class FakeFabric:
    """Register-level model of tpu_regs + iDMA frontend + core, enough to test the raw protocol
    off-board: iDMA push into the instruction FIFO, CTRL ownership/RUN/flush pulses, CORE_STATUS,
    IRQ_TAG, perf counters.  The core is tpurt.fastsim.FastCoreSim on FakeRegion memory."""

    def __init__(self, region, version=0x400, build_id=0x6AF93498, vpu=False):
        self.region, self.vpu = region, vpu
        self.r = {R_VERSION: version, R_BUILD_ID: build_id, R_STATUS: 0, R_CTRL: 0, R_LEVEL: 0, R_CSTATUS: 0,
                  R_IRQ_TAG: 0}
        self.idma = {}
        self.next_id = self.done_id = 0
        self.ififo = []
        self.perf = [0] * 8

    def _mem(self):
        region = self.region

        class M:
            def read(self, a, n):
                o = a - region.phys
                if not 0 <= o <= region.size - n:
                    raise RuntimeError("fake DMA outside the region")
                return region.mem[o:o + n].copy()

            def write(self, a, d):
                o = a - region.phys
                if not 0 <= o <= region.size - len(d):
                    raise RuntimeError("fake DMA outside the region")
                region.mem[o:o + len(d)] = np.frombuffer(d, np.uint8)
        return M()

    def rd(self, off):
        if off == D_NEXT_ID:
            self.next_id += 1
            g = lambda o: self.idma.get(o, 0) | (self.idma.get(o + 4, 0) << 32)
            src, dst, n = g(D_SRC), g(D_DST), g(D_LEN)
            if dst == IFIFO_PUSH_ADDR:
                data = self._mem().read(src, n)
                self.ififo += [int.from_bytes(data[i:i + 16].tobytes(), "little") for i in range(0, n, 16)]
                self.r[R_LEVEL] = len(self.ififo) << 16
            self.done_id = self.next_id
            return self.next_id
        if off == D_DONE_ID:
            return self.done_id
        if R_PERF0 <= off < R_PERF0 + 32:
            return self.perf[(off - R_PERF0) // 4]
        return self.r.get(off, 0)

    def wr(self, off, v):
        if off >= IDMA:
            self.idma[off] = v
            return
        if off == R_SCRATCH:
            self.r[off] = v
        if off != R_CTRL:
            return
        self.r[R_CTRL] = v
        if v & CTRL_IF_FLUSH:
            self.ififo = []
            self.r[R_LEVEL] = 0
        if v & (CTRL_ERRCLR | CTRL_IRQ_CLR):
            self.r[R_CSTATUS] &= ~(CS_IRQ | CS_CFG_ERR | CS_DMA_ERR | 0xFF00)
            self.r[R_STATUS] = 0
        if v & CTRL_RUN and v & CTRL_UB_OWN and self.ififo:
            from .. import fastsim
            m = fastsim.FastCoreSim(enable_dma=True, mem=self._mem(), vpu_enabled=self.vpu).run(self.ififo)
            self.ififo = []
            self.r[R_LEVEL] = 0
            cs = 0
            if m.cfg_err:
                cs |= CS_CFG_ERR | ((m.cfg_err_op & 0xFF) << 8)
            if m.halted:
                cs |= CS_HALTED
            if m.irq_count:
                cs |= CS_IRQ
            self.r[R_CSTATUS] = cs
            self.r[R_IRQ_TAG] = m.irq_tag or 0
            c = m.cycles
            self.perf = [sum(c.values()), c["matmul"], 0, c["w_load"], 0, 0, c["activate"], m.instr_count]
        if not v & CTRL_RUN:
            self.r[R_CSTATUS] &= ~CS_BUSY
