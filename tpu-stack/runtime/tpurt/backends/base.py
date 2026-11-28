"""Backend interface shared by sim / mock / board (driver) / raw (UIO, PYNQ).

A backend allocates DMA-able buffers (`DevBuffer`: handle, writable `mem`, size) and runs one TPU
program with a relocation list [(instr_index, DevBuffer, byte_offset)], exactly the /dev/tpu0 ABI
(software/driver/include/uapi/tpu_kv260.h).  Results are tpuv1's tpu.RunResult.

Cache hooks: sync_for_device / sync_for_cpu are no-ops on coherent memory (the driver's buffers are
dma_alloc_coherent) and real cache maintenance on the raw u-dma-buf / PYNQ paths.
"""
import errno
import time

from .. import _paths, isa

_paths.need_tpuv1_runtime()
import tpu as tpudrv            # noqa: E402  (tpuv1 software/runtime/tpu.py: ABI constants, RunResult)

RunResult = tpudrv.RunResult
RUN_OK, RUN_CFG_ERR, RUN_DMA_FAULT, RUN_TIMEOUT, RUN_PUSH_FAIL = range(5)


class BackendUnavailable(RuntimeError):
    """The backend cannot run here (no device, missing module...). Message says what to do."""


class TpuRunError(RuntimeError):
    def __init__(self, result, what=""):
        super().__init__(f"TPU job {what} failed: {result}")
        self.result = result


class DevBuffer:
    def __init__(self, backend, handle, size, mem, phys=None, raw=None):
        self.backend, self.handle, self.size, self.mem, self.phys, self.raw = backend, handle, size, mem, phys, raw
        self.freed = False

    def __repr__(self):
        return f"DevBuffer(h={self.handle}, size={self.size}" + (f", phys={self.phys:#x})" if self.phys else ")")


class Job:
    """A program + relocations, packed once and run many times."""

    def __init__(self, program, relocs, label=""):
        self.program, self.relocs, self.label = bytes(program), list(relocs), label
        self.n_instr = len(self.program) // 16


class Backend:
    name = "base"
    coherent = True
    has_vpu = False
    lut_rom = None        # {idx: words}: VPU LUT-ROM contents for simulated backends (vpu.jepa_lut_rom)
    max_buf_bytes = 64 << 20
    estimated_timing = False        # True: wall_ns is a model estimate, not a measurement
    preferred_arena = 4 << 20       # DmaPool arena size that suits this backend

    def alloc(self, size):
        raise NotImplementedError

    def free(self, buf):
        raise NotImplementedError

    def run(self, program, relocs, timeout_ms=0):
        return self.run_job(self.prepare(program, relocs), timeout_ms)

    def prepare(self, program, relocs, label=""):
        return Job(program, relocs, label)

    def run_job(self, job, timeout_ms=0):
        raise NotImplementedError

    def sync_for_device(self, buf, offset=0, length=None):
        pass

    def sync_for_cpu(self, buf, offset=0, length=None):
        pass

    def info(self):
        return {"backend": self.name}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def relocate(program, relocs, addr_of, size_of):
    """User-space version of the kernel's tpu_validate(): bounds-check every relocation, patch the
    DDR address fields, require every DMA instruction to be relocated exactly once and the weight
    FIFO never to underflow.  Raises OSError(EINVAL/ENOENT) like the ioctl.  Returns (words, used)."""
    n = len(program) // 16
    if len(program) % 16 or not 1 <= n <= isa.MAX_INSTR:
        raise OSError(errno.EINVAL, "bad program length")
    words = [int.from_bytes(program[16 * i:16 * i + 16], "little") for i in range(n)]
    seen, used = set(), []
    for idx, buf, off in relocs:
        if not 0 <= idx < n or idx in seen:
            raise OSError(errno.EINVAL, "relocation index out of range or duplicated")
        seen.add(idx)
        w = words[idx]
        if w & 0xFF not in isa.DMA_OPS:
            raise OSError(errno.EINVAL, "relocation on a non-DMA instruction")
        size = size_of(buf)
        if size is None:
            raise OSError(errno.ENOENT, "unknown buffer")
        span = isa.dma_span(w)
        if off % 16 or off > size or span > size - off:
            raise OSError(errno.EINVAL, f"relocation [{off}, +{span}) exceeds the {size}-byte buffer")
        words[idx] = isa.patch_addr(w, addr_of(buf) + off)
        if buf not in used:
            used.append(buf)
    pushed = consumed = 0
    for i, w in enumerate(words):
        op = w & 0xFF
        if op in isa.DMA_OPS and i not in seen:
            raise OSError(errno.EINVAL, f"DMA instruction {i} has no relocation")
        if op == isa.OP_RW:
            pushed += (w >> 16) & 0xFFFF
        if op in (0x10, 0x11) and w & (1 << 9):
            consumed += 1
            if consumed > pushed:
                raise OSError(errno.EINVAL, "weight FIFO underflow (MATMUL before its READ_WEIGHTS)")
    return words, used


def perf_from_cycles(cycles, n_instr):
    total = sum(cycles.values())
    return dict(total=total, array_active=cycles.get("matmul", 0), w_stall=0, w_shift=cycles.get("w_load", 0),
                non_matrix=total - cycles.get("matmul", 0), sync=0, act=cycles.get("activate", 0), retired=n_instr)


def now_ns():
    return time.perf_counter_ns()
