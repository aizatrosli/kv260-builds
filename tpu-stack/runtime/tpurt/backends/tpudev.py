"""Backends on tpuv1's Python driver binding (software/runtime/tpu.py, class Device):

  board  /dev/tpu0 through the kernel driver (ioctl ALLOC/mmap/RUN/RECOVER).  Buffers are
         dma_alloc_coherent (non-cacheable): no cache maintenance, but read them once with a
         sequential copy rather than element by element.
  mock   tpuv1's tpu_mock.MockKernel (an executable model of the driver ABI: validation, handles,
         limits, relocation) behind the very same tpu.Device code path; its simulator is swapped
         for tpurt's numpy-accelerated subclass unless fast=False.

prepare() packs program + relocations once into stable buffers; run_job() is then a single ioctl.
"""
import ctypes
import os
import struct
import types
from pathlib import Path

from .base import Backend, BackendUnavailable, DevBuffer, Job
from .. import _paths

_paths.need_tpuv1_runtime()
import tpu as tpudrv            # noqa: E402


class _PreparedJob(Job):
    def __init__(self, program, relocs, label=""):
        super().__init__(program, relocs, label)
        self._prog = bytearray(self.program)
        self._rel = bytearray(b"".join(struct.pack(tpudrv.RELOC_FMT, i, b.handle, off) for i, b, off in relocs))
        self._pptr = ctypes.addressof((ctypes.c_char * len(self._prog)).from_buffer(self._prog))
        self._rptr = (ctypes.addressof((ctypes.c_char * len(self._rel)).from_buffer(self._rel)) if self._rel else 0)


class TpuDeviceBackend(Backend):
    def __init__(self, dev, name):
        self.dev, self.name = dev, name
        try:
            info = dev.info()
            self.max_buf_bytes = info.max_buf_bytes
            self._info = info
        except OSError as e:
            raise BackendUnavailable(f"{name}: GET_INFO failed: {e}") from None

    def alloc(self, size):
        b = self.dev.alloc(size)
        d = DevBuffer(self, b.handle, size, b.mem, raw=b)
        d.alloc_size = b.alloc_size
        return d

    def free(self, buf):
        if not buf.freed:
            buf.freed = True
            buf.raw.free()

    def prepare(self, program, relocs, label=""):
        return _PreparedJob(program, relocs, label)

    def run_job(self, job, timeout_ms=0):
        if not isinstance(job, _PreparedJob):
            job = self.prepare(job.program, job.relocs, job.label)
        return self.dev.run_raw(job._pptr, job._rptr, job.n_instr, len(job.relocs), timeout_ms, 0)

    def recover(self):
        self.dev.recover()

    def info(self):
        i = self.dev.info()
        return dict(backend=self.name, hw_version=i.hw_version, build_id=i.build_id, state=i.state_name,
                    irq="on" if i.irq_mode else "poll", max_buf_bytes=i.max_buf_bytes,
                    total_buf_bytes=i.total_buf_bytes, total_buf_limit=i.total_buf_limit,
                    jobs_ok=i.jobs_ok, jobs_failed=i.jobs_failed, irqs=i.irq_count,
                    vpu=self.has_vpu, timing="measured (kernel ktime, RUN -> IRQ)" if self.name == "board" else
                    "mock (1 us per instruction)")

    def close(self):
        if self.dev is not None:
            self.dev.close()
            self.dev = None


def driver_diagnosis(path="/dev/tpu0"):
    """Human-readable reason /dev/tpu0 is unusable, with the fix."""
    if not Path("/sys/class/fpga_manager/fpga0").exists():
        return "not a Zynq MPSoC board (no /sys/class/fpga_manager/fpga0): use --backend sim or mock"
    try:
        state = Path("/sys/class/fpga_manager/fpga0/state").read_text().strip()
    except OSError:
        state = "?"
    if state != "operating":
        return f"PL not loaded (fpga0 state={state}): sudo xmutil loadapp tpu_kv260"
    if not Path("/sys/bus/platform/drivers/tpu_kv260").exists():
        return "tpu_kv260 driver not loaded: sudo tpuv1/software/driver/tpu-load.sh (builds via make -C software/driver/kernel)"
    if not Path(path).exists():
        return f"driver loaded but {path} missing: check dmesg | grep tpu (bind via tpu-load.sh)"
    if not os.access(path, os.R_OK | os.W_OK):
        return f"no permission on {path} (mode 0660): run as root or add a udev rule / group"
    return None


class BoardBackend(TpuDeviceBackend):
    name = "board"

    def __init__(self, path="/dev/tpu0"):
        why = driver_diagnosis(path) if not os.path.exists(path) or not os.access(path, os.R_OK | os.W_OK) else None
        if why:
            raise BackendUnavailable(f"board backend unavailable: {why}")
        try:
            dev = tpudrv.Device(path)
        except OSError as e:
            raise BackendUnavailable(f"board backend: cannot open {path}: {e}") from None
        super().__init__(dev, "board")
        hv = self._info.hw_version
        self.has_vpu = (hv >> 8) >= 6 or bool(os.environ.get("TPURT_ASSUME_VPU"))   # assumption: VPU image reports version >= 0x600 (not yet defined by the RTL)


class MockBackend(TpuDeviceBackend):
    name = "mock"

    def __init__(self, fast=True, vpu=True, max_buf=16 << 20, total_limit=256 << 20):
        _paths.need_tpuv1_model()
        import tpu_mock
        if fast:
            from .. import fastsim

            owner = self

            class _Sim(fastsim.FastCoreSim):
                def __init__(self, *a, **k):
                    super().__init__(*a, vpu_enabled=vpu, lut_rom=owner.lut_rom, **k)
            # tpu_mock resolves `isasim.CoreSim` at run time; give it the accelerated subclass
            # (module-level swap, inside this process only; tpuv1 sources are untouched).
            tpu_mock.isasim = types.SimpleNamespace(CoreSim=_Sim)
        else:
            import sim as golden
            tpu_mock.isasim = golden
        self.kernel = tpu_mock.MockKernel(max_buf=max_buf, total_limit=total_limit)
        super().__init__(tpudrv.Device(backend=self.kernel.backend()), "mock")
        self.has_vpu = vpu and fast
        # tpu_mock copies every buffer a job touches into a byte dict: small arenas keep that cheap
        self.preferred_arena = 128 << 10
