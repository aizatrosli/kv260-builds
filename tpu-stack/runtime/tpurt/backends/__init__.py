"""Backend factory.  get_backend('sim' | 'mock' | 'board' | 'raw' | 'auto', **options)

  sim    tpuv1 ISA simulator (numpy-accelerated), in process; estimated timing
  mock   tpuv1 tpu_mock (driver ABI model) behind tpuv1's tpu.Device binding
  board  /dev/tpu0 via the tpu_kv260 kernel driver (ioctl + mmap'ed coherent DMA buffers)
  raw    user-space UIO + u-dma-buf, or PYNQ (pynq.MMIO + pynq.allocate) when pynq is importable
  null   jobs complete instantly, nothing computed: host-side cost only (bench)
  auto   board if /dev/tpu0 works, else sim
"""
from .base import Backend, BackendUnavailable, DevBuffer, Job, RunResult, TpuRunError, relocate  # noqa: F401

NAMES = ("sim", "mock", "board", "raw", "null", "auto")


def get_backend(name="auto", **kw):
    if name == "sim":
        from .sim import SimBackend
        return SimBackend(**kw)
    if name == "null":
        from .sim import NullBackend
        return NullBackend(**kw)
    if name == "mock":
        from .tpudev import MockBackend
        return MockBackend(**kw)
    if name == "board":
        from .tpudev import BoardBackend
        return BoardBackend(**kw)
    if name in ("raw", "pynq", "uio"):
        from .raw import RawBackend
        return RawBackend(flavor={"raw": "auto", "pynq": "pynq", "uio": "uio"}[name], **kw)
    if name == "auto":
        try:
            from .tpudev import BoardBackend
            return BoardBackend(**kw)
        except BackendUnavailable:
            from .sim import SimBackend
            return SimBackend()
    raise ValueError(f"unknown backend {name!r} (choose from {', '.join(NAMES)})")
