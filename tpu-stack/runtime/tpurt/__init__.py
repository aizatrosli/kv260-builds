"""tpurt: host runtime for the tpuv1 INT8 TPU on the AMD Kria KV260.

  backends   get_backend("sim" | "mock" | "board" | "raw" | "auto")      (tpurt.backends)
  buffers    DmaPool / Block / WeightStore: contiguous DMA blocks, zero-copy numpy views,
             cache hooks, weight residency                                (tpurt.buffers)
  program    step graph -> <=2048-instruction TPU programs                (tpurt.program)
  hostops    bit-exact LayerNorm / softmax / GELU / residual (C + numpy)  (tpurt.hostops)
  jepa       JepaExecutor: end-to-end I-JEPA encoder                      (tpurt.jepa)
  profiler   per-op host/device timing                                    (tpurt.profiler)
  vpu        provisional VECTOR-opcode encoding for the future VPU        (tpurt.vpu)

CLIs:  python -m tpurt.bench --help        board_smoke.py --help
"""
from .backends import get_backend, BackendUnavailable, TpuRunError   # noqa: F401

__version__ = "0.1.0"
