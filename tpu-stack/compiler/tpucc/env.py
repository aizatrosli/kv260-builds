"""Locate the upstream tpuv1 tree and the jepa-tpu model and import their Python modules.

Nothing in those trees is modified; we only put their model directories on sys.path.
Override the locations with TPUV1_ROOT / JEPA_ROOT.
"""
import os
import sys
from pathlib import Path

# never drop __pycache__ into the tpuv1 / jepa-tpu trees we import from (read-only for us)
sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
TPUV1_ROOT = Path(os.environ.get("TPUV1_ROOT", "/home/user/tpuv1"))
JEPA_ROOT = Path(os.environ.get("JEPA_ROOT", str(_HERE.parents[2] / "jepa-tpu")))

for sub in ("model/isa", "model/quantization", "model/cnn", "model/mlp"):
    p = str(TPUV1_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import asm          # noqa: E402  tpuv1 assembler (model/isa/asm.py)
import sim as isasim  # noqa: E402  tpuv1 ISA simulator (model/isa/sim.py)
import tflite_int8  # noqa: E402

N = 16                       # systolic array size
UB_ROWS = isasim.UB_ROWS     # 65536 rows x 16 B = 1 MiB
ACC_ROWS = isasim.ACC_ROWS   # 2048 rows x 16 x int32 = 128 KiB
NSETS = isasim.NSETS         # 4 requant parameter sets
WFIFO_TILES = 64             # Weight FIFO: 1024 words = 64 tiles
IFIFO_WORDS = 2048           # instruction FIFO: max instructions per program


def jepa_modules():
    """Import jepa-tpu's intops (bit-exact integer spec). Returns the module."""
    p = str(JEPA_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    import intops
    return intops


def jepa_quantize():
    """Import jepa-tpu's quantize.py (int_forward golden; pulls in torch)."""
    jepa_modules()
    import quantize
    return quantize


def load_qmodel(path=None):
    import numpy as np
    return np.load(path or (JEPA_ROOT / "qmodel.npy"), allow_pickle=True).item()
