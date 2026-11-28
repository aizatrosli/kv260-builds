"""Locate the sibling source trees this runtime builds on (never modified, only imported).

  tpuv1     github ilmiahmad/tpuv1 checkout: software/runtime (tpu.py, tpu_isa.py, tpu_mock.py) and
            model/isa (sim.py, asm.py).  On the board only software/runtime is required.
  jepa-tpu  the JEPA integer graph: intops.py, qmodel.npy (quantize.py only for cross-checks).

Override with TPURT_TPUV1=/path and TPURT_JEPA=/path.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../tpu-stack/runtime/tpurt
RUNTIME = HERE.parent                           # .../tpu-stack/runtime
BUILDS = RUNTIME.parents[1]                     # .../kv260-builds

# Keep bytecode caches out of the sibling trees we import from (tpuv1, jepa-tpu are read-only to us).
if sys.pycache_prefix is None and not os.environ.get("PYTHONPYCACHEPREFIX"):
    sys.pycache_prefix = str(RUNTIME / ".pycache")


def _first(env, candidates, marker):
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    for c in candidates:
        c = Path(c).expanduser()
        if (c / marker).exists():
            return c.resolve()
    return None


TPUV1 = _first("TPURT_TPUV1", [BUILDS.parent / "tpuv1", BUILDS / "tpuv1", "~/tpuv1", "/root/tpuv1",
                               "/home/user/tpuv1"], "software/runtime/tpu.py")
JEPA = _first("TPURT_JEPA", [BUILDS / "jepa-tpu", "~/jepa-tpu", "/root/jepa-tpu"], "intops.py")


def _add(p):
    if p is not None and p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def need_tpuv1_runtime():
    """Make tpuv1/software/runtime importable (tpu, tpu_isa, tpu_mock, tpu_mlp)."""
    if TPUV1 is None:
        raise ImportError("tpuv1 checkout not found: set TPURT_TPUV1=/path/to/tpuv1")
    _add(TPUV1 / "software/runtime")


def need_tpuv1_model():
    """Make tpuv1/model/{isa,quantization} importable (sim, asm, tflite_int8). Host-side only."""
    need_tpuv1_runtime()
    for sub in ("model/quantization", "model/isa"):
        if not (TPUV1 / sub).exists():
            raise ImportError(f"{TPUV1 / sub} not found (the ISA simulator is needed for the sim/mock backends)")
        _add(TPUV1 / sub)


def need_jepa():
    if JEPA is None:
        raise ImportError("jepa-tpu not found: set TPURT_JEPA=/path/to/jepa-tpu")
    _add(JEPA)
    return JEPA


def mnist_path():
    """First MNIST npz found (x_test uint8 [N,28,28]); None -> callers use synthetic images."""
    for c in [os.environ.get("TPURT_MNIST", ""), BUILDS / "tpu-stack/quant/data/mnist.npz",
              "/root/data/mnist.npz", "~/data/mnist.npz"]:
        if c and Path(c).expanduser().exists():
            return Path(c).expanduser()
    return None
