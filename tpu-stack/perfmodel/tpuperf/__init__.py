"""tpuperf: cycle-approximate performance + resource model for the tpuv1 INT8 TPU on KV260."""
from .config import HwConfig, tpuv1_board, tpuv1_sim_rom, tpuv1_sim_axi, proposed, XCK26
from .workloads import Gemm, VecOp, Workload, get as get_workload, ZOO
from .model import simulate, Result
from .resources import estimate as estimate_resources

__all__ = ["HwConfig", "tpuv1_board", "tpuv1_sim_rom", "tpuv1_sim_axi", "proposed", "XCK26",
           "Gemm", "VecOp", "Workload", "get_workload", "ZOO", "simulate", "Result",
           "estimate_resources"]
__version__ = "0.1.0"
