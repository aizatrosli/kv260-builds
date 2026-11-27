"""Calibration against the measured tpuv1 / jepa-tpu numbers.

Measured references
  board (KV260, 100 MHz, Phase B autonomous jobs, tpuv1 README):
    mlp_wide 354 us (72 % array utilisation), mlp3 67 us, digits_b128 153 us
  Verilator (jepa-tpu README, unmodified tpuv1 core, 73 encoder matmuls, 81.9 M MACs):
    weights from ROM : matmul busy 4.1 ms, 0 weight stalls, 78 % utilisation
    weights from AXI : matmul busy 5.5 ms, 142 k weight-stall cycles, 58 % utilisation
    VPU (tb_vpu, 25 % random stalls) : ~1.8 ms serial

Fitted parameters (everything else is read from the RTL):
  dma_latency_ns, host_overhead_us, axi_efficiency  <- the three board MLPs
  axi_push_bytes_per_cycle                          <- the jepa AXI weight-stall count
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, List

from . import config as C
from . import workloads as W
from .model import simulate


@dataclass
class Bench:
    name: str
    source: str
    measured: float
    unit: str
    fn: Callable[[], float]

    def evaluate(self) -> dict:
        m = self.fn()
        return dict(benchmark=self.name, source=self.source, unit=self.unit,
                    measured=self.measured, model=round(m, 3),
                    error_pct=round(100 * (m - self.measured) / self.measured, 1))


def _board(w):
    return simulate(w, C.tpuv1_board())


def benchmarks(board: C.HwConfig = None, rom: C.HwConfig = None,
               axi: C.HwConfig = None) -> List[Bench]:
    board = board or C.tpuv1_board()
    rom = rom or C.tpuv1_sim_rom()
    axi = axi or C.tpuv1_sim_axi()
    j = W.jepa_tiny()
    b = lambda w: simulate(w, board)                                            # noqa: E731
    return [
        Bench("mlp_wide", "board", 354.0, "us", lambda: b(W.mlp_wide()).total_us),
        Bench("mlp_wide util", "board", 72.0, "%",
              lambda: 100 * (lambda r: r.active_cycles / (r.total_us * r.clock_mhz))(
                  b(W.mlp_wide()))),
        Bench("mlp3", "board", 67.0, "us", lambda: b(W.mlp3()).total_us),
        Bench("digits_b128", "board", 153.0, "us", lambda: b(W.digits(128)).total_us),
        Bench("jepa matmul (ROM weights)", "verilator", 4100.0, "us",
              lambda: simulate(j, rom).matmul_us),
        Bench("jepa util (ROM weights)", "verilator", 78.0, "%",
              lambda: 100 * simulate(j, rom).util),
        Bench("jepa matmul (AXI weights)", "verilator", 5500.0, "us",
              lambda: simulate(j, axi).matmul_us),
        Bench("jepa weight stalls (AXI)", "verilator", 142_000.0, "cycles",
              lambda: simulate(j, axi).weight_stall_cycles),
        Bench("jepa util (AXI weights)", "verilator", 58.0, "%",
              lambda: 100 * simulate(j, axi).util),
        Bench("jepa VPU serial", "verilator", 1800.0, "us",
              lambda: simulate(j, rom.with_(vpu_backpressure=0.25)).vpu_us),
    ]


def calibration_table(**kw) -> List[dict]:
    return [bm.evaluate() for bm in benchmarks(**kw)]


def fit_board(measured=None, lat_ns=range(150, 801, 25), host_us=(2, 3, 4, 5, 6, 8, 10),
              eff=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Grid fit of the three board-level DMA/host parameters (minimax relative error).
    ``measured``: {'mlp_wide': us, 'mlp3': us, 'digits_b128': us} (defaults: README)."""
    measured = measured or {"mlp_wide": 354.0, "mlp3": 67.0, "digits_b128": 153.0}
    ws = {"mlp_wide": W.mlp_wide(), "mlp3": W.mlp3(), "digits_b128": W.digits(128)}
    best = None
    for lat, host, e in itertools.product(lat_ns, host_us, eff):
        cfg = C.tpuv1_board().with_(dma_latency_ns=float(lat), host_overhead_us=float(host),
                                    axi_efficiency=e)
        err = max(abs(simulate(ws[k], cfg).total_us / v - 1) for k, v in measured.items())
        if best is None or err < best[0]:
            best = (err, dict(dma_latency_ns=float(lat), host_overhead_us=float(host),
                              axi_efficiency=e))
    return best


def fit_axi_push(stalls=142_000.0, lo=1.0, hi=16.0, iters=30):
    """Bisection on the testbench AXI push rate so the jepa weight-stall count matches."""
    j = W.jepa_tiny()
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = simulate(j, C.tpuv1_sim_axi().with_(axi_push_bytes_per_cycle=mid)).weight_stall_cycles
        if s > stalls:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
