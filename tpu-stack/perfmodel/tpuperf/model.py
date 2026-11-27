"""Cycle-approximate timing model.

The model replays the tpuv1 program structure tile by tile (an event model, not a cycle loop):

  for each M chunk (<= half the accumulator rows)       # ACC ping-pong halves
    for each output column tile nt                       # one "group"
      SET_CONFIG x (2C+1)       (wait for ACTIVATE idle)  # cfg_mode / act_blocks_cfg
      for each K tile kt
        [READ_WEIGHTS]          (Phase B, program order) # weight_source == "ddr"
        MATMUL B rows           (tile swap waits for the weight loader)
      SYNC(MATRIX)              (drain R + C + pipeline) # sync_per_tile
      ACTIVATE B rows           (8-stage requant, overlaps the next group's setup)

Per tile the swap row waits for: the matrix sequencer (previous tile's rows), the UB read
pipeline, the weight loader (guard R + C after the previous swap in the shadow design, then
R shift cycles), and the weight words themselves (DMA latency + bytes / bandwidth, FIFO space).
VPU ops use the jepa-tpu vpu.sv FSM cycle counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from math import ceil
from typing import Dict, List

from .config import HwConfig
from .workloads import Gemm, VecOp, Workload

NEG = -1e18


@dataclass
class GemmTiming:
    cycles: float = 0.0          # wall cycles of this GEMM (all instances)
    active: float = 0.0          # array rows streamed (tpuv1 perf counter 'array_active')
    weight_stall: float = 0.0    # swap row ready but tile not loaded (perf counter 'weight_stall')
    tiles: int = 0
    groups: int = 0
    instrs: int = 0
    weight_bytes_moved: float = 0.0
    spill_cycles: float = 0.0

    def add(self, o: "GemmTiming", k: int = 1):
        for f in ("cycles", "active", "weight_stall", "tiles", "groups", "instrs",
                  "weight_bytes_moved", "spill_cycles"):
            setattr(self, f, getattr(self, f) + k * getattr(o, f))


@dataclass
class Result:
    workload: str
    config: str
    clock_mhz: float
    macs: int
    matmul_cycles: float
    active_cycles: float
    weight_stall_cycles: float
    vpu_cycles: float
    vpu_exposed_cycles: float
    io_cycles: float
    host_us: float
    instrs: int
    tiles: int
    weight_bytes_moved: float
    spill_cycles: float
    per_layer: Dict[str, float] = field(default_factory=dict)

    @property
    def total_cycles(self) -> float:
        return self.matmul_cycles + self.vpu_exposed_cycles + self.io_cycles

    @property
    def matmul_us(self) -> float:
        return self.matmul_cycles / self.clock_mhz

    @property
    def vpu_us(self) -> float:
        return self.vpu_cycles / self.clock_mhz

    @property
    def total_us(self) -> float:
        return self.total_cycles / self.clock_mhz + self.host_us

    @property
    def util(self) -> float:
        """Array utilisation over matmul time (tpuv1 'array_active / total' definition)."""
        return self.active_cycles / self.matmul_cycles if self.matmul_cycles else 0.0

    @property
    def gops(self) -> float:
        return 2 * self.macs / (self.total_us * 1e3) if self.total_us else 0.0

    def row(self) -> dict:
        return dict(workload=self.workload, config=self.config, clock_mhz=self.clock_mhz,
                    gmacs=round(self.macs / 1e9, 4), total_us=round(self.total_us, 2),
                    matmul_us=round(self.matmul_us, 2), vpu_us=round(self.vpu_us, 2),
                    io_us=round(self.io_cycles / self.clock_mhz, 2), host_us=self.host_us,
                    util=round(self.util, 4), gops=round(self.gops, 2),
                    weight_stall_cycles=int(self.weight_stall_cycles), instrs=self.instrs,
                    weight_mb_moved=round(self.weight_bytes_moved / 1e6, 3),
                    fps=round(1e6 / self.total_us, 1) if self.total_us else 0.0)


# ---------------------------------------------------------------------------------------------
# GEMM tile-level event model
# ---------------------------------------------------------------------------------------------
def _dyn_source(cfg: HwConfig) -> str:
    if cfg.ub_weight_path:
        return "onchip"
    if cfg.name.startswith("tpuv1_sim"):
        return cfg.weight_source          # jepa-tpu streamed K/V through the weight stream
    # no UB -> weight path: K/V written to DDR and read back as weights
    return "ddr_roundtrip"


def _instance(cfg: HwConfig, M: int, KT: int, NT: int, source: str) -> GemmTiming:
    R, C = cfg.rows, cfg.cols
    ic = cfg.instr_cycles
    pipe_in = cfg.ub_latency + cfg.seq_pipe
    lat = cfg.dma_latency_cycles
    bw = cfg.dma_bytes_per_cycle
    tbytes = cfg.tile_bytes
    tile_xfer = tbytes / bw
    push_xfer = tbytes / cfg.axi_push_bytes_per_cycle
    half = max(1, cfg.acc_rows // 2)
    mchunks = [min(half, M - i) for i in range(0, M, half)]

    t = 0.0                  # decoder time
    act_done = 0.0
    act_done_hist: List[float] = []   # per group, for ACC ping-pong reuse
    seq_free = 0.0
    row_end = 0.0
    swaps: List[float] = []
    load_done_prev = NEG
    arr_prev = 0.0 if source in ("axi_push",) else NEG
    chan_free = 0.0
    stall = 0.0
    active = 0.0
    instrs = 0
    tiles = 0
    last_drain = 0.0
    prog_ddr = source in ("ddr", "ddr_roundtrip")
    if source == "ddr_prefetch":
        arr_prev = lat - tile_xfer   # one latency, then streaming
    for m in mchunks:
        for nt in range(NT):
            g = len(act_done_hist)
            # ---- parameter setup
            if cfg.act_blocks_cfg:
                t = max(t, act_done)
            t += cfg.cfg_instrs * ic
            instrs += cfg.cfg_instrs
            for kt in range(KT):
                i = len(swaps)
                # ---- weight words
                if source == "onchip":
                    arr = NEG
                elif source == "axi_push":
                    space = swaps[i - cfg.wfifo_tiles] if i >= cfg.wfifo_tiles else 0.0
                    arr = max(arr_prev, space) + push_xfer
                    arr_prev = arr
                elif source == "ddr_prefetch":
                    space = swaps[i - cfg.wfifo_tiles] if i >= cfg.wfifo_tiles else NEG
                    arr = max(arr_prev, space) + tile_xfer
                    arr_prev = arr
                else:   # program-order READ_WEIGHTS
                    t += ic + cfg.dma_check_cycles
                    instrs += 1
                    start = max(t + lat, chan_free)
                    arr = start + tile_xfer
                    chan_free = arr
                # ---- MATMUL command
                t_mm = max(t + ic, seq_free)
                t = t_mm
                instrs += 1
                earliest = t_mm + pipe_in
                if kt == 0 and not cfg.sync_per_tile and g >= 2:
                    earliest = max(earliest, act_done_hist[g - 2])   # ACC half reuse
                if cfg.weight_buffering == "double":
                    lf = load_done_prev
                    if i >= 2:
                        lf = max(lf, swaps[i - 2] + cfg.guard + 1)
                else:
                    lf = swaps[i - 1] + cfg.guard + 1 if i >= 1 else NEG
                ready = max(lf + cfg.load_cycles, arr + 1)
                load_done_prev = ready
                data_ready = max(earliest, row_end)
                row_start = max(data_ready, ready)
                stall += row_start - data_ready
                swaps.append(row_start)
                row_end = row_start + m
                seq_free = row_end - pipe_in
                active += m
                tiles += 1
            drain = row_end + cfg.array_latency + cfg.acc_pipe
            last_drain = drain
            act_time = m / cfg.act_rows_per_cycle + cfg.act_latency
            if cfg.sync_per_tile:
                t = max(t + ic, drain)                       # SYNC(MATRIX)
                t += ic                                      # ACTIVATE
                act_done = max(t, act_done) + act_time
                instrs += 2
            else:
                t += ic
                act_done = max(drain, act_done) + act_time
                instrs += 1
            act_done_hist.append(act_done)
    end = max(t, act_done, last_drain) + 2 * ic             # final SYNC + IRQ
    instrs += 2
    gt = GemmTiming(cycles=end, active=active, weight_stall=stall, tiles=tiles,
                    groups=len(act_done_hist), instrs=instrs)
    return gt


@lru_cache(maxsize=65536)
def _gemm_cached(cfg: HwConfig, M, K, N, count, dynamic, taps, depthwise, ub_in, ub_out,
                 host_im2col=False):
    g = Gemm("x", M, K, N, count=count, dynamic=dynamic, taps=taps, depthwise=depthwise,
             in_bytes=ub_in, out_bytes=ub_out, host_im2col=host_im2col)
    R, C = cfg.rows, cfg.cols
    KT, NT = g.k_tiles(R, C), g.n_tiles(C)
    source = _dyn_source(cfg) if dynamic else cfg.weight_source
    one = _instance(cfg, M, KT, NT, source)
    total = GemmTiming()
    total.add(one, count)
    bw = cfg.dma_bytes_per_cycle
    # weight bytes actually moved from off-chip
    if source in ("ddr", "ddr_prefetch", "axi_push", "ddr_roundtrip"):
        total.weight_bytes_moved = one.tiles * cfg.tile_bytes * count
    if source == "ddr_roundtrip":       # K/V must first be written to DDR
        total.cycles += count * (cfg.dma_latency_cycles + g.operand_bytes / count / bw)
    # UB capacity: excess working set spills to DDR and comes back (serial DMA)
    ws = g.ub_in + g.ub_out
    ub = cfg.ub_kib * 1024
    if ws > ub and cfg.io_dma:
        spill = 2 * (ws - ub) / bw + 2 * cfg.dma_latency_cycles
        if cfg.weight_source in ("ddr_prefetch", "onchip"):
            # decoupled DMA engine: spill traffic overlaps compute, sharing DMA bandwidth
            # with the weight stream
            spill = max(0.0, spill + total.weight_bytes_moved / bw - total.cycles)
        # tpuv1: READ/WRITE_HOST_MEM wait for the matrix/activation units -> serial
        total.spill_cycles = spill
        total.cycles += spill
    return total


def gemm_timing(cfg: HwConfig, g: Gemm) -> GemmTiming:
    return _gemm_cached(cfg, g.M, g.K, g.N, g.count, g.dynamic, g.taps, g.depthwise,
                        g.ub_in, g.ub_out, g.host_im2col)


# ---------------------------------------------------------------------------------------------
# VPU (jepa-tpu rtl/vpu.sv FSM)
# ---------------------------------------------------------------------------------------------
SQRT_CYCLES = 22     # bit-by-bit isqrt from 2^40, 2 bits per cycle
DIV_CYCLES = 30      # restoring division, 30 quotient bits


def vpu_job_cycles(kind: str, rows: int, backpressure: float = 0.0,
                   pipelined: bool = False) -> float:
    """vpu.sv FSM: IDLE(1) LOAD(rows) [EXPSUM(rows)] [SQRT 22] [DIV 30] EMIT(rows).
    ``pipelined``: the scalar isqrt/divide overlaps the neighbouring jobs' streams."""
    s = 1.0 / (1.0 - backpressure)          # random valid/ready stalls on the streams
    stream = 1 + 2 * rows * s
    if kind in ("add", "gelu"):
        return stream
    if kind == "ln":
        scalar = SQRT_CYCLES + DIV_CYCLES
        return max(stream, scalar) if pipelined else stream + scalar
    if kind == "softmax":
        stream += rows
        return max(stream, DIV_CYCLES) if pipelined else stream + DIV_CYCLES
    raise ValueError(kind)


def vecop_cycles(cfg: HwConfig, v: VecOp) -> float:
    if v.kind == "gelu" and cfg.gelu_in_act:
        return 0.0
    rows = ceil(v.elems / cfg.vpu_lanes)
    return v.jobs * vpu_job_cycles(v.kind, rows, cfg.vpu_backpressure, cfg.vpu_pipelined)


# ---------------------------------------------------------------------------------------------
# Whole workload
# ---------------------------------------------------------------------------------------------
def simulate(w: Workload, cfg: HwConfig) -> Result:
    mm = GemmTiming()
    per_layer: Dict[str, float] = {}
    for g in w.gemms:
        gt = gemm_timing(cfg, g)
        mm.add(gt)
        key = g.name.split(".")[-1]
        per_layer[key] = per_layer.get(key, 0.0) + gt.cycles
    vpu = 0.0
    for v in w.vecops:
        c = vecop_cycles(cfg, v)
        vpu += c
        key = "vpu_" + v.kind
        per_layer[key] = per_layer.get(key, 0.0) + c
    bw = cfg.dma_bytes_per_cycle
    io = 0.0
    if cfg.io_dma:
        lat = cfg.dma_latency_cycles
        io += lat + w.input_bytes / bw + lat + w.output_bytes / bw
        if cfg.program_upload:
            io += lat + mm.instrs * 16 / bw
    return Result(workload=w.name, config=cfg.name, clock_mhz=cfg.clock_mhz, macs=w.macs,
                  matmul_cycles=mm.cycles, active_cycles=mm.active,
                  weight_stall_cycles=mm.weight_stall, vpu_cycles=vpu,
                  vpu_exposed_cycles=vpu - cfg.vpu_overlap * min(vpu, mm.cycles), io_cycles=io,
                  host_us=cfg.host_overhead_us, instrs=mm.instrs, tiles=mm.tiles,
                  weight_bytes_moved=mm.weight_bytes_moved, spill_cycles=mm.spill_cycles,
                  per_layer=per_layer)
