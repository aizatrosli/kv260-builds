"""First-order FPGA resource estimator for XCK26, anchored on the measured tpuv1 build.

tpuv1 (16x16, 1 MiB UB, 128 KiB ACC, no VPU) measured: 32.2k LUT, 40k FF, 328 DSP48E2,
32 URAM, 44 BRAM36. The coefficients below reproduce that point and scale per PE, per output
column (requant lane, ACC path, deskew), and per memory (BRAM/URAM aspect ratios).

DSP packing: two INT8 x INT8 products that share an activation fit one DSP48E2
(a * (w1 << 18 + w2), AMD WP486). The lower field only holds a few accumulated products, so
the column cascade is cut every 4 rows and the two fields are re-added in fabric: modelled as
extra LUT/FF per PE.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .config import HwConfig, XCK26

# ---- logic coefficients (fitted to the tpuv1 measurement) -----------------------------------
LUT_FIXED = 18_000          # iDMA, AXI crossbar, decoder, FIFOs control, regs, perf counters
LUT_PER_COL = 600           # requant lane + ACC RMW adder slice + deskew + data setup
LUT_PER_PE = 16             # shadow/active weight mux, valid/swap
LUT_PER_PE_PACKED = 24      # + field split / re-add every 4 rows
FF_FIXED = 18_000
FF_PER_COL = 800
FF_PER_PE = 36
FF_PER_PE_PACKED = 44
DSP_PER_COL = 4             # requant multiplies (32x32 -> 64 split over DSPs)
DSP_FIXED = 8               # decoder DMA-legality multipliers etc.
# ---- VPU (jepa-tpu rtl/vpu.sv, per lane) ---------------------------------------------------
VPU_LUT_PER_LANE = 450
VPU_FF_PER_LANE = 520
VPU_DSP_PER_LANE = 3        # 49-bit LayerNorm product
VPU_BRAM = 2                # exp / GELU LUTs + row buffer
# ---- memory primitives ---------------------------------------------------------------------
URAM_BYTES = 4096 * 72 // 8          # 36 KiB (4096 x 72)
BRAM_MODES = [(512, 72), (1024, 36), (2048, 18), (4096, 9)]

LUT_ROUTABLE = 0.80                  # practical utilisation ceiling for timing closure
TIGHT = 0.70


def bram36(depth: int, width_bits: int) -> int:
    return min(ceil(width_bits / w) * ceil(depth / d) for d, w in BRAM_MODES)


def uram(depth: int, width_bits: int) -> int:
    return ceil(width_bits / 72) * ceil(depth / 4096)


@dataclass
class Resources:
    lut: int
    ff: int
    dsp: int
    bram36: int
    uram: int
    weight_store_bytes: int
    notes: str = ""

    def util(self) -> dict:
        return dict(lut=self.lut / XCK26["lut"], ff=self.ff / XCK26["ff"],
                    dsp=self.dsp / XCK26["dsp"], bram36=self.bram36 / XCK26["bram36"],
                    uram=self.uram / XCK26["uram"])

    @property
    def feasible(self) -> bool:
        u = self.util()
        return (u["lut"] <= LUT_ROUTABLE and u["ff"] <= LUT_ROUTABLE and u["dsp"] <= 1.0
                and u["bram36"] <= 1.0 and u["uram"] <= 1.0)

    @property
    def status(self) -> str:
        if not self.feasible:
            u = self.util()
            over = [k for k, v in u.items() if v > (LUT_ROUTABLE if k in ("lut", "ff") else 1.0)]
            return "INFEASIBLE(" + ",".join(over) + ")"
        if max(self.util().values()) > TIGHT:
            return "tight"
        return "ok"

    def row(self) -> dict:
        u = self.util()
        return dict(lut=self.lut, ff=self.ff, dsp=self.dsp, bram36=self.bram36, uram=self.uram,
                    lut_pct=round(100 * u["lut"], 1), dsp_pct=round(100 * u["dsp"], 1),
                    bram_pct=round(100 * u["bram36"], 1), uram_pct=round(100 * u["uram"], 1),
                    status=self.status)


def place_bytes(nbytes: int, width_bits: int, uram_free: int, bram_free: int):
    """Put a read-only store in URAM first, then BRAM. Returns (uram, bram)."""
    if nbytes <= 0:
        return 0, 0
    depth = ceil(nbytes * 8 / width_bits)
    u = uram(depth, width_bits)
    if u <= uram_free:
        return u, 0
    # fill the free URAM, rest into BRAM
    per_uram_depth = 4096
    cols = ceil(width_bits / 72)
    u_depth = (uram_free // cols) * per_uram_depth
    rest = max(0, depth - u_depth)
    return (uram_free // cols) * cols, bram36(rest, width_bits)


def estimate(cfg: HwConfig, weight_store_bytes: int = 0) -> Resources:
    """Resources of ``cfg``; ``weight_store_bytes`` = on-chip weight ROM/URAM (weight_source
    'onchip')."""
    R, C = cfg.rows, cfg.cols
    pes = R * C
    packed = cfg.dsp_packing
    lut = LUT_FIXED + LUT_PER_COL * C + (LUT_PER_PE_PACKED if packed else LUT_PER_PE) * pes
    ff = FF_FIXED + FF_PER_COL * C + (FF_PER_PE_PACKED if packed else FF_PER_PE) * pes
    if cfg.weight_buffering == "double":   # second shadow register per PE + select
        lut += 4 * pes
        ff += 8 * pes * cfg.weight_bits // 8
    dsp = ceil(pes / (2 if packed else 1)) + DSP_PER_COL * C + DSP_FIXED
    if cfg.weight_bits == 4 and packed:
        dsp = ceil(pes / 2) + DSP_PER_COL * C + DSP_FIXED    # INT4 x INT8 packs the same way
    bram = 0
    ur = 0
    # Unified Buffer: rows of R bytes (input lanes)
    ub_rows = cfg.ub_kib * 1024 // R
    ur += uram(ub_rows, R * 8)
    # Accumulators: acc_rows x C x INT32 in BRAM
    bram += bram36(cfg.acc_rows, C * 32)
    # Weight FIFO (tiles x R words of C*bits)
    bram += bram36(cfg.wfifo_tiles * R, C * cfg.weight_bits)
    # Instruction FIFO
    bram += bram36(cfg.ififo_instrs, 128)
    if cfg.with_vpu:
        lut += VPU_LUT_PER_LANE * cfg.vpu_lanes
        ff += VPU_FF_PER_LANE * cfg.vpu_lanes
        dsp += VPU_DSP_PER_LANE * cfg.vpu_lanes
        bram += VPU_BRAM
    notes = []
    if cfg.weight_source == "onchip" and weight_store_bytes:
        wb = int(weight_store_bytes * cfg.weight_bits / 8)
        u, b = place_bytes(wb, C * cfg.weight_bits * max(1, cfg.loader_rows_per_cycle),
                           XCK26["uram"] - ur, XCK26["bram36"] - bram)
        ur += u
        bram += b
        notes.append(f"weights on chip {wb / 1024:.0f} KiB")
    else:
        weight_store_bytes = 0
    # timing-closure cost of higher clocks: more pipeline registers
    if cfg.clock_mhz > 150:
        ff += int(0.10 * ff)
        lut += int(0.03 * lut)
    return Resources(lut=int(lut), ff=int(ff), dsp=int(dsp), bram36=int(bram), uram=int(ur),
                     weight_store_bytes=weight_store_bytes, notes="; ".join(notes))


def clock_risk(cfg: HwConfig) -> str:
    """Qualitative timing-closure risk on XCK26 (-2 grade fabric; DSP48E2 Fmax ~ 600+ MHz)."""
    f = cfg.clock_mhz
    if f <= 110:
        return "met (tpuv1: +1.25 ns slack at 100 MHz)"
    if f <= 160:
        return "low: register VPU/requant multipliers, decoder DMA checks"
    if f <= 210:
        return "medium: full DSP pipelining (AREG/BREG/MREG/PREG), retimed control"
    if f <= 260:
        return "high: floorplanned array, URAM cascade pipelining"
    return "very high: beyond typical KV260 designs"
