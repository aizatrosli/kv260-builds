"""Hardware configuration for the cycle-approximate model.

Every timing knob that the tpuv1 RTL fixes is a field here, so the same model covers the
as-built design (presets ``tpuv1_*``) and hypothetical KV260 variants (``proposed``).
Defaults are the as-built tpuv1 values, read from the RTL:

* ``matrix_unit.sv``   latency L = 2N (R + C for a non-square array)
* ``weight_loader.sv`` one N-byte word shifted per cycle; guard G = 2N after a swap row before
                       the shadow chain may move again -> min tile period G + N + 1 = 3N + 1
* ``matmul_seq.sv``    UB read latency 3 + skid FIFO, rows back-to-back at 1/cycle
* ``decoder.sv``       if_ready = !inst_v -> 2 cycles per instruction; DMA ops +2 legality
                       cycles; SET_CONFIG waits until the activation unit is idle
* ``activation_unit``  8-stage requant pipeline, 1 row/cycle
* ``compile_mlp.py``   33 SET_CONFIGs (2 per column + 1) per output column tile, SYNC(MATRIX)
                       after every column tile, one READ_WEIGHTS per tile in Phase B
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict

# XCK26 (Kria K26 SOM) device totals
XCK26 = dict(lut=117_120, ff=234_240, dsp=1_248, bram36=144, uram=64)


@dataclass(frozen=True)
class HwConfig:
    name: str = "tpuv1"
    # ---- array ---------------------------------------------------------------------------
    rows: int = 16                 # R: input lanes (K dimension per tile)
    cols: int = 16                 # C: output columns (N dimension per tile)
    clock_mhz: float = 100.0
    dsp_packing: bool = False      # 2 INT8 MACs per DSP48E2 (shared activation, WP486 style)
    # weight buffering: "shadow" = tpuv1 (one shadow tile, guard R+C after every swap):
    #   min tile period 2R+C+1 (49 cycles at 16x16, 97 at 32x32);
    # "double" = two shadow banks loaded alternately: a bank is rewritten once the tile two
    #   back has swapped -> min period max(R, (2R+C+1)/2)
    weight_buffering: str = "shadow"
    weight_bits: int = 8           # 8 or 4 (INT4 weights halve weight bytes / bandwidth)
    loader_rows_per_cycle: int = 1  # weight words shifted per cycle (wider loader -> faster)
    # ---- memories --------------------------------------------------------------------------
    ub_kib: int = 1024             # Unified Buffer
    acc_rows: int = 2048           # accumulator rows (two halves, ping-pong per column tile)
    wfifo_tiles: int = 64          # weight FIFO depth in tiles (1024 x 128b in tpuv1)
    ififo_instrs: int = 2048
    # ---- weight source ---------------------------------------------------------------------
    # "ddr"      : READ_WEIGHTS per tile, issued in program order (tpuv1 Phase B)
    # "ddr_prefetch": decoupled weight DMA streams ahead into the FIFO (large bursts)
    # "onchip"   : weights in ROM/URAM next to the loader (1 word/cycle, never late)
    # "axi_push" : host pushes words into the FIFO over AXI at axi_push_bytes_per_cycle
    #              (the jepa-tpu Verilator testbench "weights from AXI" case)
    weight_source: str = "ddr"
    axi_push_bytes_per_cycle: float = 2.35   # fitted: jepa AXI case, 142k stalls
    ub_weight_path: bool = False   # Q.K^T / P.V operands loaded from UB (else via weight path)
    # ---- DDR / AXI -------------------------------------------------------------------------
    hp_ports: int = 1              # 128-bit HP ports used by the DMA
    axi_bytes_per_beat: int = 16
    axi_efficiency: float = 0.80   # beats per cycle achieved on long bursts
    axi_max_mhz: float = 333.0     # HP port / interconnect clock ceiling
    ddr_peak_gbs: float = 19.2     # DDR4-2400 x64
    ddr_efficiency: float = 0.70
    dma_latency_ns: float = 300.0  # fitted; issue -> first beat (iDMA front/back end + PS DDR read)
    dma_check_cycles: int = 2      # decoder legality pipeline per DMA instruction
    # ---- control ---------------------------------------------------------------------------
    instr_cycles: int = 2          # decoder issue rate (cycles per instruction)
    cfg_mode: str = "setconfig"    # "setconfig": 2C+1 SET_CONFIG per column tile
                                   # "param_rom": requant params indexed by layer (1 instr)
    act_blocks_cfg: bool = True    # SET_CONFIG waits for ACTIVATE to drain (tpuv1)
    sync_per_tile: bool = True     # SYNC(MATRIX) after every column tile (tpuv1 compiler)
    ub_latency: int = 3
    seq_pipe: int = 3              # skid FIFO + input register before the array
    acc_pipe: int = 2              # y -> ACC read-modify-write
    act_latency: int = 11          # ACC read + skid + 8-stage requant + UB write
    act_rows_per_cycle: float = 1.0
    # ---- host / job --------------------------------------------------------------------------
    host_overhead_us: float = 0.0  # ioctl + IRQ + program upload (board only)
    io_dma: bool = True            # input/output activations come from / go to DDR
    program_upload: bool = False   # program streamed into the IFIFO by DMA before RUN
    # ---- VPU (transformer ops) ---------------------------------------------------------------
    vpu_lanes: int = 16
    vpu_backpressure: float = 0.0  # fraction of stalled cycles on VPU streams (tb: 0.25)
    vpu_overlap: float = 0.0       # fraction of VPU time hidden under matmuls (0 = serial)
    gelu_in_act: bool = False      # GELU as an activation-unit LUT (free, fused into requant)
    vpu_pipelined: bool = False    # isqrt/divide of job i overlap the streams of job i+1
    with_vpu: bool = False         # count VPU resources
    notes: str = ""

    # ------------------------------------------------------------------ derived quantities
    @property
    def macs_per_cycle(self) -> int:
        return self.rows * self.cols

    @property
    def peak_gops(self) -> float:
        return 2 * self.macs_per_cycle * self.clock_mhz / 1e3

    @property
    def guard(self) -> int:
        return self.rows + self.cols   # swap wavefront reaches PE(R-1, C-1)

    @property
    def load_cycles(self) -> int:
        return -(-self.rows // self.loader_rows_per_cycle)

    @property
    def tile_bytes(self) -> float:
        return self.rows * self.cols * self.weight_bits / 8

    @property
    def array_latency(self) -> int:
        return self.rows + self.cols

    @property
    def dma_bytes_per_cycle(self) -> float:
        """Sustained DMA bytes per PL cycle (HP ports, AXI clock ceiling, DDR ceiling)."""
        axi_clk = min(self.clock_mhz, self.axi_max_mhz)
        port = self.axi_bytes_per_beat * self.axi_efficiency * axi_clk / self.clock_mhz
        ddr = self.ddr_peak_gbs * 1e3 * self.ddr_efficiency / self.clock_mhz   # bytes/cycle
        return min(self.hp_ports * port, ddr)

    @property
    def dma_latency_cycles(self) -> float:
        return self.dma_latency_ns * self.clock_mhz / 1e3

    @property
    def cfg_instrs(self) -> int:
        return 2 * self.cols + 1 if self.cfg_mode == "setconfig" else 1

    def with_(self, **kw) -> "HwConfig":
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------------------
def tpuv1_board() -> HwConfig:
    """As built and measured on the KV260 (Phase B: autonomous DDR -> TPU -> DDR jobs)."""
    return HwConfig(name="tpuv1_board", weight_source="ddr", host_overhead_us=4.0,
                    io_dma=True, program_upload=True)


def tpuv1_sim_rom() -> HwConfig:
    """jepa-tpu Verilator run: weight + instruction FIFOs fed from on-chip ROM, UB preloaded."""
    return HwConfig(name="tpuv1_sim_rom", weight_source="onchip", io_dma=False,
                    host_overhead_us=0.0)


def tpuv1_sim_axi() -> HwConfig:
    """jepa-tpu Verilator run: weights pushed into the Weight FIFO by the AXI testbench master."""
    return HwConfig(name="tpuv1_sim_axi", weight_source="axi_push", io_dma=False,
                    host_overhead_us=0.0)


def proposed(rows=32, cols=32, clock_mhz=200.0, **kw) -> HwConfig:
    """A KV260-sized upgrade: all the cheap control fixes plus a bigger, faster array."""
    base = dict(name=f"proposed_{rows}x{cols}@{int(clock_mhz)}", rows=rows, cols=cols,
                clock_mhz=clock_mhz, dsp_packing=True, weight_buffering="double",
                weight_source="onchip", cfg_mode="param_rom", act_blocks_cfg=False,
                sync_per_tile=False, ub_weight_path=True, ub_kib=256, hp_ports=2,
                gelu_in_act=True, vpu_lanes=cols, vpu_overlap=0.8, with_vpu=True,
                host_overhead_us=4.0, io_dma=True, program_upload=False)
    base.update(kw)
    return HwConfig(**base)
