"""Instruction encoders: tpuv1's own board-safe software/runtime/tpu_isa.py (re-exported), plus the
vector-unit extension ops X_VEC/X_LUT/X_VCFG of jepa_top (see tpurt/vpu.py)."""
from . import _paths

_paths.need_tpuv1_runtime()
from tpu_isa import (OP_NOP, OP_RHM, OP_RW, OP_WHM, OP_MATMUL, OP_ACTIVATE, OP_SET_CONFIG,   # noqa: E402,F401
                     OP_SYNC, OP_IRQ, OP_HALT, U_DMA, U_WEIGHT, U_MATRIX, U_ACT, U_ALL,
                     nop, halt, sync, interrupt_host, matmul, activate, set_config,
                     read_host_mem, write_host_mem, read_weights, to_bytes)

DMA_OPS = (OP_RHM, OP_RW, OP_WHM)
MAX_INSTR = 2048
UB_ROWS, ACC_ROWS, ACC_HALF, N = 65536, 2048, 1024, 16
MAX_DMA_ROWS = 4095                      # row_len field is 12 bits


def opcode(word):
    return word & 0xFF


def dma_fields(word):
    """(ub_row, ddr_addr, row_len, reps, ddr_stride_bytes) of RHM/WHM; (tiles, ddr_addr) for RW."""
    op = word & 0xFF
    addr = ((word >> 32) & ((1 << 45) - 1)) << 4
    if op == OP_RW:
        return (word >> 16) & 0xFFFF, addr
    return ((word >> 16) & 0xFFFF, addr, (word >> 77) & 0xFFF, (word >> 89) & 0xFFF,
            ((word >> 101) & 0xFFFFFF) << 4)


def dma_span(word):
    """Bytes of DDR one DMA instruction touches (same rule as the kernel's dma_span)."""
    op = word & 0xFF
    if op == OP_RW:
        return ((word >> 16) & 0xFFFF) * 256
    reps, length, stride = (word >> 89) & 0xFFF, ((word >> 77) & 0xFFF) * 16, ((word >> 101) & 0xFFFFFF) * 16
    return (reps - 1) * stride + length if reps and length else 0


def patch_addr(word, addr):
    if addr & 15 or addr >= 1 << 49:
        raise ValueError(f"DMA address {addr:#x} must be 16-byte aligned and < 2^49")
    return (word & ~(((1 << 45) - 1) << 32)) | ((addr >> 4) << 32)
