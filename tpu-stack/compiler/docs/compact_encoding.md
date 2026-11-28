# Compact program encoding for tpuv1: proposal (spec only)

Status: proposal. Nothing here is implemented in RTL, `asm.py` or `sim.py`. The size numbers are
computed by `tpucc.encoding.estimate()` from the schedule tpucc actually emits for the JEPA encoder.
Rerun `python -m tpucc.report` to refresh them.

## 1. Why the program is large

| program | instructions | bytes |
|---|---:|---:|
| jepa-tpu baseline: 73 per-matmul programs from `compile_mlp` | 25,878 | 404.3 KiB |
| tpucc, one schedule, no dedupe | 25,928 | 405.1 KiB |
| tpucc, SET_CONFIG dedupe and parameter-set caching (today's ISA) | 21,002 | 328.2 KiB |

After dedupe, 14,370 of the 21,002 instructions (68%) are still `SET_CONFIG`. A linear layer has
per-channel M0, shift and bias, so every 16-column output tile has to rewrite 32 registers. Each
register costs a 16-byte instruction to carry 4–5 bytes of payload. Most of the remaining
instructions are 5,000 `MATMUL`s. Their fields differ only by affine offsets: `ub_row += M` per
k-tile, and the output row `+= M` per n-tile.

The two proposals below are independent of each other. Each one is useful on its own.

## 2. Proposal A: `LOAD_QPARAMS` and a requant-parameter table

### 2.1 The table

The table is a read-only memory of rows. Each row holds one complete 16-channel parameter set, 160 B
(10 × 16 B words):

| word | content |
|---|---|
| 0–3 | M0[c] int32, channels 0..15 (4 per word, little-endian) |
| 4–7 | bias[c] int32, channels 0..15 |
| 8 | shift[c] int8, channels 0..15 |
| 9 | byte 0 zp_out, byte 1 act_min, byte 2 act_max; rest reserved (0) |

The compiler emits one row per *distinct* tile configuration. Identical tiles share a row, for
example every Q·Kᵀ tile of a layer, whose multiplier is one scalar. For the JEPA encoder that is
**452 rows = 70.6 KiB**. The rows are 440 linear-layer tiles plus one row each for qk and pv in
every block. The table can sit in the same BRAM/URAM ROM as the weights in the hard-coded flow, or
be filled once from DDR by DMA (`READ_WEIGHTS`-style, new target) before the first run.

Optional: the shift is the same for the whole layer in this model (−9/−10). With a per-row common
shift the row shrinks to 144 B, about 63.6 KiB.

### 2.2 Instruction

`LOAD_QPARAMS`, proposed opcode `0x33`, 16 bytes:

| bits | field | meaning |
|---|---|---|
| [7:0] | opcode | 0x33 |
| [11:8] | qset | target parameter set (0..3; ≥4 → cfg_err) |
| [31:16] | row | table row (0..65535; ≥ table size → cfg_err) |
| [127:32] | reserved | must be 0 (cfg_err otherwise, like every opcode) |

Semantics: copy row `row` into parameter set `qset`. The effect is exactly 33 `SET_CONFIG`s
(kinds 1, 2, 3). It takes 10 cycles, one table word per cycle.

Hazard: it behaves like `SET_CONFIG` on that set. Proposal: a one-bit-per-set hardware interlock,
where the instruction stalls while an issued `ACTIVATE` that reads the set has not finished. Then
the compiler needs no `SYNC U_ACT` for parameter reloads. The `sim.py` hazard rule changes from
"error" to "stall".

Opcode allocation: 0x33/0x34 are unused in `asm.py`. Coordinate with the VECTOR opcode being added
in parallel.

### 2.3 Size

Each run of requant `SET_CONFIG`s becomes one `LOAD_QPARAMS`. There are 452 runs, one per tile load
that is not already cached in a set:

- **7,084 instructions = 110.7 KiB of program + 70.6 KiB table.**

## 3. Proposal B: `LOOP` with affine field strides (on top of A)

### 3.1 Instruction

`LOOP`, proposed opcode `0x34`, 16 bytes:

| bits | field | meaning |
|---|---|---|
| [7:0] | opcode | 0x34 |
| [15:8] | body_len | number of following instructions in the body (1..255); nested LOOPs count with their bodies |
| [31:16] | count | iterations (1..65535; 0 → cfg_err) |
| [47:32] | d_src | signed stride added to `ub_row` of MATMUL / CONVOLVE per iteration |
| [63:48] | d_dst | signed stride added to `ub_row` of ACTIVATE (and WRITE/READ_HOST_MEM) |
| [75:64] | d_acc | signed stride added to `acc_row` of MATMUL / CONVOLVE / ACTIVATE |
| [87:76] | acc_ring | if ≠ 0, the acc_row offset of this loop wraps modulo acc_ring (ACC double-buffering) |
| [103:88] | d_qrow | stride added to `row` of LOAD_QPARAMS |
| [107:104] | d_kw | stride added to CONVOLVE `kw` |
| [111:108] | d_kh | stride added to CONVOLVE `kh` |
| [112] | acc_first | MATMUL/CONVOLVE `acc` is forced to 0 on iteration 0 and 1 afterwards (K accumulation) |
| [113] | qset_toggle | qset of LOAD_QPARAMS / ACTIVATE is XORed with (iteration & 1) |
| [127:114] | reserved | 0 |

Semantics: the body runs `count` times. Loops nest up to 4 deep, with a stack of (pc, count,
iteration) entries. A field in the body is used as the encoded value plus Σ over the enclosing
loops of iteration × stride, with acc_ring wrapping per loop. Instructions without the field
ignore that stride. `READ_WEIGHTS` sits outside the loops and requests the op's whole tile stream.
This needs the weight DMA to fill the Weight FIFO in the background as space frees (decoupled from
the decoder). Otherwise a stream longer than the FIFO's 64 tiles would deadlock the decoder.

Hardware: a loop buffer of 32 × 16 B replays the body, because the Instruction FIFO is a stream and
cannot be re-read. It also needs a 4-entry loop stack and four 16-bit adders on the field path. This
is small next to the VPU.

### 3.2 Example: `qkv0` (M=64, K=128, N=384 → KT=8, NT=24)

Today this layer is 1,016 instructions (768 SET_CONFIG, 192 MATMUL, 24 ACTIVATE, 29 SYNC,
3 READ_WEIGHTS). In loop form it is 7:

```
READ_WEIGHTS ddr=W_qkv0 tiles=192
LOOP   count=24 body=5 d_dst=64 d_acc=64 acc_ring=128 d_qrow=1 qset_toggle=1   ; n-tiles
  LOAD_QPARAMS qset=0 row=ROW_qkv0
  LOOP count=8 body=1 d_src=64 acc_first=1                                  ; k-tiles
    MATMUL ub_row=H acc_row=0 B=64 new_weights=1
  SYNC   MATRIX
  ACTIVATE acc_row=0 ub_row=QKV rows=64 qset=0
```

The four heads of Q·Kᵀ (or P·V) take one more outer `LOOP count=4` with d_src = 2·64 (next q slice)
and d_dst = 4·64 (next score buffer).

### 3.3 Size

Each TPU op becomes 7 instructions (conv: 10; +1 for an extra M-block loop). DMA and kernel control
instructions are unchanged. Heads are not merged in this estimate, so it is conservative:

- **763 instructions = 11.9 KiB of program + 70.6 KiB table** (vs 404.3 KiB).
- The whole encoder then fits in **one** 2,048-entry Instruction FIFO load. A full-chain program
  needs no ROM-fed instruction stream.

## 4. Effect on the hard-coded memory budget (JEPA encoder)

| item | today (README) | with A+B |
|---|---:|---:|
| weights | 1,154 KiB | 1,154 KiB |
| position embedding | 8 KiB | 8 KiB |
| program | 404 KiB | 12 KiB |
| requant table | (inside program) | 71 KiB |
| **total** | **1,566 KiB = 12.8 Mb** | **1,245 KiB = 10.2 Mb** |

## 5. Smaller steps without new opcodes (already done or possible in tpucc)

- Done: emit a SET_CONFIG only when the register value changes, and reuse a parameter set that
  already holds a tile's config. This gives −19% (404 → 328 KiB). It assumes the parameter
  registers survive a kernel relaunch. Without that assumption the result is 331 KiB.
- Possible: order output tiles across layers so that consecutive tiles share bias and M0. This
  helps nothing for this model, because every channel differs.
- Not possible: folding bias into the matmul through a constant input lane. |bias| is up to
  8.5k, which needs more than 16 int8 weight rows per channel.
