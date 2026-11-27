# tpu-stack: software for the KV260 TPU

Libraries built around [tpuv1](https://github.com/ilmiahmad/tpuv1) and the `jepa-tpu/` encoder, written while waiting for the board. Everything is tested in Python simulation only. Nothing here edits tpuv1 or `jepa-tpu/`.

Paths default to `/home/user/tpuv1`; set `TPUV1_ROOT` / `TPUV1_DIR` to your clone. Needs Python 3, numpy, torch (CPU) for `quant/`, gcc for the runtime's C kernels.

| Folder | Package | What it does | Tests |
|---|---|---|---|
| `quant/` | `tpuquant` | PyTorch -> INT8/INT4 PTQ (minmax / percentile / MSE), bit-exact integer interpreter, per-layer SQNR & sensitivity, npz+JSON export | 40 pass |
| `compiler/` | `tpucc` | Graph IR, fusion, 16x16 tiling, UB liveness allocation, tpuv1 ISA codegen, schedule report, compact-encoding spec | 59 pass |
| `perfmodel/` | `tpuperf` | Cycle-approximate model calibrated to board + Verilator numbers (<7% error), XCK26 resource estimator, design-space exploration | 34 pass |
| `runtime/` | `tpurt` | Backends (sim / mock / board `/dev/tpu0` / raw UIO / null), DMA buffer pool, end-to-end JEPA executor, `bench`, `board_smoke.py` | 58 pass, 2 skip (board) |
| `bringup/` | — | KV260 bring-up guide, day 0-3 timeline, board/host check scripts, device-tree overlay | `bash -n` + shellcheck clean |

## Headline results

- **Bit-exact:** the full JEPA encoder compiled by `tpucc` runs in tpuv1's unmodified ISA simulator and matches `jepa-tpu/quantize.int_forward`; `tpurt` matches it end to end on sim and mock backends.
- **Program size:** 404 KiB -> 328 KiB by removing redundant SET_CONFIGs. A requant-parameter table plus a LOOP opcode (spec in `compiler/docs/compact_encoding.md`) would bring it to ~12 KiB + a 71 KiB table, fitting the 2048-instruction buffer.
- **Accuracy:** calibration method barely matters (0.9981 cosine for all three). The softmax's fixed 7-bit output is the main loss; a per-layer softmax scale gives 0.99915. INT4 (groups of 32) with three layers kept at INT8 keeps 0.9981 at 612 KiB instead of 1154 KiB.
- **Hardware roadmap** (`perfmodel/RECOMMENDATIONS.md`): today ~6.3 ms/image. Streamlined control (requant table, no per-tile SYNC) is the cheapest win (1.3x on JEPA, zero resources). A 32x32 array with DSP packing at 200 MHz with an overlapped VPU reaches ~0.6 ms (~10x) at ~71% LUT, 60% DSP. 64x64 does not fit the XCK26.
- **Host cost:** LN/softmax/GELU/residual in C take ~0.75 ms/image on x86; estimated 2-2.5 ms on the Cortex-A53, about as much as the TPU. Batch >= 8 or move them onto the VPU.

## First things to run on the board

See `bringup/TIMELINE.md`. Short version:

1. `bringup/scripts/board_check.sh` (Ubuntu, boot firmware, xmutil, kernel headers).
2. Build/install the tpuv1 bitstream with `bringup/scripts/install_app.sh`, load the driver (`tpuv1/software/driver/tpu-load.sh`).
3. `sudo python3 runtime/board_smoke.py` (driver -> registers -> DMA loopback -> 16x16 matmul -> MLP -> JEPA).
4. `sudo python3 -m tpurt.bench --backend board --json day1.json`, then `python -m tpuperf calibrate` with the real numbers.
