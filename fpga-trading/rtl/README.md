# rtl/ - hardware top-of-book + feature engine

This is synthesizable Verilog (2001/2005 subset, plain `.v`, accepted by Vivado, Verilator and Icarus) for the
engine described in `../SPEC.md`. It takes the 16-byte wire messages on AXI4-Stream, maintains an
8-level bid book and an 8-level ask book, and emits one 8 x int32 feature vector per message.
The vectors match `ftrade.features.TopBook` bit-exactly. The design targets the KV260 (XCK26, Zynq
UltraScale+) at 250 MHz.

| file | what |
|---|---|
| `tob_engine.v` | Top level: AXIS in -> book -> features -> output FIFO -> AXIS out. Also holds the cycle counter, the latency and count registers, and credit flow control. |
| `feed_rx.v` | AXIS slave (32-bit, 4 beats/message, `tlast` on beat 3). Assembles and decodes messages, counts framing errors. |
| `book_side.v` | One side of the book, N=8 levels in flip-flops. Parameter `DESC` sets bid (descending) or ask (ascending). Updates in 2 cycles with parallel compare + shift. |
| `features.v` | F0..F7. A fixed-latency pipeline (21 cycles) that never stalls. |
| `div_r4_pipe.v` | Fully pipelined radix-4 restoring divider (2 quotient bits per stage, one result per cycle). |
| `delay_sr.v` | Delay line with no reset, so Vivado maps it to SRL16/SRL32. |

## Block diagram

```
             s_axis 32b (4 beats/msg)
                  |
           +------v------+  tready = !beat3 | !full | admit   (honest back-pressure)
           |   feed_rx   |  staging regs w0..w2, 1-entry output reg, framing check
           +------+------+  stamps cycle_o on the last beat
                  | msg (kind, side, price, vol, rx_cycle)    valid/ready
     admit = !book_busy && inflight < FIFO_DEPTH              (credit flow control)
                  |
   +--------------v-------------------------------------+
   | cycle A: price vs all 8 slots of the addressed side |  8x(==, >/<) in parallel,
   |          -> match / better thermometers -> per-slot |  prefix-OR, 2-bit select/slot
   |          select {keep,new,prev,next};  dlt=s*v-flow |
   | cycle B: every slot loads its selected source       |  4:1 mux per bit = shift
   |          (insert / delete / overwrite in one cycle) |  flow += dlt >>> 4
   |   [ bid book_side DESC=1 ]   [ ask book_side DESC=0 ] |
   +--------------+--------------------------------------+
                  | b1, a1, bv1..5, av1..5, non-empty, flow, rx_cycle
   +--------------v--------------------------------------+
   | features (21 cycles, no stall)                       |
   |  S1 spread, pair sums  S2 B, A, bv1+av1, mul.1       |
   |  S3 |B-A|, sign, B+A, 3*(bv1+av1), mul.2             |
   |  S4 |B-A|<<15, 3*(B+A), mul.3 (32x32 -> 4 DSP48E2)   |
   |  16 radix-4 stages: micro = sp*bv1 / (bv1+av1)       |
   |   8 radix-4 stages: imb = |B-A|<<15 / (B+A) (+8 SRL) |
   |  F/0 fix-up (sign, x/0 = 0); other fields via SRL    |
   +--------------+--------------------------------------+
                  | 256b vector + rx_cycle
           +------v------+  16 x 288b LUTRAM, write-through bypass when empty
           | output FIFO |
           +------+------+  lat_last = first-valid cycle - rx_cycle
                  |
           m_axis: OUT_W=32 -> 8 beats/msg, word k = F_k, tlast on the 8th (default)
                   OUT_W=256 -> 1 beat/msg, F_k at [32k +: 32], tlast always 1
```

## Interfaces

`tob_engine #(.OUT_W(32 | 256), .FIFO_AW(4))`. All ports share one clock `clk_i`. `rst_ni` is a
synchronous, active-low reset (the AXI `aresetn` convention). Synchronous resets are used throughout, as
Xilinx recommends for UltraScale+. The datapath registers have no reset at all.

| port | dir | description |
|---|---|---|
| `s_axis_tdata[31:0]`, `tvalid`, `tready`, `tlast` | in | SPEC wire format. Word 0 = type/side, 1 = price, 2 = volume, 3 = ts (ignored). |
| `m_axis_tdata[OUT_W-1:0]`, `tvalid`, `tready`, `tlast` | out | Feature vector F0..F7 as int32, little-endian word order. In memory after DMA it is an `int32[8]`. |
| `cycle_o` | out | Free-running cycle counter. |
| `lat_last_o`, `lat_min_o`, `lat_max_o` | out | Per-message latency in cycles: from the cycle the last input beat is accepted to the first cycle the first output beat is valid. |
| `msg_in_o`, `msg_out_o`, `rx_err_o` | out | Messages admitted to the book, vectors started on the output, and framing errors. |

The status outputs are plain registers. Put them behind an AXI GPIO, or behind a 10-line AXI-Lite read mux,
for PS readback. No register port is built in.

Framing errors are counted and the message is discarded, so no vector is emitted for it. There are three cases:
- `tlast` before beat 3 (a runt).
- No `tlast` on beat 3. The receiver then drops beats until the next `tlast`.
- An unknown type byte.

Any nonzero side byte means ask/sell, the same as `ftrade.wire.decode`. The reserved bytes are ignored.

## Latency and throughput (measured in Verilator, `tb/run.sh`)

The pipeline has a fixed latency and does not depend on message content, so clear, level and trade all take the same
time. The numbers below are per message, from the last input beat accepted to the first output beat valid.

| cycle | stage |
|---|---|
| 0 | Last input beat accepted (`tvalid & tready & tlast`). |
| 1 | `feed_rx` output register valid. The book admits the message (cycle A: parallel compare). |
| 2 | Book cycle B: shift/insert/delete commit, flow update. |
| 3 | `features` S1 (reads the committed book). |
| 4..6 | S2..S4: sums, abs/sign, 3*d, 3-stage DSP multiply. |
| 7..22 | 16 radix-4 divider stages (micro_off). imb_q15 uses 8 stages plus an 8-deep SRL alignment. |
| 23 | Sign and zero fix-up, output vector registered. |
| 24 | FIFO bypass loads the output register. |
| **25** | **First output beat valid.** |

| condition | clear | level | trade | throughput |
|---|---|---|---|---|
| Unloaded (gap 40 cycles), either OUT_W | 25 / 25 | 25 / 25 | 25 / 25 | - |
| OUT_W=256, back-to-back input, no stalls | 25 / 25 | 25 / 25 | 25 / 25 | **0.250 msg/cycle** (input-bound: 4 beats/msg) |
| OUT_W=32, back-to-back input, no stalls | 25 / 129 | 37 / 129 | 29 / 129 | **0.125 msg/cycle** (output-bound: 8 beats/msg) |
| OUT_W=256, 85% output stalls | 25 / 183 | 25 / 286 | 25 / 224 | 0.150 |

Latency cells are min / max in cycles. The 70,246-message `all` stimulus produced every row.

At 250 MHz:
- **25 cycles = 100 ns** wire-to-wire inside the engine. That excludes the DMA and PS.
- OUT_W=256 sustains 62.5 M msg/s. OUT_W=32 sustains 31.25 M msg/s.
- Luno sends at most a few thousand messages per second, so this rate is more than enough.

With 32-bit output, a back-to-back burst queues in the 16-deep FIFO. When the FIFO is full, the credit
logic de-asserts `s_axis_tready`, so latency saturates at about 129 cycles. Nothing is ever dropped.

Inside the engine:
- The book can take one message every 2 cycles.
- The feature pipeline can take one per cycle.
- So a 64-bit or 128-bit input stream would raise throughput with no core change.

## Verification

`tb/run.sh` runs the whole flow:
1. Lint both widths.
2. Build the Verilator C++ testbench `tb/tb_tob.cpp` for OUT_W=32 and OUT_W=256.
3. Generate stimulus with `tb/gen_stim.py`. Every message is encoded with `ftrade.wire.to_words`.
4. Run 5 stall configurations per width.
5. Check every vector against `ftrade.features.TopBook` with `tb/check.py`.

`tests/test_rtl_tob.py` wraps the same flow in pytest (17 tests, about 20 s) and skips if `verilator` is missing.

The stimulus has four parts, 70,246 vectors in total:
- **50,000** `ftrade.synth.generate` events.
- **20,000** dense fuzz events: a narrow price band, so matches, deletes, overflow and crossing are frequent. 20% of volumes are 0 and 25% are at or near 2^32. Prices include 0, 2^31 and 2^32-1. There are random clears.
- **235** hand-written edge-case events:
  - empty book, one-sided book
  - overflow at 9 to 12 levels
  - insert worse than worst when full (ignored), insert equal to worst (overwrite)
  - mid-book insert/delete, delete missing, delete-best until empty
  - crossed and locked books
  - top-5 sums that wrap (B = 0 gives imb = -32768)
  - 33-bit B+A, 64-bit products
  - flow wrapping through int32 with 2^32-1 volumes, negative flow, clear mid-stream
- **15** framing frames: 4 malformed (two runts, a missing `tlast`, an unknown type) that must be dropped, and 11 good ones, including one with a junk side byte and reserved bytes.

The testbench also checks:
- Random `tvalid` gaps (30% and 70%) and random `tready` stalls (40% and 85%).
- AXIS rules: data and `tvalid` stay stable while stalled, `tlast` is only on the last beat, no extra or missing vectors.
- `lat_last_o` equals the latency the testbench measures independently, for every message.

`check.py` prints operation coverage. One run had 18.5k inserts, 3.1k insert-with-drop, 9.9k ignored inserts on a
full side, 11.9k overwrites, 17.1k deletes, 5.0k deletes of a missing level, 19.5k crossed or locked vectors and
15.4k top-5 wraps. I also checked that the checker catches mutations: flipping the imb rounding, the flow shift, or
the delete thermometer each gives thousands of mismatches.

**Result: 0 mismatches** on all 10 runs (2 widths x 5 stall configurations x 70,246 vectors), plus the pytest suite.

Lint: `verilator --lint-only -Wall` is clean for both widths. There are four local `lint_off UNUSEDSIGNAL` waivers,
each commented in the source:
- unused levels 6..8 of the book outputs (debug, pruned in synthesis)
- the bits shifted out by `>>> 4`
- the unused borrow of `A-B`
- the unused upper bits of the output word shifter

`iverilog -g2005 -Wall` compiles with no warnings.

## Resources (rough; generic Yosys 0.33 `synth_xilinx -family xcup`, OUT_W=32)

| | count | XCK26 % |
|---|---|---|
| LUT (logic) | ~6.4k (+0.9k INV that Vivado absorbs) | ~5.5% of 117k |
| FF | ~4.8k | ~2% of 234k |
| SRL16/32 | 333 | (delay lines) |
| LUTRAM | 21 RAM32M16 (output FIFO) | |
| DSP48E2 | 4 (32x32 multiply) | 0.3% |
| BRAM / URAM | 0 | |

Most of the FFs are the two divider pipelines: about 133 bits per stage x 24 stages. The tpuv1 TPU uses about 27%
LUT, 17% FF and 26% DSP, so both cores fit together easily.

## Timing at 250 MHz (4 ns): where to look in Vivado

I have not run Vivado. These are design-time estimates of the deepest paths, roughly in order of risk.

1. **Book cycle A** (`book_side`):
   - Path: the `feed_rx` price register fans out to 16 32-bit comparators (8 slots x 2 sides; == and >/< share
     inputs), then an 8-bit prefix-OR, then the 2-bit select encode.
   - Depth: about one CARRY8 chain (4 x CARRY8 for 32 bits) plus 3 to 4 LUT levels. This is the most likely critical path.
   - Fixes if needed: register the price one cycle earlier (it is known at beat 1, three beats before `tlast`). Or
     split equality and magnitude into separate cycles, since there is slack: the book has 2 cycles per message but
     messages arrive at most every 4.
2. **Book cycle B**: a registered select drives a 4:1 mux on 64 bits per slot. The select fans out to 65 loads per
   slot. Vivado will replicate if needed.
3. **Divider stage** (`div_r4_pipe`): three parallel 36-bit subtracts, then a priority 4:1 mux of the remainder.
   That is one 36-bit carry chain (5 x CARRY8) plus 2 LUTs. This path is comfortable. The radix-4 choice keeps it
   at one chain per stage. Radix-2 x 2 per stage would chain two carry chains.
4. **32x32 multiply**: written as `p1 <= a*b; p2 <= p1; p3 <= p2` so Vivado can retime into the AREG/BREG, MREG and
   PREG of the 4-DSP cascade. Check that `p2` and `p3` really land inside the DSPs. If not, add a fourth stage
   (+1 cycle).
5. **Flow update**: cycle A computes `±vol - flow` as a negate-then-subtract of 35 bits, which is two chains (about
   2.5 ns). Cycle B is a 32-bit add. Fine, but it is a known two-chain path.
6. **Output**: LUTRAM async read, then the bypass mux, then an 8:1 word select, into `m_axis_tdata`.
   - `m_axis_tready` drives `load` combinationally, which enables about 40 flops for OUT_W=32 (about 290 for
     OUT_W=256) plus the FIFO pointers. Fine if the consumer drives `tready` from a register, as AXI DMA and AXIS
     register slices do. Otherwise, add an AXIS register slice.
   - The latency min/max update is subtract, then compare, then load, which is two chains. It is statistics only,
     so it can be pipelined by a cycle if it shows up.
7. **`s_axis_tready`** is combinational from the `feed_rx` output-valid register, `st_a` and the credit compare.
   It is short, but it goes straight to the DMA. Add an AXIS register slice in the block design.

## How this maps to a real HFT feed handler / book builder

A production FPGA market-data path has the same stages, wider and faster:

- **Line handler** (`feed_rx`): a real one parses UDP multicast (ITCH/MDP/SBE) out of a 10/25G MAC.
  - It cuts through: fields are decoded as they arrive, not after the whole frame.
  - It does sequence-gap detection and A/B line arbitration.
  - The 16-byte normalized message here stands in for what a parser emits after decoding.
- **Book builder** (`book_side`): the top N levels live in registers and update with parallel compares and a
  one-cycle shift. This is the standard "systolic" or "shift-register" price-level book. Its useful property is
  that latency is constant whatever the price position or book occupancy, so there is no tail.
  - Real systems keep thousands of instruments: per-symbol books live in BRAM/URAM, and only the active symbol's
    top levels are loaded into a register window.
  - Order-by-order feeds (add/modify/delete by order id) also need an order hash table to turn order events into
    level deltas. Here the feed is already level-aggregated, like Luno's.
- **Feature / signal engine** (`features`): fixed-point features (imbalance, microprice, flow EMA) computed in a
  fixed-latency pipeline with no back-pressure.
  - Real designs avoid true dividers where they can, for example by comparing `B*2^15` against `k*(B+A)`, or by
    using DSP reciprocal + Newton. The SPEC here needs exact integer quotients, so a pipelined exact divider is the
    honest choice.
- **Flow control**: the design admits messages with credits and never stalls inside the pipeline. An HFT pipeline
  does the same, except it must never back-pressure the exchange, so it sizes FIFOs for the burst rate and raises
  an alarm on overflow.
- **Latency accounting** (`cycle_o`, `lat_*`): hardware timestamps at ingress and egress, as used for
  tick-to-trade measurement. Production designs use PTP-disciplined clocks.

Ways to cut the 25 cycles, if it ever matters:
- radix-16 or two radix-4 steps per stage: 16 -> 8 divider cycles
- speculative compare on beat 1 (the price arrives three beats before `tlast`)
- skip the `feed_rx` output register in cut-through mode
- use the fact that real spreads are tiny: micro only needs about log2(spread) quotient bits, but that gives
  variable latency, so it is not used here

## Integration on the KV260 next to the TPU (`../../tpu-stack`)

The tpuv1 design (see `tpu-stack/bringup/BRINGUP.md`) uses:
- `pl_clk0` = 100 MHz
- the HPM0 128-bit aperture at `0xA000_0000` (UIO `a0000000.fabric`, 8 MiB window)
- HP0 128-bit for its data, and `pl_ps_irq0`

To add the engine without touching that:

1. **Clocking**: enable `pl_clk1` = 250 MHz (or 200 MHz to start) for the engine. The simplest first bring-up is
   to run it on `pl_clk0` = 100 MHz in the TPU's domain, where latency is 250 ns. Use `proc_sys_reset` to generate
   `rst_ni`.
2. **Data**: add an AXI DMA (scatter-gather off, simple mode) in the engine's clock domain.
   - MM2S stream (32-bit) -> `s_axis`.
   - `m_axis` (OUT_W=32, 8 words, `tlast` per vector; or OUT_W=256 with the DMA's S2MM at 256 bits) -> S2MM.
   - The DMA's AXI masters go through a SmartConnect, which does the clock conversion, into **S_AXI_HP1_FPD**.
     That keeps them off the TPU's HP0. HP1 is 128-bit, like HP0, so `config-afi` needs the HP1 regids set to 0 as well.
   - Add AXIS register slices on both streams for timing.
3. **Control**: put the DMA's AXI-Lite and an AXI GPIO for the status registers on **M_AXI_HPM1_FPD**
   (`0xB000_0000`). `config-afi` already sets HPM1 to 128-bit (`0xA00`). The TPU's HPM0 map is unchanged.
   - Add two `generic-uio` nodes (DMA, status) to a copy of `tpu-stack/bringup/overlay/tpu_kv260.dtso`. Route
     the DMA's `s2mm_introut` to `pl_ps_irq0[1]` (SPI 90).
4. **Software**: the PS writes `ftrade.wire.encode` messages into a CMA/udmabuf buffer and starts MM2S
   (length 16*n). It then receives 32*n bytes on S2MM as `int32[n][8]`, which is exactly
   `ftrade.features.run(events)`. Those feature vectors feed the predictor directly, and could later go to the TPU
   through the existing `tpurt` DMA buffer pool. For live trading the same buffers can be filled from the Luno
   websocket handler on the A53. The FPGA then replaces the Python `TopBook` with a bit-identical, fixed
   100 ns engine. The DMA round trip of a few µs dominates, which is expected for an educational PS-fed design. A
   real deployment feeds the PL from the MAC directly.

## Notes on SPEC / golden-model corner cases (implemented as written, not changed)

- **micro_off for crossed books.** SPEC says to compute F7 on the unsigned 32-bit spread. For a crossed book
  (bid 105, ask 100, bv1 = av1 = 1), F2 = -5 but F7 = floor((2^32-5)*1/2) = **2147483645**, not -3. The
  golden model and the RTL agree, and synth.generate never produces crossed books. A learner fed fuzzed or live
  data should know about this.
- **imb_q15 uses the wrapped top-5 sums.** Five volumes near 2^32 can wrap B to a small number and flip the sign
  of imbalance. This is per SPEC ("as unsigned, mod 2^32"). It is tested.
- **Unknown type byte.** `ftrade.wire.decode` raises (KeyError) and `TopBook` raises ValueError. The RTL drops the
  message, emits no vector, and counts it in `rx_err_o`. Missing or early `tlast` is handled the same way.
- **Side byte other than 0 or 1.** Treated as ask/sell, the same as `wire.decode` (`"ask" if s`).
- **ts field** (word 3): not used by the engine. Only the RTL's own cycle stamps are carried.
