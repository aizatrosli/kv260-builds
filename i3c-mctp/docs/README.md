# I3C + MCTP signal generator for the KV260

This is an I3C Basic SDR Controller and MCTP-over-I3C packet generator for the KV260 PL, controlled by the PS through AXI4-Lite. It produces real bus traffic for I3C and MCTP:
- CCCs and ENTDAA;
- private writes and reads;
- IBI handling;
- multi-packet MCTP messages with PEC;
- the HDR Exit and Target Reset patterns;
- deliberately broken frames (bad T-bit, bad PEC, wrong header version, undersized transmission unit) for negative testing.

**Status:** verified in simulation only (Verilator 5.038). It has not been synthesised in Vivado or run on a board yet. See [What is not verified](#what-is-not-verified).

## 1. Specifications used

Every protocol rule in the RTL, the testbench and the Python model cites one of these documents. The extracted facts, with section numbers, are in `docs/research/`.

| Document | Version | Where it came from |
|---|---|---|
| MIPI I3C Basic | **v1.1.1**, 9 Jun 2021, Public Release Edition (SHA-256 `a0fad683…3265753`) | mipi.org only offers it through an e-mail form, so the unmodified public PDF was taken from a GitHub mirror. See `docs/research/i3c_report.md` §0. |
| DMTF DSP0236 MCTP Base | **1.4.0**, 2026-07-31 (1.3.3 used to cross-check) | dmtf.org |
| DMTF DSP0233 MCTP I3C Transport Binding | **1.0.1**, 2024-03-25 | dmtf.org |
| DMTF DSP0239 MCTP IDs and Codes | **1.12.0**, 2026-02-24 | dmtf.org |
| SMBus | 2.0 | smbus.org. DSP0233 §5.3.1 defines the PEC "same as SMBus". |
| AMD UG1089, DS987, UG1091, DS925, UG571, plus Xilinx Kria-PYNQ and kria-vitis-platforms XDCs | see `docs/research/kv260_report.md` | docs.amd.com, github.com/Xilinx |

Some sources were not available:
- the member-only full MIPI I3C spec;
- I3C Basic v1.2, which is e-mail gated;
- the v1.1.1 errata;
- the KV260 carrier schematic (XTP682) and carrier XDC (XTP686), which need an AMD login.

Anything that would need these is listed as unverified below rather than guessed.

## 2. What the hardware does

```
 PS (Linux, sw/kv260_driver.py)
   │ AXI4-Lite (i3c_mctp_axil.sv)
   ├── CMD FIFO ──┐            ┌── MSG FIFO (message bytes)
   │              ▼            ▼
   │       i3c_mctp_gen.sv: command mux ◄── MCTP sequencer ◄── mctp_i3c_tx.sv (packetizer + CRC-8 PEC)
   │              ▼
   │       i3c_sdr_engine.sv (bit engine: S/Sr/P, OD/PP, T-bit, ACK hand-over, arbitration, ENTDAA, IBI, stall)
   └── RSP FIFO ◄─┘
                  ▼
       kv260_i3c_top.sv: SCL (push-pull), SDA (IOBUF), SDA_PU (switched pull-up), TRIG → PMOD J2
```

| Feature | Implementation | Spec |
|---|---|---|
| START, Repeated START, STOP | `OP_START` / `OP_STOP` | I3C §5.1.2 |
| Arbitrable header after S (Open Drain), Push-Pull header after Sr with an Open-Drain ACK | `OP_ADDR` with or without `F_ARB` | §5.1.2.2, §5.1.2.2.1, §5.1.2.2.4; Tables 88–90 |
| ACK hand-over into Controller write data | Target releases at SCL rise, Controller drives Low from SCL rise | §5.1.2.3.1, Fig 142 |
| T-bit odd parity on writes, T = XOR(D,1) | `OP_WRITE` (`F_PARINV` inverts it) | §5.1.2.3.3 |
| End-of-Data T-bit on reads; Controller abort via Sr | `OP_READ` / `F_LAST` | §5.1.2.3.4 |
| ENTDAA: 64-bit PID+BCR+DCR in Open Drain, DA + PAR = ~XOR(DA) | `OP_DAA`, one command per round | §5.1.4.2 |
| IBI: from a free bus (Target pulls SDA), or won in arbitration; ACK/NACK, MDB read | automatic, `CTRL[1:0]`, `CTRL[7:4]` | §5.1.2.2, §5.1.6.2, §5.1.2.3.2 |
| Controller Clock Stalling, SCL Low only, ≤ 100 µs | engine stalls the 9th bit if the next command is late, then aborts with P | §5.1.2.5, Table 10 |
| HDR Exit Pattern (4 SDA falls) / Target Reset Pattern (14 transitions) | `OP_PATTERN` with 7 or 14 | §5.2.1.1.1 / §5.1.11.3 |
| Any CCC (broadcast or direct, with Defining Byte) | composed from raw commands; codes are in `sw/mctp_i3c.py` (Table 16) | §5.1.9 |
| MCTP packet: `[0001b] [Dest EID] [Src EID] [SOM EOM Seq TO Tag] [IC\|Type …] PEC` | `mctp_i3c_tx.sv` | DSP0233 §5.2.1.1 Table 1; DSP0236 §8.2 Table 1 |
| Fragmentation: equal transmission unit (TU) for non-EOM packets, sequence number mod 4, constant TO and tag | `mctp_i3c_tx.sv` | DSP0236 §8.2, §8.4.2 |
| PEC: CRC-8 0x07, init 0, no reflection, over the address byte with RnW and all packet bytes; excludes 7'h7E | `crc8_pec.sv` | DSP0233 §5.3.1; SMBus 2.0 §5.4.1.3 |
| Optional 7'h7E header before the Target address | `MCTP_CFG0[21]` | DSP0233 §5.2.1.1 (l.405) |
| Target address NACK (buffer full or absent) | P, rest of the packet dropped, counted in `MCTP_STAT` | DSP0233 §5.2.1.2 |
| MCTP read after IBI with MDB 0xAE, ended by T=0 | `read_mctp_packet()` in the driver | DSP0233 §5.2.2.1–5.2.2.2, Table 2 |

**Default timing** (registers `TIM0..TIM2`, 100 MHz PL clock; the testbench checks each value):

| Parameter | Default | Spec limit (pure I3C bus) |
|---|---|---|
| tLOW / tHIGH (Push-Pull) | 40 / 40 ns, fSCL 12.5 MHz | ≥ 24 ns, tDIG ≥ 32 ns, fSCL ≤ 12.9 MHz (Table 87) |
| tLOW_OD / tHIGH (Open Drain) | 250 / 40 ns | tLOW_OD ≥ 200 ns, tDIG_H ≥ 32 ns (Table 86) |
| tHIGH_INIT (first broadcast after reset) | 250 ns | ≥ 200 ns (Table 86) |
| tCAS, tCASr (S/Sr to SCL fall) | 80 ns measured | ≥ 38.4 ns / ≥ 19.2 ns; ≤ 1 µs in ENTAS0 (Table 86) |
| tCBP, tCBSr (SCL rise to P/Sr) | 40 ns measured | ≥ 19.2 ns (Tables 86, 87) |
| Bus free after P | 1 µs | ≥ tCAS on a pure bus (§5.1.3.2.1) |
| Clock stall limit | 90 µs | ≤ 100 µs (Table 10) |

## 3. Verification (simulation)

Run `VL=/path/to/verilator ./run_sim.sh` in `i3c-mctp/`. It needs Verilator 5.030 or newer and python3, and takes under a minute. Add `+VCD` for waves in `build/tb.vcd`.

The tests use pieces written independently of the RTL, so the RTL is never compared with a copy of itself:
- `tb/i3c_target_bfm.sv` is a behavioural I3C Target written from the spec. It handles S/Sr/P detection, header ACK, parity checking, ENTDAA arbitration, GETSTATUS, SETMWL/SETMRL, IBI with an MDB, and private reads with End-of-Data.
- `tb/i3c_timing_mon.sv` measures every SCL/SDA edge against Tables 86/87 and Table 10.
- The testbench resolves the bus as a wired-AND and flags any moment when two drivers fight (contention).
- `sw/mctp_i3c.py` is a separate Python MCTP model. `sw/check_dump.py` compares every byte the Target received with packets the model builds itself.

| Check | Result |
|---|---|
| `tb_i3c_mctp`, 11 scenarios: ENTDAA of 2 Targets (PID arbitration order, PAR, final NACK); broadcast SETMWL; direct GETSTATUS; 150-byte MCTP message → 3 packets; Set EID / Get EID control requests, with and without 7'h7E; corrupted PEC; inverted T-bit; NACK of an absent address; IBI from a free bus with MDB 0xAE then MCTP read with End-of-Data; IBI winning arbitration against a Controller header; HDR Exit and Target Reset patterns; host underrun (3 µs stall absorbed with SCL Low; 150 µs stall aborted at 90 µs) | 37 checks pass, 0 timing violations, 0 contention |
| `check_dump.py`: generator output vs Python model (bytes, PEC, SOM/EOM, sequence number, TU rule, reassembly) | 25 checks pass |
| `tb_axil`: the same flows through the AXI4-Lite register map (ENTDAA, 70-byte MCTP message → 2 packets) | pass |
| Monitor self-test: tLOW_OD 150 ns, tHIGH_INIT 40 ns, 16.7 MHz SCL, tCAS 30 ns and tCBP 10 ns must each be flagged | all 5 caught |
| `sw/test_mctp_i3c.py`: CRC-8/SMBUS check value 0xF4; T-bit examples from §5.1.2.3.3; fragmentation | pass |

**On the Windows + Proxmox setup:** run everything in the Linux VM. Ubuntu's packaged Verilator 5.020 is older than the repo's other flows accept, so build 5.038 from source:

```
git clone -b v5.038 https://github.com/verilator/verilator && cd verilator
autoconf && ./configure && make -j$(nproc) && sudo make install
```

Build dependencies: `flex bison libfl-dev help2man g++ make autoconf`.

## 4. Wiring it to a real I3C Target on the KV260

**Use PMOD J2, the only general-purpose PL header on the KV260.** The carrier has no Raspberry Pi header: J21 is a jumper and J9 is a camera connector (UG1089 Table 1).

J2's 8 signals are HDIO bank 45, and AMD's own XDCs use LVCMOS33 on them, so this is a 3.3 V interface. `constraints/kv260_pmod_i3c.xdc` puts the signals on these pins:

| Signal | pmod[i] | LOC | Drive |
|---|---|---|---|
| SCL | 0 | H12 | push-pull, always |
| SDA | 1 | E10 | IOBUF: push-pull or open-drain |
| SDA_PU | 2 | D10 | '1' through Rp during Open-Drain phases, Z otherwise |
| TRIG | 3 | C11 | high while a frame is on the bus (scope trigger) |

Which J2 physical pin number each pmod[i] lands on is in the login-gated schematic and is not verified here. Check it with a meter before wiring.

**Level translation.** I3C devices run at 1.2, 1.8 or 3.3 V (I3C Table 82). For a 1.2 V or 1.8 V Target you need a translator that passes 12.5 MHz push-pull and also open drain.

- **NXP P3A1604UK** is the best fit found. Its datasheet says it supports I3C:
  - VCCA 0.72–1.98 V, VCCB 1.62–3.63 V;
  - push-pull 20 MHz, open drain 3.4 MHz;
  - it has internal switched 10 kΩ pull-ups and needs at least one external pull-up for open drain.

  Our Open-Drain bit rate is about 1/(250+40 ns) ≈ 3.4 Mbit/s and push-pull is 12.5 MHz, so both are inside its ratings.
- **TI TCA39306** claims "12.5 MHz supported". It is a passive FET translator, and its own datasheet warns that not every I3C use will work through one.
- **NXP P3A9606** tops out at 1.98 V on both sides, so it cannot connect to a 3.3 V PMOD.
- **TXS0102** has no I3C claim and is not recommended.

**Pull-up and High-Keeper.** I3C requires both (§5.1.3.1; Table 82 notes):
- an active Open-Drain pull-up that the Controller switches on only during Open-Drain phases;
- a weak High-Keeper on SDA and SCL.

The RTL drives `SDA_PU` high only in Open-Drain phases. Fit Rp between SDA_PU and SDA, sized for the bus: Rp must be at least (VDD−VOL)/3 mA, which is about 1.0 kΩ at 3.3 V, and at most tr/(0.8473·Cb), which is 2833 Ω for 120 ns and 50 pF (Table 82). Add a weak keeper resistor on both lines.

The FPGA's internal PULLUP (DS925 IRPU 75–190 µA, about 17–44 kΩ) is far too weak for Rp. With a translator, the pull-up belongs on the Target side. How the P3A1604's internal switched pull-ups combine with an external Rp needs its §7.6 and a scope (not verified).

**Read sampling margin.** The engine samples SDA through a 2-flop synchroniser at the SCL rising edge. The pad value it uses is (tLOW_PP − 2 cycles) after SCL falls, which is 20 ns at the defaults. The Target's tSCO is at most 12 ns (Table 87), and a translator adds a delay in each direction. If reads fail at 12.5 MHz, set `TIM0` to tLOW_PP=6 and tHIGH_PP=4 (10 MHz, 40 ns margin) and measure on a scope.

## 5. Software

- **`sw/kv260_driver.py`** runs on the KV260's Linux through `/dev/mem`. The base address is whatever the Vivado address editor assigns (default `--base 0xA0000000`). Examples: `daa 0x0A 0x0B`, `getstatus 0x0A`, `mctp-get-eid 0x0A`. Its methods are `entdaa`, `ccc_broadcast`, `ccc_direct_get`, `send_mctp` and `read_mctp_packet`.
- **`sw/mctp_i3c.py`** holds the spec constants with their table references:
  - message types (DSP0239 Table 1), control commands and completion codes (DSP0236 Tables 12/13), I3C CCC codes (Table 16);
  - packet build, parse and reassembly, and control-message builders.
- **Register map and command format:** header comments of `rtl/i3c_mctp_axil.sv` and `rtl/i3c_sdr_engine.sv`.
- **Vivado (not done here):** add `rtl/*.sv` and the XDC to a project, then build a block design with the Zynq UltraScale+ PS, `kv260_i3c_top` as a module reference on M_AXI_HPM0_FPD through an AXI SmartConnect, `pl_clk0` = 100 MHz, and `proc_sys_reset`.

## What is not verified

**Not done yet:**
- **Hardware.** No Vivado synthesis or timing run, and no board test. The design is plain RTL with LUTRAM FIFOs (4 K × 8 message FIFO, 512 × 32 command and response FIFOs).

**Needs a document we could not get:**
- **MIPI material:** HDR-TSP/TSL and other full-spec features (member-only); changes in I3C Basic v1.2 and the v1.1.1 errata (both gated).
- **Target Reset Pattern edge spacing:** the extracted spec text gives none. This design uses the HDR Exit spacing (≥ tDIG_H) by analogy.
- **IC=1 message integrity check (CRC-32):** DSP0236 leaves it to each message-type spec, for example NVMe-MI. The generator can send IC=1 messages, but the integrity check bytes must be computed by software as part of the message.
- **KV260 board details:** J2 pin numbers, the VCCO_HDA rail, and whether there are series resistors or ESD parts on J2 (all in the login-gated schematic and BOM).
- **NXP NCA9306:** its datasheet was not retrieved.

**Not implemented:**
- **I3C:** HDR-DDR and HDR-BT data modes (only the HDR Exit Pattern is generated), Hot-Join acceptance, Controller Role hand-off, legacy I2C (Fm/Fm+) transfers, and the optional A6 Push-Pull header shortcut (§5.1.2.2.2). All arbitrable headers stay in Open Drain, which the spec always allows.
- **MCTP in hardware:** reassembly of received messages, and the DSP0233 timing rules (PT ≥ 100 ms, MT1–MT4). These are host-software jobs on the PS.
