# sigrok / PulseView decoders: I3C SDR -> MCTP over I3C -> MCTP

Three stacked libsigrokdecode (API 3) decoders that turn an SCL/SDA capture into I3C frames, MCTP
packets and reassembled MCTP messages, with the same spec references as the generator
(see [../docs/README.md](../docs/README.md) §1).

| Decoder id  | Input -> output        | What it shows |
|-------------|------------------------|---------------|
| `i3c_sdr`   | logic -> `i3c`         | S / Sr / P, arbitrable vs push-pull address header, ACK/NACK, data bytes with the T-bit (odd parity on writes, End-of-Data on reads), broadcast and direct CCCs by name (I3C Basic v1.1.1 Table 16), ENTDAA rounds (PID / BCR / DCR / assigned DA, DA parity), HDR Exit and Target Reset patterns, optional SDR timing check (tLOW_OD >= 200 ns, push-pull tLOW >= 24 ns, Table 86/87). |
| `mctp_i3c`  | `i3c` -> `mctp_pkt`    | One MCTP packet per private write/read (DSP0233 v1.0.1): header version, dest/src EID, SOM/EOM/Seq/TO/Tag, IC + message type, CRC-8 PEC over the address byte with RnW and the packet (§5.3.1), and a one-byte 0xAE read after START as the MCTP Pending Read IBI (Table 2). Option `addrs` restricts it to some Target addresses. |
| `mctp`      | `mctp_pkt` -> -        | Message reassembly by (tag, TO, source EID) with sequence and transmission-unit checks (DSP0236 v1.4.0 §8.4.2, §8.6, §8.9), message type names (DSP0239 v1.12.0 Table 1), MCTP Control Rq/D/IID, command (Table 12), completion code (Table 13), and the payloads of Set/Get Endpoint ID, Get MCTP Version Support and Get Message Type Support. |
| `amd_apml`  | `i2c` or `i3c` -> -    | AMD APML sideband: SB-TSI (temperature sensor) and SB-RMI (remote management) register accesses, the SB-RMI soft mailbox, and the processor state access commands (read MSR / CPUID). Works on SMBus (stack on mainline `i2c`) and on I3C (stack on `i3c_sdr`). See [AMD APML](#amd-apml-sb-rmi-and-sb-tsi) below. |

The id is `i3c_sdr`, not `i3c`, so it can sit next to other community I3C decoders.
Mainline libsigrokdecode has no I3C decoder.

## Install

Copy the folders in [decoders/](decoders) (`i3c_sdr`, `mctp_i3c`, `mctp`, `amd_apml`) into a decoder
directory that libsigrokdecode searches, then restart PulseView:

- Windows, per user: `%LOCALAPPDATA%\libsigrokdecode\decoders\`
- Windows, all users: `%ProgramData%\libsigrokdecode\decoders\`
- Linux / the Proxmox VM: `~/.local/share/libsigrokdecode/decoders/`
- Anywhere: point `SIGROKDECODE_PATH` at this `decoders/` folder. (The `sigrok-cli` launcher in the
  Sipeed Windows package sets `SIGROKDECODE_DIR` itself, so use `SIGROKDECODE_PATH` or the folders
  above rather than `SIGROKDECODE_DIR` there.)

The SLogic32U3 needs Sipeed's sigrok fork (driver `sipeed-slogic-analyzer`): mainline PulseView does
not know the device. Install PulseView / sigrok-cli from https://github.com/sipeed/SLogic/releases.
The decoders were tested with two libsigrokdecode builds: distro 0.5.3, and commit 71f45144, the
one the Sipeed release bundles.

Check the install with `sigrok-cli -L | grep -i -e i3c -e mctp`.

## Try it without hardware

[examples/i3c_mctp_demo_sim_200MSps.sr](examples/i3c_mctp_demo_sim_200MSps.sr) is the bus from the
Verilator testbench (all scenarios of `tb_i3c_mctp`) resampled to 200 MS/s. Open it in PulseView, add
the decoder **I3C SDR** (assign SCL/SDA), then stack **MCTP/I3C** on it and **MCTP** on that.
From the command line:

```
sigrok-cli -i examples/i3c_mctp_demo_sim_200MSps.sr -P i3c_sdr,mctp_i3c,mctp \
           -A i3c_sdr=xfer:daa:warning,mctp_i3c=packet:ibi:warning,mctp=message:control:warning
```

It contains ENTDAA of two Targets, SETMWL, direct GETSTATUS, a 3-packet 150-byte vendor message,
Set/Get Endpoint ID, one injected parity error, one injected PEC error, two Pending Read IBIs (one
won by arbitration), a NACK of an absent address, the Target Reset and HDR Exit patterns, and two
Controller clock stalls (the second one hits the stall limit and the frame is aborted after 0x33).

## Probing with the SLogic32U3

- **Sample rate.** Push-pull SDR SCL runs up to 12.9 MHz with tLOW/tHIGH down to 24 ns (Table 87). Use the
  4-channel mode, 800 MS/s on a 5 Gbps port or up to 1400 MS/s on 10 Gbps (Sipeed's figures), so each
  short phase gets several samples. The decoders were checked at 200 and 100 MS/s on the default
  generator timing (4 clocks = 40 ns per push-pull phase); 100 MS/s is the lowest rate tested and is too coarse for faster
  timing settings. The timing check needs at least ~4 samples per phase to be meaningful.
- **Threshold.** The SLogic threshold is 0-6 V in 0.1 V steps (default 1.7 V, which is wrong for
  1.8 V I3C). Set it to half the bus voltage: 0.9 V for 1.8 V, 0.6 V for 1.2 V, 1.6-1.7 V on the
  3.3 V FPGA side of the level translator. Example:
  `sigrok-cli -d sipeed-slogic-analyzer -c samplerate=800m:voltage_threshold=0.9-0.9 --channels 0,1 ...`
  Check the exact option and channel names with `sigrok-cli -d sipeed-slogic-analyzer --show` on
  your build; they were not verified on a real unit.
- **Where to probe.** Probe SCL and SDA on the Target side of the translator (that is the I3C bus
  the spec timing applies to), ground at the Target. The FPGA side (PMOD J2 pmod[0] SCL, pmod[1] SDA)
  shows the Controller's intent but not the Target's drive through the translator.
- **Trigger.** `trig` on pmod[3] (LOC C11) is high while a frame is on the bus. Put it on a third
  channel and trigger on its rising edge, or trigger on SDA falling while SCL is high (START).
- **Loading.** Each SLogic input is 100 kOhm; its capacitance is not published. I3C allows at most
  50 pF bus capacitance (Cb), so keep probe leads short and check rise times in open-drain phases.

## AMD APML: SB-RMI and SB-TSI

`amd_apml` decodes the two APML targets of an AMD processor socket. Stack it on **I2C** for SMBus
parts or on **I3C SDR** for APML-I3C parts:

```
sigrok-cli -i examples/apml_smbus_synthetic_50MSps.sr -P i2c,amd_apml -A amd_apml
sigrok-cli -i examples/apml_i3c_synthetic_200MSps.sr  -P i3c_sdr,amd_apml -A amd_apml
```

It shows:

- **SB-TSI:** every register by name, with its fields decoded. The CPU temperature and the high and
  low thresholds are combined from their integer and decimal registers in 0.125 degC steps; the
  offset is decoded as 11-bit two's complement. It also handles Send/Receive Byte through the
  address pointer, and the SMBus Alert Response Address read.
- **SB-RMI registers:** single and block reads and writes. On SMBus the optional PEC is checked.
- **SB-RMI soft mailbox:** one line for the request (message name, DataIn) when SoftwareInterrupt is
  written, and one for the response (DataOut with its unit, error code) once OutBndMsg1-4 have been read.
- **Processor state access (71h/72h/73h):** Read Register (MSR), Read CPUID, Load Address, Write
  Register and Read Data/Status. The thread number is decoded, including the second thread byte on
  rev 21h; the status code (Table 134) and the result are shown. The SMBus intermediate and final
  PECs are checked.
- **I3C addresses:** SETDASA and SETNEWDA are followed, so the decoder recognises the targets by
  their static address after they get a dynamic address.

Options:

- `rmi_addrs` (default `3c,38`) and `tsi_addrs` (default `4c,48`) are static addresses for socket 0
  and socket 1.
- `rmi_cmd_len` (`auto`, `1`, `2`) sets the SB-RMI command code length: 2 bytes on rev 21h parts
  ([57238] 5.4.1.1). In `auto` the decoder learns it from a Revision read (21h means 2 bytes), from
  SB-TSI Config[SBRMIAddrMode], or from the length field of a 73h command. If none of these appears
  in the capture, set it by hand.

Sources: AMD Processor Programming References, downloaded from AMD's documentation portal
(docs.amd.com, search for the number):

| PPR | Part | APML | Used for |
|-----|------|------|----------|
| 56214 rev 3.05 | Family 19h Model 21h | SMBus, SB-RMI rev 10h | ch. 5 APML (Tables 79-91), 5.6 SB-RMI registers, ch. 6 SB-TSI (Tables 92-95) |
| 55901 rev 0.47 | Family 19h Model 11h | I3C, SB-RMI rev 20h | vol 3 ch. 4: 1-byte command over I3C, Thread128CS |
| 57238 rev 0.24 | Family 1Ah Model 02h | I3C, SB-RMI rev 21h | vol 4 ch. 5 (Tables 127-149), 5.6 registers, ch. 6 SB-TSI (Tables 150-153) |

Where the sources fall short, so check against your part's own PPR:

- **SB-RMI address.** None of the three PPRs gives the SB-RMI address as a number. 56214 shows the
  address byte as `0111_XXX0b`, which is 7-bit 38h-3Fh. The defaults 3Ch (socket 0) and 38h
  (socket 1) come from the Linux `sbrmi` driver documentation. SB-TSI 4Ch/48h is in 57238 Table 153.
- **Mailbox message names.** The names for 40h and above come from a PDF table whose cells wrap, so
  they were put back together by hand. Message 10h/11h are named after 56214; they are reserved in 57238.
- **SMBus ARA.** The ARA address 0Ch is from the SMBus specification, not from the PPRs.
- **Field meanings by revision.** Control bit 7 and bit 1, and SB-TSI Config bits 3/1/0, mean
  different things on different revisions. The decoder uses the Revision value once it has seen it,
  and shows both names until then.
- **Test captures.** `examples/apml_*_synthetic_*.sr` and `test_apml.py` use waveforms generated
  bit by bit from the PPR tables ([gen_apml_capture.py](gen_apml_capture.py)). They have not been
  checked against a real AMD platform.

For probing: APML on SMBus runs at 100 kHz or 400 kHz ([56214] 5.5.2), so any SLogic sample rate of
10 MS/s or more is enough; APML-I3C runs at up to 12.5 MHz ([57238] 5.5.2), so use the I3C settings
above. The SIC/SID pins are not at SMBus voltage ([56214] 5.3.1.1), so probe on the BMC side of the
board's level translator and set the threshold for that rail.

## Known limits

- On the wire a Target's IBI and a Controller private read look the same (address header after S
  with RnW=1, ACK, data). `i3c_sdr` labels them "Private read or IBI"; `mctp_i3c` reports a
  single-byte 0xAE read as the MCTP Pending Read IBI, which is a heuristic.
- HDR modes are not decoded (only the HDR Exit Pattern is recognised).
- What has been verified: decoding of the simulated captures, checked byte-for-byte against what the
  behavioural Targets received ([test_decoders.py](test_decoders.py), run by `../run_sim.sh` when
  sigrok-cli is installed), and the timing check against an out-of-spec run (tLOW_OD = 150 ns).
  What has not: captures from a real SLogic32U3 or a real I3C Target.
