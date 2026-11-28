# sigrok / PulseView decoders: I3C SDR -> MCTP over I3C -> MCTP

Three stacked libsigrokdecode (API 3) decoders that turn an SCL/SDA capture into I3C frames, MCTP
packets and reassembled MCTP messages, with the same spec references as the generator
(see [../docs/README.md](../docs/README.md) §1).

| Decoder id  | Input -> output        | What it shows |
|-------------|------------------------|---------------|
| `i3c_sdr`   | logic -> `i3c`         | S / Sr / P, arbitrable vs push-pull address header, ACK/NACK, data bytes with the T-bit (odd parity on writes, End-of-Data on reads), broadcast and direct CCCs by name (I3C Basic v1.1.1 Table 16), ENTDAA rounds (PID / BCR / DCR / assigned DA, DA parity), HDR Exit and Target Reset patterns, optional SDR timing check (tLOW_OD >= 200 ns, push-pull tLOW >= 24 ns, Table 86/87). |
| `mctp_i3c`  | `i3c` -> `mctp_pkt`    | One MCTP packet per private write/read (DSP0233 v1.0.1): header version, dest/src EID, SOM/EOM/Seq/TO/Tag, IC + message type, CRC-8 PEC over the address byte with RnW and the packet (§5.3.1), and a one-byte 0xAE read after START as the MCTP Pending Read IBI (Table 2). Option `addrs` restricts it to some Target addresses. |
| `mctp`      | `mctp_pkt` -> -        | Message reassembly by (tag, TO, source EID) with sequence and transmission-unit checks (DSP0236 v1.4.0 §8.4.2, §8.6, §8.9), message type names (DSP0239 v1.12.0 Table 1), MCTP Control Rq/D/IID, command (Table 12), completion code (Table 13), and the payloads of Set/Get Endpoint ID, Get MCTP Version Support and Get Message Type Support. |

The id is `i3c_sdr`, not `i3c`, so it can sit next to other community I3C decoders.
Mainline libsigrokdecode has no I3C decoder.

## Install

Copy the three folders in [decoders/](decoders) (`i3c_sdr`, `mctp_i3c`, `mctp`) into a decoder
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

## Known limits

- On the wire a Target's IBI and a Controller private read look the same (address header after S
  with RnW=1, ACK, data). `i3c_sdr` labels them "Private read or IBI"; `mctp_i3c` reports a
  single-byte 0xAE read as the MCTP Pending Read IBI, which is a heuristic.
- HDR modes are not decoded (only the HDR Exit Pattern is recognised).
- What has been verified: decoding of the simulated captures, checked byte-for-byte against what the
  behavioural Targets received ([test_decoders.py](test_decoders.py), run by `../run_sim.sh` when
  sigrok-cli is installed), and the timing check against an out-of-spec run (tLOW_OD = 150 ns).
  What has not: captures from a real SLogic32U3 or a real I3C Target.
