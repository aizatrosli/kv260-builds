#!/usr/bin/env python3
"""Run the sigrok decoder stack on the simulated bus capture and check what it decodes.

  python3 sigrok/test_decoders.py [build_dir]

Needs sigrok-cli (SIGROK_CLI=/path overrides) and the Verilator captures written by run_sim.sh:
  build/i3c_bus.vcd       (generator traffic, all scenarios of tb_i3c_mctp)
  build/sim_dump.txt      (bytes the behavioural Targets actually received / sent)
  build/mut/i3c_bus.vcd   (same run with tLOW_OD = 150 ns, out of spec)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "build")
CLI = os.environ.get("SIGROK_CLI", "sigrok-cli")
ENV = dict(os.environ, SIGROKDECODE_DIR=os.path.join(HERE, "decoders"))
errors = 0


def run(vcd, ds, stack, ann):
    cmd = [CLI, "-I", "vcd:downsample=%d" % ds, "-i", vcd, "-P", stack, "-A", ann]
    out = subprocess.run(cmd, env=ENV, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def check(cond, msg):
    global errors
    print(("ok: " if cond else "FAIL: ") + msg)
    if not cond:
        errors += 1


def expect(lines, needle, what=None, count=None):
    n = sum(needle in ln for ln in lines)
    check(n >= 1 if count is None else n == count,
          "%s (%d match%s for %r)" % (what or needle, n, "" if n == 1 else "es", needle))


vcd = os.path.join(B, "i3c_bus.vcd")
for ds, rate in ((5, "200 MS/s"), (10, "100 MS/s")):
    print("== capture resampled to %s" % rate)
    i3c = run(vcd, ds, "i3c_sdr", "i3c_sdr=xfer:daa:pattern:warning:timing")
    expect(i3c, "ENTDAA: PID 0x04D200001001 BCR 0x06 DCR 0xCC -> DA 0x0A ACK", "ENTDAA round 1")
    expect(i3c, "ENTDAA: PID 0x04D200002002 BCR 0x06 DCR 0xCC -> DA 0x0B ACK", "ENTDAA round 2")
    expect(i3c, "ENTDAA: no Target left", "ENTDAA end")
    expect(i3c, "Broadcast SETMWL: 00 45", "SETMWL 69")
    expect(i3c, "GETSTATUS -> 0x0B R: 00 07", "direct GETSTATUS reply")
    expect(i3c, "T-bit parity error on 0x55", "injected parity error", count=1)
    expect(i3c, "Private write 0x21 (NACK)", "NACK of absent address")
    expect(i3c, "Target Reset Pattern", "Target Reset Pattern", count=1)
    expect(i3c, "HDR Exit Pattern", "HDR Exit Pattern", count=1)
    expect(i3c, "Private write 0x0A: 33", "frame aborted by the stall limit ends after 0x33")
    check(not any("tLOW" in ln for ln in i3c), "no timing warnings on the in-spec capture")

    # every private write the decoder saw must equal what the Target model received
    dec = [ln.split(": ", 2)[2] for ln in i3c if re.search(r"Private write 0x0A: ", ln)]
    dump = [ln.split()[2:] for ln in open(os.path.join(B, "sim_dump.txt")) if ln.startswith(("WR 0a", "BADPEC 0a"))]
    dump = [" ".join(x.upper() for x in d) for d in dump]
    check(all(d in dec for d in dump), "all %d MCTP frames to 0x0A decoded byte-exact" % len(dump))

    full = run(vcd, ds, "i3c_sdr,mctp_i3c,mctp", "mctp_i3c=packet:ibi:warning,mctp=message:control:warning")
    expect(full, "MCTP pkt to 0x0A: EID 0x08->0x1D SOM seq 2 tag 5 TO, 64 B payload", "packet 1 of 3")
    expect(full, "MCTP pkt to 0x0A: EID 0x08->0x1D EOM seq 0 tag 5 TO, 22 B payload", "packet 3 of 3")
    expect(full, "MCTP Vendor Defined - IANA, EID 0x08->0x1D, tag 5 TO, 150 byte(s) in 3 packet(s)", "reassembly")
    expect(full, "Control Request: Set Endpoint ID (IID 1): Set EID 0x1D", "Set EID request")
    expect(full, "Control Request: Get Endpoint ID (IID 2)", "Get EID request")
    expect(full, "Control Response: Get Endpoint ID (IID 2): SUCCESS, EID 0x1E, simple endpoint", "Get EID response")
    expect(full, "PEC mismatch: got 0x18, computed 0x19", "injected PEC error", count=1)
    expect(full, "IBI from 0x0B: MDB 0xAE = MCTP Pending Read", "Pending Read IBI from 0x0B")
    expect(full, "IBI from 0x0A: MDB 0xAE = MCTP Pending Read", "Pending Read IBI from 0x0A (won arbitration)")
    check(sum("PEC BAD" in ln for ln in full) == 1, "exactly one packet with a bad PEC")

mut = os.path.join(B, "mut", "i3c_bus.vcd")
if os.path.exists(mut):
    print("== out-of-spec capture (tLOW_OD = 150 ns)")
    i3c = run(mut, 5, "i3c_sdr", "i3c_sdr=timing")
    expect(i3c, "tLOW_OD 150 ns < 200 ns (Table 86)", "timing check flags tLOW_OD")

print("test_decoders: %s" % ("ALL PASS" if errors == 0 else "%d FAILURES" % errors))
sys.exit(1 if errors else 0)
