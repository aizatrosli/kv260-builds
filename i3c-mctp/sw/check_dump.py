#!/usr/bin/env python3
"""Cross-check the simulator's bus dump against the independent Python MCTP model (mctp_i3c.py).

Dump format (written by tb/tb_i3c_mctp.sv):
  MSG <da> <dst> <src> <to> <tag> <seq> <tu> <message bytes...>   message the generator was asked to send
  WR <da> <bytes...>        one private write as received by the Target (bytes after the address)
  BADPEC <da> <bytes...>    a write sent with PEC corruption enabled
  RD <da> <bytes...>        one private read as received by the Controller
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mctp_i3c as m  # noqa: E402


def main(path: str) -> int:
    lines = [ln.split() for ln in open(path) if ln.strip()]
    errors, checks = 0, 0

    def chk(cond: bool, msg: str) -> None:
        nonlocal errors, checks
        checks += 1
        if not cond:
            errors += 1
            print("check_dump FAIL:", msg)

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln[0] == "MSG":
            da, dst, src = (int(x, 16) for x in ln[1:4])
            to, tag, seq, tu = (int(x) for x in ln[4:8])
            msg = bytes(int(x, 16) for x in ln[8:])
            exp = m.fragment(msg, addr=da, dst=dst, src=src, to=to, tag=tag, seq=seq, tu=tu)
            frames = []
            i += 1
            while i < len(lines) and lines[i][0] == "WR":
                chk(int(lines[i][1], 16) == da, "frame address")
                frames.append(bytes(int(x, 16) for x in lines[i][2:]))
                i += 1
            chk(len(frames) == len(exp), f"MSG to 0x{da:02x}: {len(frames)} frames, model expects {len(exp)}")
            for k, (f, e) in enumerate(zip(frames, exp)):
                chk(f == e.wire(), f"MSG to 0x{da:02x} packet {k}: bytes differ from model\n  got {f.hex()}\n  exp {e.wire().hex()}")
            pkts = []
            for f in frames:
                p, ok = m.parse(f, da, 0)
                chk(ok, f"PEC of frame {f.hex()}")
                pkts.append(p)
            try:
                chk(m.reassemble(pkts) == msg, "reassembled message equals the original")
            except AssertionError as ex:
                chk(False, f"reassembly rule: {ex}")
            print(f"check_dump: message to 0x{da:02x}, {len(msg)} B, {len(frames)} packet(s): checked")
            continue
        if ln[0] == "BADPEC":
            f = bytes(int(x, 16) for x in ln[2:])
            _, ok = m.parse(f, int(ln[1], 16), 0)
            chk(not ok, "corrupted PEC must fail the check")
            good = m.crc8_pec(bytes([int(ln[1], 16) << 1]) + f[:-1])
            chk(f[-1] == good ^ 0x01, "corruption flips PEC bit 0 only")
            print("check_dump: corrupted-PEC frame: rejected as expected")
        elif ln[0] == "RD":
            f = bytes(int(x, 16) for x in ln[2:])
            p, ok = m.parse(f, int(ln[1], 16), 1)
            chk(ok, "PEC of read packet (address byte with RnW=1 included, DSP0233 §5.3.1)")
            chk(p.som and p.eom and p.payload[0] == 0x00, "read packet is a single-packet control message")
            print(f"check_dump: read packet from 0x{int(ln[1], 16):02x}: PEC ok, Get EID response EID=0x{p.payload[4]:02x}")
        i += 1

    print(f"check_dump: {checks} checks, {errors} failures")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/sim_dump.txt"))
