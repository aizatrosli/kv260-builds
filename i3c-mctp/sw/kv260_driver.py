#!/usr/bin/env python3
"""PS-side driver for the I3C/MCTP generator on the KV260 (runs on the board's Linux, as root).

Register map and command encoding: rtl/i3c_mctp_axil.sv and rtl/i3c_sdr_engine.sv. The base address
is whatever you assign to the AXI slave in the Vivado address editor (0xA000_0000 is the first
M_AXI_HPM0_FPD address, pass --base if yours differs).

NOT yet run on hardware: the register sequences below are the ones tb/tb_axil.sv and
tb/tb_i3c_mctp.sv exercise in simulation.

Examples:
  sudo python3 kv260_driver.py daa 0x0A 0x0B
  sudo python3 kv260_driver.py getstatus 0x0A
  sudo python3 kv260_driver.py mctp-get-eid 0x0A
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mctp_i3c as m  # noqa: E402

# registers
ID, CTRL, STATUS, TIM0, TIM1, TIM2 = 0x00, 0x04, 0x08, 0x10, 0x14, 0x18
CMD, RSP, MCFG0, MCFG1, MCTRL, MSTAT, MSG = 0x20, 0x24, 0x30, 0x34, 0x38, 0x3C, 0x40
# engine ops / flags
OP_START, OP_STOP, OP_ADDR, OP_WRITE, OP_READ, OP_RAWWR, OP_RAWRD, OP_DAA, OP_WAIT, OP_PATTERN = range(1, 11)
F_ARB, F_IBIACK, F_NACKSTOP = 0x01, 0x02, 0x04
F_LAST = F_PARINV = F_OD = 0x01
RSP_ADDR, RSP_READ, RSP_RAWRD, RSP_DAA, RSP_IBI, RSP_ERR = 3, 5, 7, 8, 11, 15


def cmd(op: int, flags: int = 0, data: int = 0, nbits: int = 0) -> int:
    return (op << 28) | ((flags & 0xFF) << 20) | ((nbits & 0xF) << 16) | (data & 0xFFFF)


class Gen:
    def __init__(self, base: int = 0xA000_0000, size: int = 0x1000):
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        os.close(fd)
        ident = self.rd(ID)
        if ident != 0x13C3_0100:
            raise RuntimeError(f"unexpected ID 0x{ident:08x} at 0x{base:x}")

    def rd(self, off: int) -> int:
        return struct.unpack_from("<I", self.mm, off)[0]

    def wr(self, off: int, val: int) -> None:
        struct.pack_into("<I", self.mm, off, val & 0xFFFF_FFFF)

    # -- low level
    def push(self, *cmds: int) -> None:
        for c in cmds:
            while self.rd(STATUS) & 0x4:   # CMD FIFO full
                pass
            self.wr(CMD, c)

    def wait_free(self, timeout: float = 1.0) -> None:
        t0 = time.monotonic()
        time.sleep(0.0001)
        while (self.rd(STATUS) & 0x3) != 0x1:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("bus not free")

    def responses(self) -> list[int]:
        out = []
        while not self.rd(STATUS) & 0x8:
            out.append(self.rd(RSP))
        return out

    # -- I3C
    def entdaa(self, addrs: list[int]) -> list[dict]:
        """ENTDAA (I3C Basic §5.1.4.2): one OP_DAA round per address, plus one to see the final NACK."""
        self.push(cmd(OP_START), cmd(OP_ADDR, F_ARB, (m.I3C_BCAST << 1)), cmd(OP_WRITE, 0, m.CCC["ENTDAA"]))
        for a in addrs:
            self.push(cmd(OP_DAA, 0, a))
        self.push(cmd(OP_DAA, 0, 0x7F), cmd(OP_STOP))   # extra round: expect no ACK on 7'h7E/R
        self.wait_free()
        res, raw, cur = [], self.responses(), []
        for r in raw[1:]:
            kind = r >> 28
            if kind == RSP_RAWRD:
                cur.append(r & 0xFF)
            elif kind == RSP_DAA:
                if r & (1 << 20):
                    w = int.from_bytes(bytes(cur), "big")
                    res.append({"pid": w >> 16, "bcr": (w >> 8) & 0xFF, "dcr": w & 0xFF,
                                "da": r & 0x7F, "acked": bool(r & (1 << 21))})
                cur = []
        return res

    def ccc_broadcast(self, code: int, data: bytes = b"") -> bool:
        self.push(cmd(OP_START), cmd(OP_ADDR, F_ARB | F_NACKSTOP, m.I3C_BCAST << 1), cmd(OP_WRITE, 0, code))
        self.push(*(cmd(OP_WRITE, 0, b) for b in data), cmd(OP_STOP))
        self.wait_free()
        rs = self.responses()
        return bool(rs and rs[0] & (1 << 20))

    def ccc_direct_get(self, code: int, addr: int, n: int) -> bytes | None:
        """Direct GET CCC (§5.1.9.2): 7E/W, CCC, Sr, addr/R, n bytes (T-bit End-of-Data)."""
        self.push(cmd(OP_START), cmd(OP_ADDR, F_ARB | F_NACKSTOP, m.I3C_BCAST << 1), cmd(OP_WRITE, 0, code),
                  cmd(OP_START), cmd(OP_ADDR, F_NACKSTOP, (addr << 1) | 1))
        self.push(*(cmd(OP_READ, F_LAST if i == n - 1 else 0) for i in range(n)), cmd(OP_STOP))
        self.wait_free()
        rs = self.responses()
        if len(rs) < 2 or not rs[1] & (1 << 20):
            return None
        return bytes(r & 0xFF for r in rs[2:] if r >> 28 == RSP_READ)

    # -- MCTP
    def send_mctp(self, msg: bytes, *, da: int, dst: int, src: int, to: int = 1, tag: int = 0,
                  seq: int = 0, tu: int = m.BASELINE_TU, use_7e: bool = True, corrupt_pec: bool = False) -> tuple[int, int]:
        for b in msg:
            self.wr(MSG, b)
        self.wr(MCFG1, (len(msg) << 16) | (src << 8) | dst)
        self.wr(MCFG0, (tu << 24) | (int(use_7e) << 21) | (int(corrupt_pec) << 20) | (m.HDR_VERSION << 16)
                | ((seq & 3) << 12) | ((tag & 7) << 9) | ((to & 1) << 8) | (da & 0x7F))
        self.wr(MCTRL, 1)
        self.wait_free()
        st = self.rd(MSTAT)
        self.responses()
        return st & 0xFFFF, st >> 16          # packets ACKed, NACKed

    def read_mctp_packet(self, da: int, max_bytes: int = 69) -> tuple[m.Packet, bool] | None:
        """Private read of one packet (DSP0233 §5.2.2.1 form (b)); stops at T=0 End-of-Data."""
        self.push(cmd(OP_START), cmd(OP_ADDR, F_ARB | F_NACKSTOP, (da << 1) | 1))
        self.push(*(cmd(OP_READ, F_LAST if i == max_bytes - 1 else 0) for i in range(max_bytes)))
        self.push(cmd(OP_STOP))
        self.wait_free()
        data = []
        for r in self.responses():
            if r >> 28 == RSP_READ:
                data.append(r & 0xFF)
                if not r & (1 << 20):     # T=0: Target ended; later READs were flushed by the engine
                    break
        return m.parse(bytes(data), da, 1) if len(data) >= 5 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0xA000_0000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("daa"); p.add_argument("addrs", nargs="+", type=lambda s: int(s, 0))
    p = sub.add_parser("getstatus"); p.add_argument("addr", type=lambda s: int(s, 0))
    p = sub.add_parser("mctp-get-eid"); p.add_argument("addr", type=lambda s: int(s, 0))
    a = ap.parse_args()
    g = Gen(a.base)
    if a.cmd == "daa":
        for t in g.entdaa(a.addrs):
            print(f"PID 0x{t['pid']:012x} BCR 0x{t['bcr']:02x} DCR 0x{t['dcr']:02x} -> DA 0x{t['da']:02x} {'ACK' if t['acked'] else 'NACK'}")
    elif a.cmd == "getstatus":
        r = g.ccc_direct_get(m.CCC["GETSTATUS"], a.addr, 2)
        print("NACK" if r is None else f"GETSTATUS 0x{int.from_bytes(r, 'big'):04x}")
    elif a.cmd == "mctp-get-eid":
        print("sent (ACK, NACK) =", g.send_mctp(m.get_eid_request(), da=a.addr, dst=m.EID_NULL, src=0x08))
    return 0


if __name__ == "__main__":
    sys.exit(main())
