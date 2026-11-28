#!/usr/bin/env python3
"""Write synthetic AMD APML bus captures (VCD, SCL/SDA) for testing the 'amd_apml' decoder.

  python3 sigrok/gen_apml_capture.py [out_dir]     -> apml_smbus.vcd, apml_i3c.vcd

The byte sequences follow the protocol tables of the AMD PPRs (see decoders/amd_apml/pd.py):
  apml_smbus.vcd  SMBus 100 kHz, SB-RMI rev 10h (1-byte command), PEC on some transfers  [56214]
  apml_i3c.vcd    I3C SDR 12.5 MHz push-pull, SB-RMI rev 21h (2-byte command)            [57238]
The bus levels are produced bit by bit here; they are not a capture of real hardware.
"""
import os
import sys


def crc8(data, crc=0):
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Wave:
    def __init__(self):
        self.t = 0
        self.scl = self.sda = 1
        self.ev = [(0, 1, 1)]

    def set(self, scl=None, sda=None, dt=0):
        if scl is not None:
            self.scl = scl
        if sda is not None:
            self.sda = sda
        self.ev.append((self.t, self.scl, self.sda))
        self.t += dt

    def idle(self, dt):
        self.set(dt=dt)

    def vcd(self, path):
        with open(path, 'w') as f:
            f.write('$timescale 1 ns $end\n$scope module apml $end\n$var wire 1 ! SCL $end\n'
                    '$var wire 1 " SDA $end\n$upscope $end\n$enddefinitions $end\n')
            last = None
            for t, c, d in self.ev:
                if last is not None and (c, d) == last[1:]:
                    continue
                f.write('#%d\n%d!\n%d"\n' % (t, c, d))
                last = (t, c, d)
            f.write('#%d\n' % (self.t + 1000))


# ------------------------------------------------------------------ SMBus / I2C (100 kHz)
class SMBus(Wave):
    TL, TH = 5000, 5000     # ns: 100 kHz

    def start(self):
        self.set(sda=1, dt=1000)
        self.set(scl=1, dt=self.TH)
        self.set(sda=0, dt=self.TH)
        self.set(scl=0, dt=self.TL // 2)

    def bit(self, v):
        self.set(sda=v, dt=self.TL // 2)
        self.set(scl=1, dt=self.TH)
        self.set(scl=0, dt=self.TL // 2)

    def byte(self, b, ack=0):
        for i in range(7, -1, -1):
            self.bit(b >> i & 1)
        self.bit(ack)

    def stop(self):
        self.set(sda=0, dt=self.TL // 2)
        self.set(scl=1, dt=self.TH)
        self.set(sda=1, dt=20000)

    def txn(self, addr, wr=(), rd=(), pec=False, sr=True, nack_addr=False):
        """START addr/W wr... [Sr addr/R rd...] STOP. With pec=True a PEC byte ends the write (no read)
        or the read, computed over every byte of the transaction (SMBus)."""
        raw = []
        self.start()
        if wr or not rd:
            raw.append(addr << 1)
            self.byte(addr << 1, ack=1 if nack_addr else 0)
            if nack_addr:
                self.stop()
                return
            for b in wr:
                raw.append(b)
                self.byte(b)
            if pec and not rd:
                self.byte(crc8(raw))
            if rd:
                self.start()            # repeated START (SCL is low here, SDA released first)
        raw.append((addr << 1) | 1)
        if rd:
            self.byte((addr << 1) | 1)
            rd = list(rd)
            if pec:
                rd = rd + [crc8(raw + rd)]
            for i, b in enumerate(rd):
                self.byte(b, ack=1 if i == len(rd) - 1 else 0)
        self.stop()


# ------------------------------------------------------------------ I3C SDR (MIPI I3C Basic)
class I3C(Wave):
    OD_L, OD_H = 250, 250       # open-drain header: tLOW_OD >= 200 ns (Table 86)
    PP_L, PP_H = 40, 40         # push-pull 12.5 MHz

    def start(self, repeated=False):
        if repeated:
            self.set(sda=1, dt=self.PP_L)
            self.set(scl=1, dt=self.OD_H)
        else:
            self.set(dt=1000)
        self.set(sda=0, dt=50)  # tCAS / tCASr
        self.set(scl=0, dt=self.OD_L // 2)

    def bit(self, v, od):
        tl, th = (self.OD_L, self.OD_H) if od else (self.PP_L, self.PP_H)
        self.set(sda=v, dt=tl // 2)
        self.set(scl=1, dt=th)
        self.set(scl=0, dt=tl - tl // 2)

    def header(self, addr, rnw, ack=True):
        v = (addr << 1) | rnw
        for i in range(7, -1, -1):
            self.bit(v >> i & 1, od=True)
        self.bit(0 if ack else 1, od=True)

    def wbyte(self, b, par_ok=True):
        for i in range(7, -1, -1):
            self.bit(b >> i & 1, od=False)
        t = 1 ^ (bin(b).count('1') & 1)         # odd parity (I3C §5.1.2.3.3)
        self.bit(t if par_ok else t ^ 1, od=False)

    def rbyte(self, b, last):
        for i in range(7, -1, -1):
            self.bit(b >> i & 1, od=False)
        self.bit(0 if last else 1, od=False)    # T-bit: 0 = End-of-Data

    def stop(self):
        self.set(sda=0, dt=self.PP_L)
        self.set(scl=1, dt=self.PP_H * 2)
        self.set(sda=1, dt=3000)

    def private(self, addr, wr=(), rd=()):
        """S addr/W wr... [Sr addr/R rd... (T=0 on the last)] P"""
        self.start()
        if wr or not rd:
            self.header(addr, 0)
            for b in wr:
                self.wbyte(b)
            if rd:
                self.start(repeated=True)
        if rd:
            self.header(addr, 1)
            for i, b in enumerate(rd):
                self.rbyte(b, i == len(rd) - 1)
        self.stop()

    def setdasa(self, static, dyn):
        self.start()
        self.header(0x7E, 0)
        self.wbyte(0x87)                        # SETDASA (direct)
        self.start(repeated=True)
        self.header(static, 0)
        self.wbyte(dyn << 1)
        self.stop()


def le4(v):
    return [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF]


def smbus_capture(path):
    w = SMBus()
    w.idle(5000)
    RMI, TSI = 0x3C, 0x4C
    w.txn(RMI, [0x00], [0x10])                              # Revision = 10h
    w.txn(TSI, [0x01], [0x3C])                              # CpuTempInt = 60
    w.txn(TSI, [0x10], [0x20])                              # CpuTempDec -> 60.125 degC
    w.txn(TSI, [0x07, 0x5A])                                # HiTempInt = 90
    w.txn(TSI, [0x09, 0x80])                                # ConfigWr: AlertMask
    w.txn(TSI, [0x04], [0x08])                              # UpdateRate 16 Hz
    w.txn(TSI, [0x11, 0xF5])                                # offset -10.375 ([56214] Table 94)
    w.txn(TSI, [0x12, 0xA0])
    w.txn(TSI, [0x01])                                      # Send Byte (address pointer)
    w.txn(TSI, (), [0x3D])                                  # Receive Byte -> CpuTempInt 61
    w.txn(RMI, [0x01, 0x08], pec=True)                      # Control = BlkRWEn, with PEC
    # mailbox: Read TDP ([56214] Table 86, sequence 5.4.2.2.1)
    w.txn(RMI, [0x3F, 0x80])
    w.txn(RMI, [0x38, 0x05])
    for i in range(4):
        w.txn(RMI, [0x39 + i, 0x00])
    w.txn(RMI, [0x40, 0x01])
    w.txn(RMI, [0x02], [0x02])                              # Status: SwAlertSts
    w.txn(RMI, [0x37], [0x00])                              # error code
    for i, b in enumerate(le4(120000)):
        w.txn(RMI, [0x31 + i], [b])
    w.txn(RMI, [0x02, 0x02])                                # clear SwAlertSts
    # Read MSR 0xC0002001 thread 3, 8 bytes, PEC on both phases ([56214] Table 80)
    wr = [0x73, 0x07, 0x08, 0x86, 3 << 1] + le4(0xC0002001)
    pec1 = crc8([RMI << 1] + wr)
    rd = [0x09, 0x00] + list((0x1122334455667788).to_bytes(8, 'little'))
    pec2 = crc8([RMI << 1] + wr + [(RMI << 1) | 1] + rd)
    w.txn(RMI, wr + [pec1], rd + [pec2])
    # Read CPUID fn 0x00000001 ebx:eax, thread 0, no PEC ([56214] Table 81)
    w.txn(RMI, [0x73, 0x08, 0x08, 0x91, 0x00] + le4(1) + [0x00],
          [0x09, 0x00] + le4(0x00A20F10) + le4(0x00100800))
    # bad PEC on a register write
    w.start(); w.byte(RMI << 1); w.byte(0x01); w.byte(0x00); w.byte(crc8([RMI << 1, 0x01, 0x00]) ^ 0x01); w.stop()
    w.txn(0x0C, (), [TSI << 1])                             # SMBus ARA
    w.txn(0x4E, [0x01], nack_addr=True)                     # nobody at 0x4E
    w.vcd(path)


def i3c_capture(path):
    w = I3C()
    w.idle(2000)
    RMI_DA, TSI_DA = 0x30, 0x31
    w.setdasa(0x3C, RMI_DA)
    w.setdasa(0x4C, TSI_DA)
    w.private(TSI_DA, [0x03], [0x08])                       # Config: SBRMIAddrMode=1 ([57238] 6.4)
    w.private(RMI_DA, [0x00, 0x00], [0x21])                 # Revision 21h, 2-byte command
    w.private(TSI_DA, [0x01], [0x48])                       # 72 degC
    w.private(TSI_DA, [0x10], [0xE0])                       # .875
    # Read CPUID fn 0x80000008 edx:ecx thread 200 ([57238] Tables 131-133)
    thr = 200
    w.private(RMI_DA, [0x73, 0x00, 0x09, 0x08, 0x91, (thr & 0x7F) << 1, thr >> 7] + le4(0x80000008) + [0x01])
    w.private(RMI_DA, [0x02, 0x00], [0x80])                 # Status: HwAlertSts
    w.private(RMI_DA, [0x73, 0x00], [0x09, 0x00] + le4(0x12345678) + le4(0x9ABCDEF0))
    w.private(RMI_DA, [0x02, 0x00, 0x80])                   # clear HwAlertSts
    # Read MSR 0xC0002001 thread 5, invalid thread status 44h ([57238] Tables 128-130, 134)
    w.private(RMI_DA, [0x73, 0x00, 0x08, 0x08, 0x86, 5 << 1, 0x00] + le4(0xC0002001))
    w.private(RMI_DA, [0x73, 0x00], [0x09, 0x44] + [0] * 8)
    # mailbox: Read Package Power Consumption ([57238] Table 135, 5.4.2.2.2)
    w.private(RMI_DA, [0x3F, 0x00, 0x80])
    w.private(RMI_DA, [0x38, 0x00, 0x01])
    for i in range(4):
        w.private(RMI_DA, [0x39 + i, 0x00, 0x00])
    w.private(RMI_DA, [0x40, 0x00, 0x01])
    w.private(RMI_DA, [0x37, 0x00], [0x00])
    for i, b in enumerate(le4(255000)):
        w.private(RMI_DA, [0x31 + i, 0x00], [b])
    w.private(RMI_DA, [0x02, 0x00, 0x02])
    # block write from 0x18 ([57238] Table 146) and block read from 0x10 (Table 147)
    w.private(RMI_DA, [0x18, 0x00, 0x08] + [0] * 8)
    w.private(RMI_DA, [0x10, 0x00], [0x08, 1, 0, 0, 0, 0, 0, 0, 0])
    w.private(TSI_DA, [0x02], [0x10])                       # Status: TempHighAlert
    w.vcd(path)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    os.makedirs(out, exist_ok=True)
    smbus_capture(os.path.join(out, 'apml_smbus.vcd'))
    i3c_capture(os.path.join(out, 'apml_i3c.vcd'))
    print('wrote apml_smbus.vcd, apml_i3c.vcd to', out)
