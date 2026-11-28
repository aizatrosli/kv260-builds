#!/usr/bin/env python3
"""Check the 'amd_apml' decoder on the synthetic SMBus and I3C captures from gen_apml_capture.py.

  python3 sigrok/test_apml.py [build_dir]      (SIGROK_CLI=/path overrides sigrok-cli)

Every expected line below is derived by hand from the AMD PPR tables cited in
decoders/amd_apml/pd.py, independently of the decoder code.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "build")
CLI = os.environ.get("SIGROK_CLI", "sigrok-cli")
ENV = dict(os.environ, SIGROKDECODE_DIR=os.path.join(HERE, "decoders"))
errors = 0

subprocess.run([sys.executable, os.path.join(HERE, "gen_apml_capture.py"), B], check=True)


def run(vcd, ds, stack, ann="amd_apml"):
    cmd = [CLI, "-I", "vcd:downsample=%d" % ds, "-i", vcd, "-P", stack, "-A", ann]
    return subprocess.run(cmd, env=ENV, capture_output=True, text=True, check=True).stdout.splitlines()


def expect(lines, needle, what, count=1):
    global errors
    n = sum(needle in ln for ln in lines)
    ok = n == count
    print(("ok: " if ok else "FAIL: ") + "%s (%d x %r)" % (what, n, needle))
    errors += not ok


SMBUS = [
    ("SB-RMI 0x3C read Revision (0x00) = 0x10 [SB-RMI revision 0x10", "RMI Revision 10h (read byte with Sr)"),
    ("CPU temperature = 60.125 degC (0x3C/0x20)", "Tctl 60.125 from CpuTempInt/CpuTempDec (Table 93)"),
    ("SB-TSI 0x4C write HiTempInt (0x07) = 0x5A: 90 degC", "high threshold write"),
    ("SB-TSI 0x4C write ConfigWr (0x09) = 0x80: AlertMask", "ConfigWr AlertMask"),
    ("SB-TSI 0x4C read UpdateRate (0x04) = 0x08: 16 Hz", "UpdateRate 08h = 16 Hz"),
    ("Temperature offset = -10.375 degC (0xF5/0xA0)", "offset -10.375 (Table 94)"),
    ("read CpuTempInt (0x01) (Receive Byte, address pointer) = 0x3D", "Send Byte + Receive Byte (6.2.1)"),
    ("PEC 0x1B ok", "PEC on Write Byte"),
    ("SB-RMI 0x3C write Control (0x01) = 0x08 [BlkRWEn]", "Control BlkRWEn"),
    ("mailbox request 0x05 Read TDP, DataIn 0x00000000", "mailbox request (5.4.2.2.1)"),
    ("mailbox response 0x05 Read TDP: DataOut 0x0001D4C0 (120000 mW), error 00h No error", "mailbox response"),
    ("Read Register MSR 0xC0002001, thread 3, 8 byte(s)", "Read Processor Register request (Table 80)"),
    ("Intermediate PEC 0x7E ok", "intermediate PEC (5.4.1)"),
    ("Read MSR result: status 00h Success, data 0x1122334455667788", "Read Processor Register response"),
    ("Read CPUID fn 0x00000001 ECX 0 -> ebx:eax, thread 0", "Read CPUID request (Table 81)"),
    ("Read CPUID result: status 00h Success, eax=0x00A20F10 ebx=0x00100800", "Read CPUID response"),
    ("PEC 0x22 BAD (expect 0x23)", "corrupted PEC flagged"),
    ("SMBus Alert Response: alert from 0x4C", "ARA read"),
]
I3C = [
    ("SETDASA: static 0x3C -> dynamic 0x30", "SETDASA maps SB-RMI"),
    ("SB-TSI 0x4C read Config (0x03) = 0x08: SBRMIAddrMode", "Config SBRMIAddrMode (57238 6.4)"),
    ("SB-RMI 0x3C read Revision (0x00) = 0x21 [SB-RMI revision 0x21 (APML-I3C 2.1, 2-byte command)]",
     "Revision 21h with 2-byte command"),
    ("CPU temperature = 72.875 degC (0x48/0xE0)", "Tctl 72.875"),
    ("Read CPUID fn 0x80000008 ECX 0 -> edx:ecx, thread 200", "Read CPUID write phase, thread[14:7] (Table 131)"),
    ("SB-RMI 0x3C read Status (0x02) = 0x80 [HwAlertSts]", "HwAlertSts after 73h"),
    ("Read CPUID result: status 00h Success, ecx=0x12345678 edx=0x9ABCDEF0", "Read CPUID read phase (Table 133)"),
    ("Read Register MSR 0xC0002001, thread 5, 8 byte(s)", "Read MSR write phase (Table 128)"),
    ("Read MSR result: status 44h Invalid thread", "status 44h (Table 134)"),
    ("mailbox request 0x01 Read Package Power Consumption, DataIn 0x00000000", "mailbox request (5.4.2.2.2)"),
    ("mailbox response 0x01 Read Package Power Consumption: DataOut 0x0003E418 (255000 mW), error 00h",
     "mailbox response"),
    ("SB-RMI 0x3C block write AlertStatus (0x18) = 0x00", "block write (Table 146)"),
    ("SB-RMI 0x3C block read AlertStatus (0x10) = 0x01", "block read (Table 147)"),
    ("SB-TSI 0x4C read Status (0x02) = 0x10: TempHighAlert", "TSI Status"),
]

for ds in (100, 20):
    print("== SMBus capture, %d MS/s" % (1000 // ds))
    vcd = os.path.join(B, "apml_smbus.vcd")
    out = run(vcd, ds, "i2c,amd_apml")
    for needle, what in SMBUS:
        expect(out, needle, what)
    warn = run(vcd, ds, "i2c,amd_apml", "amd_apml=warning")
    expect(warn, "", "only the injected bad PEC is a warning (%s)" % warn, count=1)
for ds in (5, 10):
    print("== I3C capture, %d MS/s" % (1000 // ds))
    vcd = os.path.join(B, "apml_i3c.vcd")
    out = run(vcd, ds, "i3c_sdr,amd_apml")
    for needle, what in I3C:
        expect(out, needle, what)
    warn = run(vcd, ds, "i3c_sdr,amd_apml", "amd_apml=warning")
    expect(warn, "", "only the 44h status is a warning (%s)" % warn, count=1)
    low = run(vcd, ds, "i3c_sdr", "i3c_sdr=warning:timing")
    expect(low, "", "I3C layer: no parity/timing warnings on the synthetic capture", count=0)

print("test_apml: %s" % ("ALL PASS" if errors == 0 else "%d FAILURES" % errors))
sys.exit(1 if errors else 0)
