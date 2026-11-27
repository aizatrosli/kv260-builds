"""Concatenate the encoder's static weights into ROM init files and print the on-chip memory budget.
  python3 export_rom.py --cases /root/tpuv1/build/jepa/cases --out rom/
weights.hex : weight-tile words in execution order (128 bit/line) -> rom_stream INIT_FILE
exp_l<L>.hex / gelu_l<L>.hex : VPU LUTs per layer;  requant params live in the program (SET_CONFIG)
"""
import argparse, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--cases"); ap.add_argument("--qmodel", default="qmodel.npy"); ap.add_argument("--out", default="rom")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
names = [l.strip().split("/")[-1] for l in open(f"{a.cases}/cases.txt") if l.strip()]
static = [n for n in names if not re.match(r"^(qk|pv)\d+h\d+$", n)]
words, instrs = [], 0
for n in static:
    words += open(f"{a.cases}/{n}/wmem.hex").read().split()
for n in names:
    instrs += int(dict(l.split() for l in open(f"{a.cases}/{n}/meta.txt"))["prog_instrs"])
open(f"{a.out}/weights.hex", "w").write("\n".join(words) + "\n")
P = np.load(a.qmodel, allow_pickle=True).item()
for l in range(P["cfg"]["depth"]):
    open(f"{a.out}/exp_l{l}.hex", "w").write("".join(f"{int(v):04x}\n" for v in P[f"exp{l}"]))
    open(f"{a.out}/gelu_l{l}.hex", "w").write("".join(f"{int(v) & 0xff:02x}\n" for v in P[f"gelu{l}"]))
pos = P["pos"].size
wb, pb = len(words) * 16, instrs * 16
print(f"weight ROM: {len(words)} words = {wb/1024:.0f} KiB; pos-embed {pos/1024:.0f} KiB; "
      f"program (one instr stream per layer, unrolled) {instrs} instrs = {pb/1024:.0f} KiB")
tot = wb + pos * 1 + pb
print(f"total hard-coded: {tot/1024:.0f} KiB = {tot*8/1e6:.1f} Mb  (XCK26 PL: 144 BRAM36 + 64 URAM = 23.6 Mb; tpuv1 already uses 44 BRAM36 + 32 URAM = 10.8 Mb)")
