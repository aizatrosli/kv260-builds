"""Capture every VPU operation of one JEPA encoder inference as RTL test vectors.

  python3 vpu_vectors.py --qmodel qmodel.npy --img 0 --out build/vpu
Writes cmd.hex (64 bit/job), in.hex (256 bit/row: b<<128 | a), exp.hex (128 bit/row), meta.txt.
Job word: [63] lut_load, [62] lut_sel, [1:0] op, [13:8] rows, [35:18] ma, [53:36] mb.
A lut_load job is followed by 256 in.hex rows whose low 16 bits are the entries (no exp rows).
"""
import argparse, os
import numpy as np
import intops as I
import quantize as Q
from train import load_mnist

N = 16
OP_ADD, OP_LUT, OP_LN, OP_SM = 0, 1, 2, 3


class Rec:
    def __init__(self):
        self.cmd, self.inp, self.exp = [], [], []
        self.lut = {0: None, 1: None}
        self.count = {}

    @staticmethod
    def pack(v, bits):
        w = 0
        for i, x in enumerate(v):
            w |= (int(x) & ((1 << bits) - 1)) << (bits * i)
        return w

    def lut_load(self, sel, table):
        if self.lut[sel] is not None and np.array_equal(self.lut[sel], table):
            return
        self.lut[sel] = table.copy()
        self.cmd.append((1 << 63) | (sel << 62))
        self.inp += [int(v) & 0xFFFF for v in table]

    def job(self, op, a, out, b=None, ma=0, mb=0):
        a = np.asarray(a).reshape(-1); out = np.asarray(out).reshape(-1)
        rows = len(a) // N
        assert len(a) % N == 0 and 1 <= rows <= 32
        self.cmd.append(op | (rows << 8) | ((ma & 0x3FFFF) << 18) | ((mb & 0x3FFFF) << 36))
        b = np.zeros_like(a) if b is None else np.asarray(b).reshape(-1)
        for r in range(rows):
            sl = slice(N * r, N * r + N)
            self.inp.append(self.pack(a[sl], 8) | (self.pack(b[sl], 8) << 128))
            self.exp.append(self.pack(out[sl], 8))
        self.count[op] = self.count.get(op, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmodel", default="qmodel.npy")
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--out", default="build/vpu")
    a = ap.parse_args()
    P = np.load(a.qmodel, allow_pickle=True).item()
    _, _, xte, _ = load_mnist("/root/data/mnist.npz")
    R = Rec()
    ln, sm, l8, add = I.layernorm, I.softmax, I.lut8, I.ew_add

    def w_ln(x):
        y = ln(x)
        for t in range(x.shape[0]): R.job(OP_LN, x[t], y[t])
        return y

    def w_sm(s, lut, q=127):
        y = sm(s, lut, q)
        R.lut_load(0, lut)
        for t in range(s.shape[0]): R.job(OP_SM, s[t], y[t], ma=q)    # numerator rides on ma
        return y

    def w_l8(x, lut):
        y = l8(x, lut)
        R.lut_load(1, lut)                     # entry i = value for a = i - 128
        for t in range(x.shape[0]): R.job(OP_LUT, x[t], y[t])
        return y

    def w_add(x, b, ma, mb):
        x = np.broadcast_to(x, np.broadcast_shapes(np.shape(x), np.shape(b)))
        b = np.broadcast_to(b, x.shape)
        y = add(x, b, ma, mb)
        for t in range(x.shape[0]): R.job(OP_ADD, x[t], y[t], b[t], ma, mb)
        return y

    I.layernorm, I.softmax, I.lut8, I.ew_add = w_ln, w_sm, w_l8, w_add
    Q.int_forward(P, Q.quant_img(P, xte[a.img:a.img + 1])[0])
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/cmd.hex", "w") as f: f.write("".join(f"{w:016x}\n" for w in R.cmd))
    with open(f"{a.out}/in.hex", "w") as f: f.write("".join(f"{w:064x}\n" for w in R.inp))
    with open(f"{a.out}/exp.hex", "w") as f: f.write("".join(f"{w:032x}\n" for w in R.exp))
    with open(f"{a.out}/meta.txt", "w") as f: f.write(f"jobs {len(R.cmd)}\nin_rows {len(R.inp)}\nexp_rows {len(R.exp)}\n")
    names = {OP_ADD: "add", OP_LUT: "gelu", OP_LN: "layernorm", OP_SM: "softmax"}
    print(f"{len(R.cmd)} jobs ({', '.join(f'{names[k]} {v}' for k, v in sorted(R.count.items()))}), "
          f"{len(R.exp)} output rows -> {a.out}")


if __name__ == "__main__":
    main()
