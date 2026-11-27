"""Compile the quantised JEPA encoder into ONE program for the hard-coded TPU (jepa_top).

  python3 compile_jepa.py --qmodel qmodel.npy --out rom_e2e/ [--img 7]

Outputs (all $readmemh hex):
  prog.hex    program ROM: tpuv1 ISA + sequencer extension ops (one 128-bit word per line)
  wrom.hex    weight/data ROM (128 bit): position embedding, then weight tiles in program order
  prom.hex    requant-parameter ROM (72 bit): 17 words per entry (16 channels + zp/min/max)
  lrom.hex    LUT ROM (16 bit): exp tables (index l) and GELU tables (index depth + l), 256 each
  ub_init.hex input image rows (the only data the host provides), exp_out.hex expected output,
  meta.txt
The program is executed here by a Python machine model (tpuv1's ISA simulator for core
instructions + intops for the extension) and must equal quantize.int_forward bit for bit.

Sequencer extension (opcodes >= 0x40, executed in front of the unmodified tpuv1 core):
  X_WROM  0x40 count            stream the next `count` weight-ROM words into the Weight FIFO
  X_WUB   0x41 tr base stride kt nt   UB activations -> Weight FIFO as kt x nt tiles (tr: transpose)
  X_LOADQ 0x42 q entry          expands to the 33 SET_CONFIGs of requant set q from the param ROM
  X_VEC   0x43 op rows a b o count rstride vstride   `count` VPU jobs on UB vectors
  X_LUT   0x44 sel idx          load VPU LUT `sel` (0 exp, 1 gelu) from LUT ROM table idx
  X_VCFG  0x45 ma mb            residual-add multipliers for later X_VEC ADD
  X_UBROM 0x46 base count       copy the next `count` weight-ROM words into UB rows base..
  X_MM    0x47 q sync kt in acc out entry B   one output tile: [SYNC act] X_LOADQ, kt x MATMUL,
               (order: MATMULs, LOADQ, SYNC matrix, ACTIVATE) (the macro form of what --no-macro emits instruction by instruction)
"""
import argparse, os, sys
import numpy as np

TPU = os.environ.get("TPU", "/root/tpuv1")
sys.path.insert(0, f"{TPU}/model/isa"); sys.path.insert(0, f"{TPU}/model/quantization")
import asm                    # noqa: E402
import sim as isasim          # noqa: E402
import intops as I            # noqa: E402
from quantize import int_forward, quant_img   # noqa: E402

N = 16
X_WROM, X_WUB, X_LOADQ, X_VEC, X_LUT, X_VCFG, X_UBROM, X_MM = range(0x40, 0x48)
XF = {
    X_WROM:  {"count": (8, 24)},
    X_WUB:   {"tr": (8, 1), "base": (16, 16), "stride": (32, 16), "kt": (48, 8), "nt": (56, 8)},
    X_LOADQ: {"q": (8, 2), "entry": (16, 16)},
    X_VEC:   {"op": (8, 2), "rows": (10, 6), "a": (16, 16), "b": (32, 16), "o": (48, 16),
              "count": (64, 16), "rstride": (80, 16), "vstride": (96, 8)},
    X_LUT:   {"sel": (8, 1), "idx": (16, 8)},
    X_VCFG:  {"ma": (8, 18), "mb": (32, 18)},
    X_UBROM: {"base": (16, 16), "count": (32, 16)},
    X_MM:    {"q": (8, 2), "sync": (10, 1), "kt": (11, 6), "in_base": (17, 16), "acc_row": (33, 11),
              "out_row": (44, 16), "entry": (60, 16), "B": (76, 16)},
}
XNAMES = {X_WROM: "X_WROM", X_WUB: "X_WUB", X_LOADQ: "X_LOADQ", X_VEC: "X_VEC", X_LUT: "X_LUT",
          X_VCFG: "X_VCFG", X_UBROM: "X_UBROM", X_MM: "X_MM"}
V_ADD, V_LUT, V_LN, V_SM = 0, 1, 2, 3
ACC_HALF = 1024
NREG = 4            # ACC regions (B rows each) rotated with the 4 requant sets


def xenc(opc, **kw):
    w = opc
    for k, v in kw.items():
        lsb, width = XF[opc][k]
        v = int(v) & ((1 << width) - 1) if k in ("ma", "mb") else int(v)
        assert 0 <= v < (1 << width), (XNAMES[opc], k, v)
        w |= v << lsb
    return w


def xdec(w):
    op = w & 0xFF
    f = {k: (w >> l) & ((1 << n) - 1) for k, (l, n) in XF[op].items()}
    for k in ("ma", "mb"):
        if k in f and f[k] & (1 << 17):
            f[k] -= 1 << 18
    return op, f


def disasm(w):
    op = w & 0xFF
    if op in XF:
        _, f = xdec(w)
        return XNAMES[op] + " " + " ".join(f"{k}={v}" for k, v in f.items())
    return asm.disasm(w)


def row_word(vals):
    return isasim.row_to_int([int(v) for v in vals])


class Compiler:
    def __init__(self, P, B, macro=True):
        self.P, self.B, self.macro = P, B, macro
        self.prog, self.wrom, self.prom, self.lrom = [], [], [], []
        self.qentry = {}            # key -> entry index
        self.marks = []
        self.tile_idx = 0
        self.qset_busy, self.half_busy, self.act_pending = {}, {k: False for k in range(NREG)}, False

    # --------------------------------------------------------------- ROM helpers
    def entry(self, key, bias, m0, shift):
        if key in self.qentry:
            return self.qentry[key]
        idx = len(self.prom) // 17
        for c in range(N):
            self.prom.append((int(m0[c]) & 0xFFFFFFFF) | ((int(shift[c]) & 0xFF) << 32)
                             | ((int(bias[c]) & 0xFFFFFFFF) << 40))
        self.prom.append(0 | ((-128 & 0xFF) << 8) | (127 << 16))     # zp 0, act -128..127
        self.qentry[key] = idx
        return idx

    def sync_act(self):
        self.prog.append(asm.sync(asm.U_ACT))
        self.qset_busy, self.half_busy, self.act_pending = {}, {k: False for k in range(NREG)}, False

    def blocking(self):
        """Extension ops that touch the UB wait for the core to drain (the sequencer does the
        same in hardware); the ISA sim needs no SYNC, but tpuv1's hazard model is reset."""
        self.qset_busy, self.half_busy, self.act_pending = {}, {k: False for k in range(NREG)}, False

    # --------------------------------------------------------------- matmul (+ requant)
    def matmul(self, name, in_base, K, Nout, out_base, w_src, key, bias, m0, shift):
        """w_src: ('rom', W[K,N] int8) or ('ub', tr, base, stride). Output tile nt at out_base + nt*B."""
        B = self.B
        KT, NT = (K + 15) // 16, (Nout + 15) // 16
        need_sync = self.act_pending           # RAW: this layer may read what ACTIVATE wrote
        if need_sync:
            self.qset_busy, self.half_busy, self.act_pending = {}, {k: False for k in range(NREG)}, False
        if w_src[0] == "rom":
            W = w_src[1]
            Wp = np.zeros((KT * 16, NT * 16), np.int64); Wp[:K, :Nout] = W
            for nt in range(NT):
                for kt in range(KT):
                    tile = Wp[16 * kt:16 * kt + 16, 16 * nt:16 * nt + 16]
                    self.wrom += [row_word(tile[15 - t]) for t in range(16)]
            self.prog.append(xenc(X_WROM, count=KT * NT * 16))
        else:
            _, tr, base, stride = w_src
            self.blocking(); need_sync = False
            self.prog.append(xenc(X_WUB, tr=tr, base=base, stride=stride, kt=KT, nt=NT))
        bias = np.broadcast_to(bias, (Nout,)); m0 = np.broadcast_to(m0, (Nout,)); shift = np.broadcast_to(shift, (Nout,))
        shared = np.all(m0 == m0[0]) and np.all(shift == shift[0]) and np.all(bias == bias[0])
        for nt in range(NT):
            q, half = self.tile_idx % 4, self.tile_idx % NREG
            if self.qset_busy.get(q) or self.half_busy[half]:
                need_sync = True
                self.qset_busy, self.half_busy, self.act_pending = {}, {k: False for k in range(NREG)}, False
            sl = slice(16 * nt, 16 * nt + 16)
            e = self.entry((key, "all") if shared else (key, nt), bias[sl], m0[sl], shift[sl])
            a = half * B
            if self.macro:
                self.prog.append(xenc(X_MM, q=q, sync=int(need_sync), kt=KT, in_base=in_base, acc_row=a,
                                      out_row=out_base + nt * B, entry=e, B=B))
            else:
                if need_sync:
                    self.prog.append(asm.sync(asm.U_ACT))
                for kt in range(KT):               # matmuls first: they overlap the previous ACTIVATE
                    self.prog.append(asm.matmul(in_base + kt * B, a, B, acc=1 if kt else 0, new_weights=1))
                self.prog.append(xenc(X_LOADQ, q=q, entry=e))   # config waits for that ACTIVATE, array keeps running
                self.prog.append(asm.sync(asm.U_MATRIX))
                self.prog.append(asm.activate(a, out_base + nt * B, B, q))
            need_sync = False
            self.qset_busy[q] = True; self.half_busy[half] = True; self.act_pending = True
            self.tile_idx += 1
        self.mark(name)

    def mark(self, name):
        self.marks.append((name, len(self.prog), len(self.wrom)))

    def vec(self, op, rows, a, o, count, b=0, rstride=None, vstride=1):
        self.blocking()
        self.prog.append(xenc(X_VEC, op=op, rows=rows, a=a, b=b, o=o, count=count,
                              rstride=self.B if rstride is None else rstride, vstride=vstride))
        self.mark(f"vec{op}@{len(self.prog)}")

    def lut(self, sel, idx, table):
        need = (idx + 1) * 256
        while len(self.lrom) < need:
            self.lrom.append(0)
        self.lrom[idx * 256:(idx + 1) * 256] = [int(v) & (0xFFFF if sel == 0 else 0xFF) for v in table]
        self.blocking()
        self.prog.append(xenc(X_LUT, sel=sel, idx=idx))

    # --------------------------------------------------------------- whole encoder
    def encoder(self):
        P, B = self.P, self.B
        c = P["cfg"]; D, H, L, M = c["dim"], c["heads"], c["depth"], c["mlp"]
        dh = D // H; DT = D // 16
        # UB map (rows); every tensor is k-tile major: tile t of row b at base + t*B + b
        R = {}
        nxt = 0
        for name, cols in (("img", 16), ("pos", D), ("e", D), ("x", D), ("h", D), ("qkv", 3 * D),
                           ("s", B), ("o", D), ("a", D), ("f1", M), ("f2", D), ("out", D)):
            R[name] = nxt; nxt += ((cols + 15) // 16) * B
        self.R, self.ub_rows = R, nxt
        # position embedding: ROM -> UB
        pos = P["pos"]
        for t in range(DT):
            for b in range(B):
                self.wrom.append(row_word(pos[b, 16 * t:16 * t + 16]))
        self.blocking()
        self.prog.append(xenc(X_UBROM, base=R["pos"], count=DT * B))
        # patch embedding + pos
        p = P["embed"]
        self.matmul("embed", R["img"], p["w"].shape[0], D, R["e"], ("rom", p["w"]), "embed", p["bias"], p["m0"], p["shift"])
        self.prog.append(xenc(X_VCFG, ma=P["add0"][0], mb=P["add0"][1]))
        self.vec(V_ADD, DT, R["e"], R["x"], B, b=R["pos"])
        for l in range(L):
            self.lut(0, l, P[f"exp{l}"]); self.lut(1, L + l, P[f"gelu{l}"])
            self.vec(V_LN, DT, R["x"], R["h"], B)
            p = P[f"qkv{l}"]
            self.matmul(f"qkv{l}", R["h"], D, 3 * D, R["qkv"], ("rom", p["w"]), f"qkv{l}", p["bias"], p["m0"], p["shift"])
            for h in range(H):
                kb = R["qkv"] + (DT + h * dh // 16) * B
                vb = R["qkv"] + (2 * DT + h * dh // 16) * B
                qb = R["qkv"] + (h * dh // 16) * B
                m0, sh = P[f"qk{l}"]
                self.matmul(f"qk{l}h{h}", qb, dh, B, R["s"], ("ub", 1, kb, B), f"qk{l}", 0, m0, sh)
                self.vec(V_SM, B // 16, R["s"], R["s"], B)
                m0, sh = P[f"pv{l}"]
                self.matmul(f"pv{l}h{h}", R["s"], B, dh, R["o"] + (h * dh // 16) * B, ("ub", 0, vb, B), f"pv{l}", 0, m0, sh)
            p = P[f"proj{l}"]
            self.matmul(f"proj{l}", R["o"], D, D, R["a"], ("rom", p["w"]), f"proj{l}", p["bias"], p["m0"], p["shift"])
            self.prog.append(xenc(X_VCFG, ma=P[f"adda{l}"][0], mb=P[f"adda{l}"][1]))
            self.vec(V_ADD, DT, R["x"], R["x"], B, b=R["a"])
            self.vec(V_LN, DT, R["x"], R["h"], B)
            p = P[f"fc1{l}"]
            self.matmul(f"fc1{l}", R["h"], D, M, R["f1"], ("rom", p["w"]), f"fc1{l}", p["bias"], p["m0"], p["shift"])
            self.vec(V_LUT, M // 16, R["f1"], R["f1"], B)
            p = P[f"fc2{l}"]
            self.matmul(f"fc2{l}", R["f1"], M, D, R["f2"], ("rom", p["w"]), f"fc2{l}", p["bias"], p["m0"], p["shift"])
            self.prog.append(xenc(X_VCFG, ma=P[f"addb{l}"][0], mb=P[f"addb{l}"][1]))
            self.vec(V_ADD, DT, R["x"], R["x"], B, b=R["f2"])
        self.vec(V_LN, DT, R["x"], R["out"], B)
        self.prog += [asm.sync(asm.U_ALL), asm.interrupt_host(0x7E9A0001), asm.halt()]


# ------------------------------------------------------------------- Python machine model
class Machine:
    """tpuv1 CoreSim for core instructions + the sequencer extension, bit-exact spec."""
    def __init__(self, cp, ub_init):
        self.cp = cp
        self.core = isasim.CoreSim(ub_init, [], check_hazards=True)
        self.rp = 0                       # weight-ROM read pointer
        self.lut = {0: None, 1: None}
        self.ma = self.mb = 0

    def loadq(self, q, entry):
        base = entry * 17
        for c in range(N):
            v = self.cp.prom[base + c]
            self.core.step(asm.set_config(0x1000 | (q << 4) | c, v & 0xFFFFFFFFFF))
            self.core.step(asm.set_config(0x2000 | (q << 4) | c, (v >> 40) & 0xFFFFFFFF))
        self.core.step(asm.set_config(0x3000 | q, self.cp.prom[base + 16]))

    def ubrow(self, r):
        return np.array(self.core.ub[r], np.int64)

    def run(self):
        cp, core = self.cp, self.core
        for w in cp.prog:
            op = w & 0xFF
            if op not in XF:
                if not core.step(w):
                    assert core.halted and not core.cfg_err, core.__dict__.get("cfg_err_why")
                continue
            _, f = xdec(w)
            if op == X_WROM:
                core.wq += cp.wrom[self.rp:self.rp + f["count"]]; self.rp += f["count"]
            elif op == X_UBROM:
                core._sync(asm.U_ALL)
                for i in range(f["count"]):
                    core.ub[f["base"] + i] = isasim.int_to_row(cp.wrom[self.rp + i])
                self.rp += f["count"]
            elif op == X_WUB:
                core._sync(asm.U_ALL)
                for nt in range(f["nt"]):
                    for kt in range(f["kt"]):
                        if f["tr"]:
                            rows = [self.ubrow(f["base"] + kt * f["stride"] + 16 * nt + j) for j in range(16)]
                            tile = np.stack(rows).T          # W[i][j] = row_j[i]
                        else:
                            tile = np.stack([self.ubrow(f["base"] + nt * f["stride"] + 16 * kt + i) for i in range(16)])
                        core.wq += [row_word(tile[15 - t]) for t in range(16)]
            elif op == X_MM:
                if f["sync"]:
                    core.step(asm.sync(asm.U_ACT))
                for kt in range(f["kt"]):
                    core.step(asm.matmul(f["in_base"] + kt * f["B"], f["acc_row"], f["B"], acc=1 if kt else 0, new_weights=1))
                self.loadq(f["q"], f["entry"])
                core.step(asm.sync(asm.U_MATRIX))
                core.step(asm.activate(f["acc_row"], f["out_row"], f["B"], f["q"]))
            elif op == X_LOADQ:
                self.loadq(f["q"], f["entry"])
            elif op == -1:
                base = f["entry"] * 17; q = f["q"]
                for c in range(N):
                    v = cp.prom[base + c]
                    core.step(asm.set_config(0x1000 | (q << 4) | c, v & 0xFFFFFFFFFF))
                    core.step(asm.set_config(0x2000 | (q << 4) | c, (v >> 40) & 0xFFFFFFFF))
                core.step(asm.set_config(0x3000 | q, cp.prom[base + 16]))
            elif op == X_LUT:
                t = np.array(cp.lrom[f["idx"] * 256:(f["idx"] + 1) * 256], np.int64)
                self.lut[f["sel"]] = t if f["sel"] == 0 else np.where(t > 127, t - 256, t)
            elif op == X_VCFG:
                self.ma, self.mb = f["ma"], f["mb"]
            elif op == X_VEC:
                core._sync(asm.U_ALL)
                for v in range(f["count"]):
                    rd = lambda base: np.concatenate([self.ubrow(base + v * f["vstride"] + r * f["rstride"]) for r in range(f["rows"])])
                    a = rd(f["a"])
                    if f["op"] == V_ADD:
                        y = I.ew_add(a, rd(f["b"]), self.ma, self.mb)
                    elif f["op"] == V_LUT:
                        y = self.lut[1][a + 128]
                    elif f["op"] == V_LN:
                        y = I.layernorm(a[None])[0]
                    else:
                        y = I.softmax(a[None], self.lut[0])[0]
                    for r in range(f["rows"]):
                        core.ub[f["o"] + v * f["vstride"] + r * f["rstride"]] = [int(x) for x in y[16 * r:16 * r + 16]]
        assert core.halted and core.irq_count == 1
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmodel", default="qmodel.npy")
    ap.add_argument("--img", type=int, default=7)
    ap.add_argument("--out", default="rom_e2e")
    ap.add_argument("--no-macro", action="store_true", help="emit tiles instruction by instruction")
    ap.add_argument("--stop", type=int, default=-1, help="debug: stop after mark #k, expect the whole UB")
    a = ap.parse_args()
    P = np.load(a.qmodel, allow_pickle=True).item()
    from train import load_mnist
    _, _, xte, _ = load_mnist("/root/data/mnist.npz")
    xq = quant_img(P, xte[a.img:a.img + 1])[0]                 # [64, 16]
    B = xq.shape[0]
    cp = Compiler(P, B, macro=not a.no_macro); cp.encoder()
    if a.stop >= 0:
        name, plen, _ = cp.marks[a.stop]
        print("stop after", name)
        cp.prog = cp.prog[:plen] + [asm.sync(asm.U_ALL), asm.interrupt_host(0x7E9A0001), asm.halt()]
        cp.R["out"] = 0
    ub_init = [[0] * N for _ in range(cp.ub_rows)]
    for b in range(B):
        ub_init[cp.R["img"] + b] = [int(v) for v in xq[b]]
    m = Machine(cp, ub_init).run()
    ref = int_forward(P, xq)
    D = P["cfg"]["dim"]
    if a.stop >= 0:
        cp_out_rows = cp.ub_rows
    else:
        cp_out_rows = None
    got = None if a.stop >= 0 else np.array([[m.core.ub[cp.R["out"] + t * B + b][i] for t in range(D // 16) for i in range(16)] for b in range(B)])
    assert a.stop >= 0 or np.array_equal(got, ref), "machine model != int_forward"
    os.makedirs(a.out, exist_ok=True)
    asm.write_hex(f"{a.out}/prog.hex", cp.prog)
    with open(f"{a.out}/prog.lst", "w") as fh:
        fh.writelines(f"{i:6d}  {w:032x}  {disasm(w)}\n" for i, w in enumerate(cp.prog))
    asm.write_hex(f"{a.out}/wrom.hex", cp.wrom)
    with open(f"{a.out}/prom.hex", "w") as fh: fh.writelines(f"{v:018x}\n" for v in cp.prom)
    with open(f"{a.out}/lrom.hex", "w") as fh: fh.writelines(f"{v:04x}\n" for v in cp.lrom)
    asm.write_hex(f"{a.out}/ub_init.hex", [row_word(r) for r in ub_init[:B]])
    n_out = cp_out_rows if cp_out_rows else (D // 16) * B
    exp = [m.core.ub.get(cp.R["out"] + i, [0] * N) for i in range(n_out)]
    asm.write_hex(f"{a.out}/exp_out.hex", [row_word(r) for r in exp])
    meta = dict(prog=len(cp.prog), wrom=len(cp.wrom), prom=len(cp.prom), lrom=len(cp.lrom), in_rows=B,
                out_base=cp.R["out"], out_rows=len(exp), ub_rows=cp.ub_rows)
    with open(f"{a.out}/meta.txt", "w") as fh: fh.writelines(f"{k} {v}\n" for k, v in meta.items())
    kib = lambda n, bits: n * bits / 8 / 1024
    print(f"program {len(cp.prog)} instr ({kib(len(cp.prog),128):.0f} KiB), weight ROM {len(cp.wrom)} words "
          f"({kib(len(cp.wrom),128):.0f} KiB), param ROM {len(cp.prom)//17} entries ({kib(len(cp.prom),72):.0f} KiB), "
          f"LUT ROM {kib(len(cp.lrom),16):.0f} KiB, UB used {cp.ub_rows} rows ({kib(cp.ub_rows,128):.0f} KiB)")
    print("machine model == quantize.int_forward: bit-exact ->", a.out)


if __name__ == "__main__":
    main()
