"""Run every matmul (+ requant) of the JEPA encoder on the unmodified tpuv1 RTL core.

For one test image, the integer graph (quantize.int_forward) logs each matmul. Each one becomes a
tpu_core test case (program + weight stream + UB image + expected UB output) built with tpuv1's
own MLP compiler (single layer, zero-point 0), checked first against tpuv1's ISA simulator, then
run in Verilator by tb_tpu_core, which compares the UB output bit for bit.

  python3 rtl_check.py --qmodel qmodel.npy --img 0 --out /root/tpuv1/build/jepa
"""
import argparse, os, re, sys
import numpy as np

TPU = "/root/tpuv1"
sys.path.insert(0, f"{TPU}/model/mlp"); sys.path.insert(0, f"{TPU}/model/isa"); sys.path.insert(0, f"{TPU}/model/quantization")
import compile_mlp as C       # noqa: E402
import sim as isasim          # noqa: E402
import intops as I            # noqa: E402
from quantize import int_forward, quant_img   # noqa: E402
from train import load_mnist  # noqa: E402

N = 16


class OneLayer:
    """Duck-typed tpuv1 Net with a single layer and zp 0, built from our integer params."""
    def __init__(self, w, bias, m0, shift):
        K, Nout = w.shape
        m0 = np.broadcast_to(m0, (Nout,)); shift = np.broadcast_to(shift, (Nout,)); bias = np.broadcast_to(bias, (Nout,))
        self.dims, self.acts = [K, Nout], ["none"]
        self.layers = [dict(wq=w, bq=bias, bhw=np.asarray(bias, np.int64), m0=[int(v) for v in m0],
                            shift=[int(v) for v in shift], zp_out=0, act_min=-128, act_max=127)]


def build_case(outdir, name, x, net, expect):
    B, K = x.shape
    Kp = C.pad16(K)
    xq = np.zeros((B, Kp), np.int64); xq[:, :K] = x
    prog, out_first, ub_end, acc_rows = C.emit_program(net, B)
    ww = C.weight_words(net)
    ub = [[0] * N for _ in range(ub_end)]
    for kt in range(Kp // N):
        for b in range(B):
            ub[kt * B + b] = [int(v) for v in xq[b, N * kt:N * kt + N]]
    Np = C.pad16(expect.shape[1])
    ye = np.zeros((B, Np), np.int64); ye[:, :expect.shape[1]] = expect
    exp = [[int(v) for v in ye[b, N * nt:N * nt + N]] for nt in range(Np // N) for b in range(B)]
    s = isasim.CoreSim(ub, ww, check_hazards=True).run(prog)
    got = [s.ub.get(out_first + i) for i in range(len(exp))]
    assert got == exp, f"{name}: tpuv1 ISA sim disagrees with the JEPA integer graph"
    C.write_case(outdir, name, prog, ww, ub, out_first, exp, acc_rows, s.acc,
                 dict(B=B, layers=f"{K}-{expect.shape[1]}", out_cols=Np, irq_expected=1,
                      irq_tag=f"{s.irq_tag:08x}", cfg_err=0, cfg_err_op="00"))
    return len(prog), len(ww)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmodel", default="qmodel.npy")
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--out", default=f"{TPU}/build/jepa")
    a = ap.parse_args()
    P = np.load(a.qmodel, allow_pickle=True).item()
    _, _, xte, _ = load_mnist("/root/data/mnist.npz")
    log = []
    int_forward(P, quant_img(P, xte[a.img:a.img + 1])[0], log)
    cases = os.path.join(a.out, "cases"); os.makedirs(cases, exist_ok=True)
    dirs, tot_i, tot_w, macs = [], 0, 0, 0
    for name, x, w, acc in log:
        mt = re.match(r"^(qk|pv)(\d+)h\d+$", name)
        if mt:
            l = int(mt.group(2))
            m0, sh = P[f"{name[:2]}{l}"]
            net = OneLayer(np.asarray(w, np.int64), 0, m0, sh)
            y = I.requant(acc, 0, m0, sh)
        else:
            p = P[name]
            net = OneLayer(p["w"], p["bias"], p["m0"], p["shift"])
            y = I.requant(acc, p["bias"], p["m0"], p["shift"])
        ni, nw = build_case(cases, name, np.asarray(x, np.int64), net, y)
        tot_i += ni; tot_w += nw; macs += x.shape[0] * x.shape[1] * w.shape[1]
        dirs.append(os.path.relpath(os.path.join(cases, name), TPU))
    with open(os.path.join(cases, "cases.txt"), "w") as f:
        f.write("\n".join(dirs) + "\n")
    print(f"{len(dirs)} matmul cases, {tot_i} instructions, {tot_w} weight words, {macs/1e6:.1f} M MACs "
          f"-> {os.path.join(cases, 'cases.txt')} (all agree with tpuv1 ISA sim)")


if __name__ == "__main__":
    main()
