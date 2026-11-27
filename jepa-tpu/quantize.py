"""Post-training quantisation of the tiny JEPA encoder to the TPU-KV260 integer graph.

  python3 quantize.py --ckpt ckpt.pt --out qmodel.npz

Builds the integer program (int8 weights, int32 biases, TFLite multipliers, LUTs), runs it
with intops.py (the bit-exact spec), and reports agreement with the float PyTorch model.
"""
import argparse, math
import numpy as np
import torch
import torch.nn.functional as F
import intops as I
from model import Cfg, Encoder, patchify
from train import load_mnist, probe


def amax(t, pct=99.99):
    a = t.detach().abs().flatten().numpy()
    return float(np.percentile(a, pct)) if a.size > 1000 else float(a.max())


@torch.no_grad()
def float_trace(enc, img):
    """Float forward along the exact integer graph; returns dict of tensors at quant points."""
    c = enc.c; T = {}
    x = patchify(img, c.patch); T["img"] = x
    e = enc.embed(x); T["embed"] = e
    T["pos"] = enc.pos.expand_as(e)
    x = e + enc.pos; T["res0"] = x
    for l, b in enumerate(enc.blocks):
        h = F.layer_norm(x, x.shape[-1:])
        h = h * b.ln1.weight + b.ln1.bias
        B, Tn, D = h.shape; dh = D // c.heads
        qkv = b.qkv(h).view(B, Tn, 3, c.heads, dh)
        q = qkv[:, :, 0] / math.sqrt(dh); k = qkv[:, :, 1]; v = qkv[:, :, 2]
        T[f"q{l}"], T[f"k{l}"], T[f"v{l}"] = q, k, v
        s = torch.einsum("bthd,buhd->bhtu", q, k); T[f"s{l}"] = s
        p = s.softmax(-1)
        o = torch.einsum("bhtu,buhd->bthd", p, v).reshape(B, Tn, D); T[f"o{l}"] = o
        a = b.proj(o); T[f"p{l}"] = a
        x = x + a; T[f"ra{l}"] = x
        h = F.layer_norm(x, x.shape[-1:]) * b.ln2.weight + b.ln2.bias
        h1 = b.fc1(h); T[f"h{l}"] = h1
        g = F.gelu(h1); T[f"g{l}"] = g
        f2 = b.fc2(g); T[f"f{l}"] = f2
        x = x + f2; T[f"rb{l}"] = x
    T["out"] = enc.ln(x)
    return T


def qweight(W):
    """W [in, out] float -> int8 per output channel, scales [out]."""
    s = np.maximum(np.abs(W).max(0), 1e-8) / 127
    return np.clip(np.round(W / s), -127, 127).astype(np.int64), s


def linear_params(W, b, s_in, s_out):
    Wq, sw = qweight(W)
    bias = np.round(b / (s_in * sw)).astype(np.int64)
    m0, sh = I.qmul_vec(s_in * sw / s_out)
    return dict(w=Wq, bias=bias, m0=m0, shift=sh)


def build(enc, calib):
    c = enc.c
    T = float_trace(enc, calib)
    S = {k: amax(v) / 127 for k, v in T.items()}
    s_ln = 1.0 / I.LN_Q
    P = {"cfg": c.__dict__, "s": S}
    np_ = lambda t: t.detach().double().numpy()
    P["embed"] = linear_params(np_(enc.embed.weight).T, np_(enc.embed.bias), S["img"], S["embed"])
    P["pos"] = np.clip(np.round(np_(enc.pos[0]) / S["pos"]), -128, 127).astype(np.int64)
    P["add0"] = (I.ew_mult(S["embed"], S["res0"]), I.ew_mult(S["pos"], S["res0"]))
    dh = c.dim // c.heads
    s_res = S["res0"]
    for l, b in enumerate(enc.blocks):
        g1, be1 = np_(b.ln1.weight), np_(b.ln1.bias)
        Wqkv = np_(b.qkv.weight).T; bqkv = np_(b.qkv.bias)
        bqkv = bqkv + be1 @ Wqkv; Wqkv = g1[:, None] * Wqkv           # fold LN affine
        Wqkv[:, :c.dim] /= math.sqrt(dh); bqkv[:c.dim] /= math.sqrt(dh)  # fold 1/sqrt(dh) into Q
        s_out = np.concatenate([np.full(c.dim, S[f"q{l}"]), np.full(c.dim, S[f"k{l}"]), np.full(c.dim, S[f"v{l}"])])
        Wq_, sw = qweight(Wqkv)
        P[f"qkv{l}"] = dict(w=Wq_, bias=np.round(bqkv / (s_ln * sw)).astype(np.int64),
                            **dict(zip(("m0", "shift"), I.qmul_vec(s_ln * sw / s_out))))
        P[f"qk{l}"] = I.qmul_vec(np.array([S[f"q{l}"] * S[f"k{l}"] / S[f"s{l}"]]))
        P[f"exp{l}"] = I.exp_lut(S[f"s{l}"])
        P[f"pv{l}"] = I.qmul_vec(np.array([(1 / 127) * S[f"v{l}"] / S[f"o{l}"]]))
        P[f"proj{l}"] = linear_params(np_(b.proj.weight).T, np_(b.proj.bias), S[f"o{l}"], S[f"p{l}"])
        P[f"adda{l}"] = (I.ew_mult(s_res, S[f"ra{l}"]), I.ew_mult(S[f"p{l}"], S[f"ra{l}"]))
        g2, be2 = np_(b.ln2.weight), np_(b.ln2.bias)
        W1 = np_(b.fc1.weight).T; b1 = np_(b.fc1.bias) + be2 @ W1; W1 = g2[:, None] * W1
        P[f"fc1{l}"] = linear_params(W1, b1, s_ln, S[f"h{l}"])
        P[f"gelu{l}"] = I.gelu_lut(S[f"h{l}"], S[f"g{l}"])
        P[f"fc2{l}"] = linear_params(np_(b.fc2.weight).T, np_(b.fc2.bias), S[f"g{l}"], S[f"f{l}"])
        P[f"addb{l}"] = (I.ew_mult(S[f"ra{l}"], S[f"rb{l}"]), I.ew_mult(S[f"f{l}"], S[f"rb{l}"]))
        s_res = S[f"rb{l}"]
    P["ln_out"] = (np_(enc.ln.weight), np_(enc.ln.bias))
    return P


def lin(x, p, log=None, name=None):
    acc = I.matmul(x, p["w"])
    if log is not None:
        log.append((name, x, p["w"], acc))
    return I.requant(acc, p["bias"], p["m0"], p["shift"])


def int_forward(P, img_q, log=None):
    """img_q [T, patch_dim] int8 -> final LN output int8 [T, D] (scale 1/LN_Q).
    log, if given, collects every matmul (name, x, w, acc) for the RTL check."""
    c = Cfg(**P["cfg"]); H = c.heads; D = c.dim; dh = D // H
    e = lin(img_q, P["embed"], log, "embed")
    x = I.ew_add(e, P["pos"], *P["add0"])
    for l in range(c.depth):
        h = I.layernorm(x)
        qkv = lin(h, P[f"qkv{l}"], log, f"qkv{l}")
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        heads = []
        for hh in range(H):
            sl = slice(hh * dh, (hh + 1) * dh)
            acc = I.matmul(q[:, sl], k[:, sl].T)
            if log is not None: log.append((f"qk{l}h{hh}", q[:, sl], k[:, sl].T, acc))
            s = I.requant(acc, 0, *P[f"qk{l}"])
            p = I.softmax(s, P[f"exp{l}"])
            acc = I.matmul(p, v[:, sl])
            if log is not None: log.append((f"pv{l}h{hh}", p, v[:, sl], acc))
            heads.append(I.requant(acc, 0, *P[f"pv{l}"]))
        o = np.concatenate(heads, -1)
        a = lin(o, P[f"proj{l}"], log, f"proj{l}")
        x = I.ew_add(x, a, *P[f"adda{l}"])
        h = I.layernorm(x)
        h1 = lin(h, P[f"fc1{l}"], log, f"fc1{l}")
        g = I.lut8(h1, P[f"gelu{l}"])
        f2 = lin(g, P[f"fc2{l}"], log, f"fc2{l}")
        x = I.ew_add(x, f2, *P[f"addb{l}"])
    return I.layernorm(x)


def quant_img(P, img):
    x = patchify(img, P["cfg"]["patch"]).double().numpy()
    return np.clip(np.round(x / P["s"]["img"]), -128, 127).astype(np.int64)


def dequant_out(P, y):
    g, b = P["ln_out"]
    return y / I.LN_Q * g + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--out", default="qmodel.npz")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.ckpt)
    enc = Encoder(Cfg(**ck["cfg"])); enc.load_state_dict(ck["encoder"]); enc.eval()
    xtr, ytr, xte, yte = load_mnist("/root/data/mnist.npz")
    P = build(enc, xtr[:256])
    # agreement on held-out images
    cos = []
    with torch.no_grad():
        ref = enc(xte[:200]).double().numpy()
    for i in range(200):
        y = dequant_out(P, int_forward(P, quant_img(P, xte[i:i + 1])[0]))
        r = ref[i]
        cos.append((y * r).sum() / np.linalg.norm(y) / np.linalg.norm(r))
    cos = np.array(cos)
    print(f"int8 vs float encoder output: cosine mean {cos.mean():.4f} min {cos.min():.4f} (200 test images)")

    class IntEnc(torch.nn.Module):            # probe wrapper: integer encoder features
        def forward(self, x):
            return torch.from_numpy(np.stack([dequant_out(P, int_forward(P, quant_img(P, x[i:i + 1])[0]))
                                              for i in range(len(x))])).float()
        def eval(self): return self
    n = 2000
    if a.no_probe:
        np.save(a.out.replace(".npz", ".npy"), P, allow_pickle=True); return
    print(f"linear probe (2k train / 1k test): float {probe(enc, xtr, ytr, xte[:1000], yte[:1000], n):.3f}  "
          f"int8 {probe(IntEnc(), xtr, ytr, xte[:1000], yte[:1000], n):.3f}")
    np.save(a.out.replace(".npz", ".npy"), P, allow_pickle=True)
    print("saved", a.out.replace(".npz", ".npy"))


if __name__ == "__main__":
    main()
