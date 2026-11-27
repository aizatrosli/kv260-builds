"""Build IR graphs from the jepa-tpu quantised model and from tpuv1's MLP / CNN test networks.

Graphs are built *unfused* (matmul -> requant -> clamp -> pool as separate ops) exactly
following each model's integer reference; ``passes.fuse`` folds them for the TPU.
"""
import numpy as np

from .ir import Graph

I64 = np.int64


def _linear(g, name, x, w, bias, m0, shift, zp_out=0, act=None, act_min=-128, act_max=127, transpose_b=False):
    """matmul + requant (+ clamp for relu/relu6). Returns output tensor name."""
    xt = g.T(x)
    wt = g.T(w)
    n = wt.shape[0] if transpose_b else wt.shape[1]
    acc = g.op("matmul", name, [x, w], (xt.shape[0], n), "int32", out_name=f"{name}.acc", transpose_b=transpose_b)
    bc = lambda v: np.broadcast_to(np.asarray(v, I64), (n,)).copy()
    y = g.op("requant", f"{name}.rq", [acc], (xt.shape[0], n), out_name=name if act is None else f"{name}.rq",
             bias=bc(bias), m0=bc(m0), shift=bc(shift), zp_out=int(zp_out), act_min=-128, act_max=127)
    if act is not None:
        y = g.op("clamp", f"{name}.act", [y], (xt.shape[0], n), out_name=name, lo=int(act_min), hi=int(act_max), act=act)
    return y


# ---------------------------------------------------------------------------------------------
def from_jepa(P, name="jepa_encoder"):
    """jepa-tpu quantize.int_forward as an IR graph (input: img_q [T, patch_dim] int8)."""
    c = P["cfg"]
    D, H, depth = c["dim"], c["heads"], c["depth"]
    T = (c["img"] // c["patch"]) ** 2
    dh = D // H
    g = Graph(name)
    img = g.input("img", (T, c["chans"] * c["patch"] ** 2))

    def lin(nm, x, p):
        w = g.const(f"{nm}.w", p["w"])
        return _linear(g, nm, x, w, p["bias"], p["m0"], p["shift"])

    e = lin("embed", img, P["embed"])
    pos = g.const("pos", P["pos"])
    x = g.op("add", "add0", [e, pos], (T, D), ma=P["add0"][0], mb=P["add0"][1])
    for l in range(depth):
        h = g.op("layernorm", f"ln1_{l}", [x], (T, D))
        qkv = lin(f"qkv{l}", h, P[f"qkv{l}"])
        heads = []
        for hh in range(H):
            sl = lambda base, nm: g.op("slice", f"{nm}{l}h{hh}", [qkv], (T, dh), c0=base + hh * dh, c1=base + (hh + 1) * dh)
            q, k, v = sl(0, "q"), sl(D, "k"), sl(2 * D, "v")
            m0, sh = P[f"qk{l}"]
            s = _linear(g, f"qk{l}h{hh}", q, k, 0, m0, sh, transpose_b=True)
            p = g.op("softmax", f"sm{l}h{hh}", [s], (T, T), lut=np.asarray(P[f"exp{l}"], I64))
            m0, sh = P[f"pv{l}"]
            heads.append(_linear(g, f"pv{l}h{hh}", p, v, 0, m0, sh))
        o = g.op("concat", f"o{l}", heads, (T, D))
        a = lin(f"proj{l}", o, P[f"proj{l}"])
        x = g.op("add", f"adda{l}", [x, a], (T, D), ma=P[f"adda{l}"][0], mb=P[f"adda{l}"][1])
        h = g.op("layernorm", f"ln2_{l}", [x], (T, D))
        h1 = lin(f"fc1{l}", h, P[f"fc1{l}"])
        gl = g.op("lut", f"gelu{l}", [h1], (T, c["mlp"]), table=np.asarray(P[f"gelu{l}"], I64))
        f2 = lin(f"fc2{l}", gl, P[f"fc2{l}"])
        x = g.op("add", f"addb{l}", [x, f2], (T, D), ma=P[f"addb{l}"][0], mb=P[f"addb{l}"][1])
    out = g.op("layernorm", "ln_out", [x], (T, D))
    g.mark_output(out)
    g.validate()
    return g


def jepa_matmul_names(P):
    c = P["cfg"]
    names = ["embed"]
    for l in range(c["depth"]):
        names.append(f"qkv{l}")
        for hh in range(c["heads"]):
            names += [f"qk{l}h{hh}", f"pv{l}h{hh}"]
        names += [f"proj{l}", f"fc1{l}", f"fc2{l}"]
    return names


def synthetic_image(P, seed=0):
    """A deterministic digit-like 32x32 image (MNIST normalisation), patchified and quantised
    exactly like jepa-tpu quantize.quant_img. (MNIST itself is not available offline.)"""
    c = P["cfg"]
    S, p = c["img"], c["patch"]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S]
    img = np.zeros((S, S))
    for _ in range(3):                                   # a few thick random strokes
        x0, y0, x1, y1 = rng.uniform(6, S - 6, 4)
        t = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) / max((x1 - x0) ** 2 + (y1 - y0) ** 2, 1e-6), 0, 1)
        d = np.hypot(xx - (x0 + t * (x1 - x0)), yy - (y0 + t * (y1 - y0)))
        img = np.maximum(img, np.clip(2.2 - d, 0, 1))
    img = (img - 0.1307) / 0.3081
    x = img.reshape(S // p, p, S // p, p).transpose(0, 2, 1, 3).reshape((S // p) ** 2, p * p)
    return np.clip(np.round(x / P["s"]["img"]), -128, 127).astype(I64)


# ---------------------------------------------------------------------------------------------
def from_mlp(net, B, name="mlp"):
    """tpuv1 model/mlp/compile_mlp.Net (TFLite int8 MLP) as an IR graph; input [B, dims[0]]."""
    g = Graph(name)
    x = g.input("x", (B, net.dims[0]))
    for l, (L, act) in enumerate(zip(net.layers, net.acts)):
        w = g.const(f"fc{l}.w", L["wq"])
        x = _linear(g, f"fc{l}", x, w, L["bhw"], L["m0"], L["shift"], L["zp_out"],
                    act=None if act == "none" else act, act_min=L["act_min"], act_max=L["act_max"])
    g.mark_output(x)
    g.validate()
    return g


def from_cnn(net, name="cnn"):
    """tpuv1 model/cnn/compile_cnn.Net (conv [+relu] [+pool] ... [+fc]) as an IR graph;
    input [H*W, C] (row = y*W + x)."""
    sp = net.spec
    g = Graph(name)
    x = g.input("x", (sp["h"] * sp["w"], sp["c"]), hw=(sp["h"], sp["w"]))
    hw = (sp["h"], sp["w"])
    for l, L in enumerate(net.layers):
        nm = f"l{l}"
        if L["kind"] == "conv":
            OH, OW = L["oh"], L["ow"]
            w = g.const(f"{nm}.w", L["wq"])
            acc = g.op("conv2d", nm, [x, w], (OH * OW, L["cout"]), "int32", out_name=f"{nm}.acc",
                       hw=hw, k=L["k"], stride=L["stride"], pad=L["pad"], pad_val=int(L["zp_in"]))
            y = g.op("requant", f"{nm}.rq", [acc], (OH * OW, L["cout"]), bias=np.asarray(L["bhw"], I64),
                     m0=np.asarray(L["m0"], I64), shift=np.asarray(L["shift"], I64), zp_out=int(L["zp_out"]),
                     act_min=-128, act_max=127)
            if L["act"] != "none":
                y = g.op("clamp", f"{nm}.act", [y], (OH * OW, L["cout"]), lo=int(L["act_min"]),
                         hi=int(L["act_max"]), act=L["act"])
            hw = (OH, OW)
            if L.get("pool"):
                y = g.op("pool2", f"{nm}.pool", [y], ((OH // 2) * (OW // 2), L["cout"]), hw=hw, mode=L["pool"])
                hw = (OH // 2, OW // 2)
            g.T(y).attrs["hw"] = hw
            x = y
        else:
            P_, c = L["fh"] * L["fw"], L["fc_c"]
            ct_n = (c + 15) // 16
            flat = g.op("flatten_planar", f"{nm}.flat", [x], (1, ct_n * 16 * P_))
            Wp = np.zeros((ct_n * P_ * 16, L["nout"]), I64)            # planar K order: (ct, p, lane)
            for ct in range(ct_n):
                for p in range(P_):
                    for ln in range(16):
                        ch = 16 * ct + ln
                        if ch < c:
                            Wp[(ct * P_ + p) * 16 + ln] = L["wq"][p * c + ch]
            w = g.const(f"{nm}.w", Wp)
            x = _linear(g, nm, flat, w, L["bhw"], L["m0"], L["shift"], L["zp_out"],
                        act=None if L["act"] == "none" else L["act"], act_min=L["act_min"], act_max=L["act_max"])
    g.mark_output(x)
    g.validate()
    return g
