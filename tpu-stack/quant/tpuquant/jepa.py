"""Adapters for the jepa-tpu tiny I-JEPA encoder (read-only import from ../../jepa-tpu)."""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
JEPA_DIR = os.environ.get("JEPA_TPU_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "..", "jepa-tpu")))
MNIST = os.environ.get("TPUQUANT_MNIST", os.path.join(HERE, "..", "data", "mnist.npz"))


class _NoBytecode:
    """Import from jepa-tpu without writing __pycache__ into a directory we do not own."""
    def __enter__(self):
        self.old = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    def __exit__(self, *a):
        sys.dont_write_bytecode = self.old


def jepa_modules():
    if JEPA_DIR not in sys.path:
        sys.path.insert(0, JEPA_DIR)
    with _NoBytecode():
        import model as jm
    return jm


def load_encoder(ckpt=None):
    jm = jepa_modules()
    ck = torch.load(ckpt or os.path.join(JEPA_DIR, "ckpt.pt"), map_location="cpu")
    enc = jm.Encoder(jm.Cfg(**ck["cfg"]))
    enc.load_state_dict(ck["encoder"])
    return enc.eval()


def ensure_mnist(path=MNIST):
    """MNIST as npz (x_train, y_train, x_test, y_test); downloads via torchvision if missing."""
    if not os.path.exists(path):
        import torchvision
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = os.path.join(os.path.dirname(path), "raw")
        tr = torchvision.datasets.MNIST(raw, train=True, download=True)
        te = torchvision.datasets.MNIST(raw, train=False, download=True)
        np.savez_compressed(path, x_train=tr.data.numpy(), y_train=tr.targets.numpy(),
                            x_test=te.data.numpy(), y_test=te.targets.numpy())
    return path


def load_mnist(path=None):
    """Same preprocessing as jepa-tpu/train.py: pad to 32x32, normalise."""
    d = np.load(ensure_mnist(path or MNIST))
    f = lambda x: torch.from_numpy(np.pad(x, ((0, 0), (2, 2), (2, 2)))).float().div(255).sub(0.1307).div(0.3081)[:, None]
    return f(d["x_train"]), torch.from_numpy(d["y_train"]).long(), f(d["x_test"]), torch.from_numpy(d["y_test"]).long()


def calibrate_encoder(enc, xcal, **kw):
    from .ptq import Calibrator
    return Calibrator(enc, xcal, concrete_args={"keep": None}, name="jepa_encoder", **kw)


def cosine_rows(a, b):
    """Mean over samples of the cosine of flattened per-sample outputs (jepa-tpu's metric)."""
    a = np.asarray(a, np.float64).reshape(len(a), -1); b = np.asarray(b, np.float64).reshape(len(b), -1)
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return c


def jepa_quantize_module():
    jepa_modules()
    with _NoBytecode():
        import quantize as jq
    return jq


def load_jepa_qmodel(path=None):
    return np.load(path or os.path.join(JEPA_DIR, "qmodel.npy"), allow_pickle=True).item()


def jepa_fixed_qparams(cal, P):
    """Map jepa-tpu's calibrated scales P['s'] onto the fx nodes of the traced encoder, so the
    toolkit lowers to *the same* integer program (used to prove bit-exact compatibility)."""
    S = P["s"]; c = P["cfg"]; dh = c["dim"] // c["heads"]
    nodes = {n.name: n for n in cal.nodes}
    users = lambda n: cal.real_users(n)
    one = lambda n, kind: [u for u in users(n) if cal.kind(u) == kind][0]
    fq = {"img": (S["img"], 0), "embed": (S["embed"], 0), "pos": (S["pos"], 0)}
    fq[one(nodes["embed"], "add").name] = (S["res0"], 0)
    for l in range(c["depth"]):
        qkv = nodes[f"blocks_{l}_qkv"]
        perm = users(users(qkv)[0])[0]
        gi = {u.args[1]: u for u in users(perm)}
        q, k, v = gi[0], gi[1], gi[2]
        fq[q.name] = (S[f"q{l}"] * np.sqrt(dh), 0)
        fq[k.name] = (S[f"k{l}"], 0)
        for u in users(k):                       # k^T (the terminal of the layout chain)
            fq[u.name] = (S[f"k{l}"], 0)
        fq[v.name] = (S[f"v{l}"], 0)
        mm = one(q, "matmul")
        fq[one(mm, "div").name] = (S[f"s{l}"], 0)
        sm = one(one(mm, "div"), "softmax")
        fq[one(sm, "matmul").name] = (S[f"o{l}"], 0)
        proj = nodes[f"blocks_{l}_proj"]
        fq[proj.name] = (S[f"p{l}"], 0)
        fq[one(proj, "add").name] = (S[f"ra{l}"], 0)
        fc1 = nodes[f"blocks_{l}_fc1"]
        fq[fc1.name] = (S[f"h{l}"], 0)
        fq[one(fc1, "act:gelu").name] = (S[f"g{l}"], 0)
        fc2 = nodes[f"blocks_{l}_fc2"]
        fq[fc2.name] = (S[f"f{l}"], 0)
        fq[one(fc2, "add").name] = (S[f"rb{l}"], 0)
    return fq


def jepa_softmax_mults(cal, P):
    """{fx softmax node: jepa's per-layer numerator P['smq<l>']} for QConfig(softmax_mult=...).
    Empty for a qmodel without per-layer numerators (all layers then use 127)."""
    c = P["cfg"]
    nodes = {n.name: n for n in cal.nodes}
    one = lambda n, kind: [u for u in cal.real_users(n) if cal.kind(u) == kind][0]
    out = {}
    for l in range(c["depth"]):
        if f"smq{l}" not in P:
            continue
        perm = cal.real_users(cal.real_users(nodes[f"blocks_{l}_qkv"])[0])[0]
        q = {u.args[1]: u for u in cal.real_users(perm)}[0]
        out[one(one(one(q, "matmul"), "div"), "softmax").name] = int(P[f"smq{l}"])
    return out
