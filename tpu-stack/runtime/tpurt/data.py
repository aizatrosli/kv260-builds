"""Model + input helpers for the JEPA encoder that need neither torch nor jepa-tpu/quantize.py
(the board may not have torch).  Equivalence with quantize.quant_img / int_forward is tested."""
import numpy as np

from . import _paths

MEAN, STD = 0.1307, 0.3081


def load_qmodel(path=None):
    if path is None:
        path = _paths.need_jepa() / "qmodel.npy"
    return np.load(path, allow_pickle=True).item()


def load_images(n, start=0, seed=0):
    """n normalised 32x32 images [n, 32, 32] float64: MNIST test set (as jepa-tpu/train.load_mnist)
    when an npz is found, else deterministic synthetic strokes. Returns (images, source)."""
    p = _paths.mnist_path()
    if p is not None:
        x = np.load(p)["x_test"][start:start + n]
        x = np.pad(x, ((0, 0), (2, 2), (2, 2))).astype(np.float32) / np.float32(255)
        return ((x - np.float32(MEAN)) / np.float32(STD)).astype(np.float64), f"mnist:{p}"
    rng = np.random.default_rng(seed + start)
    img = np.zeros((n, 32, 32), np.float32)
    yy, xx = np.mgrid[:32, :32]
    for i in range(n):
        for _ in range(3):
            cy, cx, r = rng.uniform(8, 24), rng.uniform(8, 24), rng.uniform(3, 9)
            img[i] += np.exp(-((np.hypot(yy - cy, xx - cx) - r) ** 2) / 2)
    img = np.clip(img, 0, 1)
    return ((img - np.float32(MEAN)) / np.float32(STD)).astype(np.float64), "synthetic"


def patchify(img, p=4):
    """[B, H, W] (1 channel) -> [B, (H/p)*(W/p), p*p]  (model.patchify for C=1)."""
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[:, 0]
    B, H, W = img.shape
    return img.reshape(B, H // p, p, W // p, p).transpose(0, 1, 3, 2, 4).reshape(B, (H // p) * (W // p), p * p)


def quant_img(P, img):
    """[B, 32, 32] float -> int64 [B, 64, 16] (quantize.quant_img for a batch)."""
    x = patchify(np.asarray(img, np.float32), P["cfg"]["patch"]).astype(np.float64)
    return np.clip(np.round(x / P["s"]["img"]), -128, 127).astype(np.int64)


def reference_forward(P, img_q):
    """quantize.int_forward without torch: img_q [64, 16] -> int8 values [64, D]."""
    _paths.need_jepa()
    import intops as I
    c = P["cfg"]
    H, D = c["heads"], c["dim"]
    dh = D // H
    lin = lambda x, p: I.requant(I.matmul(x, p["w"]), p["bias"], p["m0"], p["shift"])
    x = I.ew_add(lin(img_q, P["embed"]), P["pos"], *P["add0"])
    for l in range(c["depth"]):
        qkv = lin(I.layernorm(x), P[f"qkv{l}"])
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        heads = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            s = I.requant(I.matmul(q[:, sl], k[:, sl].T), 0, *P[f"qk{l}"])
            heads.append(I.requant(I.matmul(I.softmax(s, P[f"exp{l}"]), v[:, sl]), 0, *P[f"pv{l}"]))
        x = I.ew_add(x, lin(np.concatenate(heads, -1), P[f"proj{l}"]), *P[f"adda{l}"])
        g = I.lut8(lin(I.layernorm(x), P[f"fc1{l}"]), P[f"gelu{l}"])
        x = I.ew_add(x, lin(g, P[f"fc2{l}"]), *P[f"addb{l}"])
    return I.layernorm(x)


def golden_forward(P, img_q):
    """jepa-tpu's own quantize.int_forward when importable (needs torch), else reference_forward."""
    try:
        _paths.need_jepa()
        import quantize
        return quantize.int_forward(P, img_q)
    except ImportError:
        return reference_forward(P, img_q)
