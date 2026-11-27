"""Integer graph IR + two interpreters.

* ``Graph.run_int``   bit-exact integer execution (the spec a compiler / RTL must match)
* ``Graph.run_float`` float execution of the same graph with optional fake-quantisation of any
  subset of activations / weights (used for float references and sensitivity analysis)

Tensors keep the PyTorch logical layout. Shapes may contain the batch symbol "N" (or "N*k").
"""
import math
from dataclasses import dataclass, field
import numpy as np

from . import intops as I
from .qscheme import quantize_np

LAYOUT_OPS = ("reshape", "permute", "getitem")
WEIGHT_OPS = ("linear", "conv2d")


@dataclass
class TensorInfo:
    name: str
    shape: list
    scale: object = 1.0          # float, or ndarray broadcastable to the tensor (batch dim = 1)
    zp: int = 0
    dtype: str = "int8"          # 'int8' | 'float' (graph inputs before quantize)
    act_min: int = -128          # value range actually produced (fused ReLU / softmax)
    act_max: int = 127

    @property
    def uniform(self):
        return np.ndim(self.scale) == 0

    def dequant(self, q):
        return (np.asarray(q, np.float64) - self.zp) * np.asarray(self.scale, np.float64)


@dataclass
class Node:
    name: str
    op: str
    inputs: list
    output: str
    attrs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)     # exported numpy arrays
    meta: dict = field(default_factory=dict)       # float-side info (not needed by a compiler)


# ------------------------------------------------------------------------------ shapes -------
def sym_dim(v2, v3):
    """Infer a symbolic dim from its values at batch 2 and 3."""
    if v2 == v3:
        return int(v2)
    f = v3 - v2
    if v2 != 2 * f:
        raise ValueError(f"cannot express dim ({v2}, {v3}) as N*k")
    return "N" if f == 1 else f"N*{f}"


def resolve_shape(shape, N):
    out = []
    for d in shape:
        if isinstance(d, str):
            out.append(N * (int(d.split("*")[1]) if "*" in d else 1))
        else:
            out.append(int(d))
    return out


def batch_of(shape, actual):
    for d, a in zip(shape, actual):
        if isinstance(d, str):
            return a // (int(d.split("*")[1]) if "*" in d else 1)
    return 1


def index_to_json(idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    out = []
    for i in idx:
        if i is None:
            out.append({"newaxis": True})
        elif i is Ellipsis:
            out.append("...")
        elif isinstance(i, slice):
            out.append({"slice": [i.start, i.stop, i.step]})
        elif isinstance(i, (int, np.integer)):
            out.append(int(i))
        else:
            raise NotImplementedError(f"getitem index {i!r}")
    return out


def index_from_json(lst):
    out = []
    for i in lst:
        if i == "...":
            out.append(Ellipsis)
        elif isinstance(i, dict) and "slice" in i:
            out.append(slice(*i["slice"]))
        elif isinstance(i, dict):
            out.append(None)
        else:
            out.append(int(i))
    return tuple(out)


def apply_layout(op, attrs, x, N):
    if op == "reshape":
        return np.reshape(x, resolve_shape(attrs["shape"], N))
    if op == "permute":
        return np.transpose(x, attrs["perm"])
    if op == "getitem":
        return x[index_from_json(attrs["index"])]
    raise ValueError(op)


def compress_scale(s):
    """Reduce a broadcast scale array to its minimal form (scalar if uniform)."""
    s = np.asarray(s, np.float64)
    if s.size == 0 or np.all(s == s.flat[0]):
        return float(s.flat[0]) if s.size else 1.0
    for ax in range(s.ndim):
        if s.shape[ax] > 1 and np.all(s == s.take([0], axis=ax)):
            s = s.take([0], axis=ax)
    return s


def im2col_nchw(x, kh, kw, stride, pad, pad_val):
    """x [B, C, H, W] -> cols [B, OH, OW, kh*kw*C] in (kh, kw, cin) order (HWIO weights)."""
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), constant_values=pad_val)
    OH = (H + 2 * pad[0] - kh) // stride[0] + 1
    OW = (W + 2 * pad[1] - kw) // stride[1] + 1
    cols = np.empty((B, OH, OW, kh, kw, C), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = xp[:, :, i:i + stride[0] * OH:stride[0],
                                         j:j + stride[1] * OW:stride[1]].transpose(0, 2, 3, 1)
    return cols.reshape(B, OH, OW, kh * kw * C)


def pool_windows(x, k, stride, pad, pad_val):
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), constant_values=pad_val)
    OH = (H + 2 * pad[0] - k[0]) // stride[0] + 1
    OW = (W + 2 * pad[1] - k[1]) // stride[1] + 1
    out = None
    for i in range(k[0]):
        for j in range(k[1]):
            v = xp[:, :, i:i + stride[0] * OH:stride[0], j:j + stride[1] * OW:stride[1]]
            out = v if out is None else np.maximum(out, v)
    return out


def _mm(a, b):
    """Float matmul on contiguous copies (numpy's batched matmul is very slow on strided views)."""
    return np.matmul(np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64))


# ------------------------------------------------------------------------------- graph -------
class Graph:
    def __init__(self, name="model"):
        self.name = name
        self.tensors = {}
        self.nodes = []
        self.inputs = []          # float graph-input tensor names
        self.outputs = []         # int8 tensor names
        self.output_post = {}     # name -> {"scale": arr, "bias": arr} host-side affine after dequant
        self.meta = {}

    # -- construction --
    def add_tensor(self, t):
        assert t.name not in self.tensors, t.name
        self.tensors[t.name] = t
        return t

    def add_node(self, n):
        self.nodes.append(n)
        return n

    def node(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def weight_nodes(self):
        return [n for n in self.nodes if n.op in WEIGHT_OPS]

    # -- integer interpreter --
    def run_int(self, inputs, keep_all=False):
        """inputs: dict name -> float array (or a single array for a 1-input graph).
        Returns dict tensor -> int64 array (outputs only unless keep_all)."""
        if not isinstance(inputs, dict):
            inputs = {self.inputs[0]: inputs}
        env = {k: np.asarray(v, np.float64) for k, v in inputs.items()}
        t0 = self.tensors[self.inputs[0]]
        N = batch_of(t0.shape, env[self.inputs[0]].shape)
        last_use = {}
        for i, n in enumerate(self.nodes):
            for x in n.inputs:
                last_use[x] = i
        for i, n in enumerate(self.nodes):
            env[n.output] = self._exec_int(n, [env[x] for x in n.inputs], N)
            if not keep_all:
                for x in n.inputs:
                    if last_use.get(x) == i and x not in self.outputs:
                        env.pop(x, None)
        return env if keep_all else {k: env[k] for k in self.outputs}

    def _exec_int(self, n, xs, N):
        a, p, op = n.attrs, n.params, n.op
        to = self.tensors[n.output]
        if op == "quantize":
            return quantize_np(xs[0], to.scale, to.zp)
        if op == "const":
            return p["value"].astype(np.int64)
        if op in LAYOUT_OPS:
            return apply_layout(op, a, xs[0], N)
        if op == "linear":
            acc = I.matmul(xs[0], p["w"])
            return I.requant(acc, p["bias"], p["m0"], p["shift"], a["zp_out"], a["act_min"], a["act_max"])
        if op == "conv2d":
            kh, kw, cin, cout = p["w"].shape
            cols = im2col_nchw(xs[0], kh, kw, a["stride"], a["padding"], a["zp_in"])
            acc = I.matmul(cols, p["w"].reshape(-1, cout))
            y = I.requant(acc, p["bias"], p["m0"], p["shift"], a["zp_out"], a["act_min"], a["act_max"])
            return y.transpose(0, 3, 1, 2)
        if op == "matmul":
            acc = I.matmul(xs[0], xs[1])
            return I.requant(acc, 0, a["m0"], a["shift"], a["zp_out"], a["act_min"], a["act_max"])
        if op == "layernorm":
            return I.layernorm(xs[0])
        if op == "affine":
            x = np.moveaxis(xs[0], a["axis"], -1)
            y = I.requant(x * p["w"], p["bias"], p["m0"], p["shift"], a["zp_out"], a["act_min"], a["act_max"])
            return np.moveaxis(y, -1, a["axis"])
        if op == "softmax":
            return I.softmax(xs[0], p["lut"], a.get("out_mult", I.SM_OUT))
        if op == "lut":
            return I.lut8(xs[0], p["lut"])
        if op == "add":
            return I.ew_add(xs[0], xs[1], a["ma"], a["mb"], a["za"], a["zb"], a["zo"])
        if op == "rescale":
            return I.rescale(xs[0], p["m"], a["za"], a["zo"])
        if op == "mean":
            acc = (xs[0] - a["zp_in"]).sum(axis=tuple(a["axes"]), keepdims=a["keepdim"])
            return I.requant(acc, 0, a["m0"], a["shift"], a["zp_out"])
        if op == "maxpool2d":
            return pool_windows(xs[0], a["kernel"], a["stride"], a["padding"], -128)
        raise NotImplementedError(op)

    def dequant(self, name, q):
        y = self.tensors[name].dequant(q)
        post = self.output_post.get(name)
        if post is not None:
            y = y * post["scale"] + post["bias"]
        return y

    def run(self, inputs):
        """Integer execution, returns dequantised float outputs (list)."""
        out = self.run_int(inputs)
        return [self.dequant(k, out[k]) for k in self.outputs]

    # -- float / fake-quant interpreter --
    def run_float(self, inputs, quant_acts=None, quant_weights=None, weights_override=None):
        """Float execution of the graph. quant_acts / quant_weights: None (no quantisation),
        'all', or a set of tensor / node names to fake-quantise. weights_override: node -> w_deq.
        Returns dict of every tensor (float)."""
        if not isinstance(inputs, dict):
            inputs = {self.inputs[0]: inputs}
        env = {k: np.asarray(v, np.float64) for k, v in inputs.items()}
        t0 = self.tensors[self.inputs[0]]
        N = batch_of(t0.shape, env[self.inputs[0]].shape)
        qa = (lambda t: True) if quant_acts == "all" else (lambda t: bool(quant_acts) and t in quant_acts)
        qw = (lambda t: True) if quant_weights == "all" else (lambda t: bool(quant_weights) and t in quant_weights)
        wo = weights_override or {}
        for n in self.nodes:
            xs = [env[x] for x in n.inputs]
            y = self._exec_float(n, xs, N, qw(n.name), wo.get(n.name))
            to = self.tensors[n.output]
            if n.op not in LAYOUT_OPS and n.op != "const" and qa(n.output):
                lo, hi = to.act_min, to.act_max
                y = (np.clip(np.round(y / to.scale) + to.zp, lo, hi) - to.zp) * to.scale
            elif n.op not in LAYOUT_OPS and n.meta.get("clamp") is not None:
                y = np.clip(y, *n.meta["clamp"])
            env[n.output] = y
        for k in self.outputs:
            post = self.output_post.get(k)
            if post is not None:
                env[k + ":post"] = env[k] * post["scale"] + post["bias"]
        return env

    def _exec_float(self, n, xs, N, qw, w_override):
        a, m, op = n.attrs, n.meta, n.op
        if op == "quantize":
            return xs[0]
        if op == "const":
            return np.asarray(m["value_float"], np.float64)
        if op in LAYOUT_OPS:
            return apply_layout(op, a, xs[0], N)
        if op in ("linear", "conv2d"):
            W = w_override if w_override is not None else (m["w_deq"] if qw else m["w_float"])
            if op == "linear":
                y = _mm(xs[0], W) + m["b_float"]
            else:
                kh, kw, cin, cout = n.params["w"].shape if "w" in n.params else m["w_shape"]
                cols = im2col_nchw(xs[0], kh, kw, a["stride"], a["padding"], 0.0)
                y = (_mm(cols, W.reshape(-1, cout)) + m["b_float"]).transpose(0, 3, 1, 2)
            return y * m.get("post_scale", 1.0)
        if op == "matmul":
            return _mm(xs[0], xs[1]) * m.get("post_scale", 1.0)
        if op == "layernorm":
            x = xs[0]; mu = x.mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + m.get("eps", 1e-5))
        if op == "affine":
            x = np.moveaxis(xs[0], a["axis"], -1)
            return np.moveaxis(x * m["g_float"] + m["b_float"], -1, a["axis"])
        if op == "softmax":
            x = xs[0] - xs[0].max(-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(-1, keepdims=True)
        if op == "lut":
            return I.FLOAT_FUNCS[a["fn"]](xs[0])
        if op == "add":
            return xs[0] + xs[1]
        if op == "rescale":
            return xs[0]
        if op == "mean":
            return xs[0].mean(axis=tuple(a["axes"]), keepdims=a["keepdim"])
        if op == "maxpool2d":
            return pool_windows(xs[0], a["kernel"], a["stride"], a["padding"], -np.inf)
        raise NotImplementedError(op)

    def float_outputs(self, env):
        return [env.get(k + ":post", env[k]) for k in self.outputs]

    # -- info --
    def summary(self):
        lines = [f"graph {self.name}: {len(self.nodes)} nodes"]
        for n in self.nodes:
            t = self.tensors[n.output]
            sc = f"{t.scale:.4g}" if t.uniform else f"[{np.min(t.scale):.3g}..{np.max(t.scale):.3g}]"
            extra = ""
            if n.op in WEIGHT_OPS:
                extra = f" w{n.attrs.get('weight_bits', 8)}:{n.attrs.get('weight_scheme', 'channel')}"
            lines.append(f"  {n.op:10s} {n.name:28s} <- {','.join(n.inputs):40s} s={sc} zp={t.zp}{extra}")
        return "\n".join(lines)

    def weight_bytes(self):
        tot = 0
        for n in self.weight_nodes():
            bits = n.attrs.get("weight_bits", 8)
            tot += math.ceil(n.params["w"].size * bits / 8)
            if "gmul" in n.params:
                tot += math.ceil(n.params["gmul"].size * 4 / 8)
        return tot
