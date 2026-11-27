"""Generic post-training quantisation of PyTorch models into the tpuquant integer graph.

    cal = Calibrator(model, calib_inputs)          # torch.fx trace + 2-pass activation stats
    g   = cal.build(QConfig(act_method="mse"))     # lower to a bit-exact integer Graph
    y   = g.run(x)                                 # integer execution, dequantised outputs

Supported (after torch.fx tracing): nn.Linear, nn.Conv2d (groups=1), nn.LayerNorm (last dim),
GELU / ReLU / ReLU6 / Sigmoid / Tanh / SiLU (module or functional), softmax(dim=-1),
tensor @ tensor (attention), residual add (tensor + tensor or + parameter), scalar mul/div
(folded into the producer's requant), mean / AdaptiveAvgPool2d(1), MaxPool2d, and the layout
ops view/reshape/flatten/squeeze/unsqueeze/permute/transpose/getitem/contiguous/dropout.
Unsupported ops raise NotImplementedError naming the fx node.
"""
import inspect
import math
import operator
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from . import intops as I
from .graph import (Graph, Node, TensorInfo, apply_layout, compress_scale, index_to_json,
                    resolve_shape, sym_dim)
from .observers import Observer, make_observer
from .qscheme import act_qparams, quantize_weight, quantize_np


@dataclass
class QConfig:
    act_method: str = "percentile"       # minmax | percentile[:p] | mse
    act_symmetric: bool = True           # False -> TFLite asymmetric (zp != 0) where allowed
    weight_bits: int = 8                 # default weight bits (8 or 4)
    layer_bits: dict = field(default_factory=dict)   # fx node name -> bits override
    int4_scheme: str = "group_int"       # channel | group_int | group_pow2 | group_float
    group_size: int = 32
    weight_method: str = "minmax"        # minmax | mse (per-channel clip search)
    split_scales: bool = True            # per-channel output scales when a linear's output is
                                         # split by layout ops (e.g. fused QKV -> q, k, v)
    fold_ln: bool = True                 # fold LayerNorm gamma/beta into consuming linears
    output_affine: str = "host"          # final-LN affine: 'host' (after dequant) | 'int'
    method_overrides: dict = field(default_factory=dict)   # fx node name -> observer method
    softmax_mult: object = None          # None: probs at 1/127 (jepa spec); int: 1/int;
                                         # observer name (e.g. 'percentile:99.99'): calibrated
    fixed_qparams: dict = field(default_factory=dict)      # fx node name -> (scale, zp) (imports)

    def bits_for(self, name):
        return self.layer_bits.get(name, self.weight_bits)


# ------------------------------------------------------------------------------ fx helpers ----
_LAYOUT_METHODS = {"view", "reshape", "flatten", "squeeze", "unsqueeze", "permute", "transpose",
                   "unflatten", "view_as", "reshape_as"}
_LAYOUT_FUNCS = {torch.reshape, torch.flatten, torch.squeeze, torch.unsqueeze, torch.permute,
                 torch.transpose}
_IDENTITY_METHODS = {"contiguous", "clone", "detach", "float"}
_ACT_MODULES = {nn.GELU: "gelu", nn.ReLU: "relu", nn.ReLU6: "relu6", nn.Sigmoid: "sigmoid",
                nn.Tanh: "tanh", nn.SiLU: "silu"}
_ACT_FUNCS = {F.gelu: "gelu", F.relu: "relu", torch.relu: "relu", F.relu6: "relu6",
              torch.sigmoid: "sigmoid", F.sigmoid: "sigmoid", torch.tanh: "tanh", F.tanh: "tanh",
              F.silu: "silu"}
_ACT_METHODS = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh"}
_SHAPE_ACCESS = {"shape", "size", "dim"}


class _Recorder(fx.Interpreter):
    def __init__(self, gm, hook):
        super().__init__(gm)
        self.hook = hook

    def run_node(self, n):
        v = super().run_node(n)
        self.hook(n, v)
        return v


def _to_np(t):
    return t.detach().double().cpu().numpy()


class Calibrator:
    """Trace a model with torch.fx and collect activation statistics on calibration data."""

    def __init__(self, model, calib, concrete_args=None, batch_size=64, name=None):
        model = model.eval()
        self.model = model
        self.gm = fx.symbolic_trace(model, concrete_args=concrete_args)
        self.modules = dict(self.gm.named_modules())
        self.name = name or type(model).__name__
        calib = calib if isinstance(calib, (tuple, list)) else (calib,)
        self.calib = calib
        self.nodes = list(self.gm.graph.nodes)
        # graph inputs = the first len(calib) forward() parameters; the rest get their concrete
        # (or default) value when the traced module is interpreted
        sig = list(inspect.signature(model.forward).parameters.values())
        ca = concrete_args or {}
        self.extra = [ca[p.name] if p.name in ca else p.default for p in sig[len(calib):]
                      if p.kind == p.POSITIONAL_OR_KEYWORD]
        self.placeholders = [n for n in self.nodes if n.op == "placeholder"][:len(calib)]
        self._probe()
        self._calibrate(batch_size)

    # -- tracing helpers --
    def run(self, *xs, hook=None):
        with torch.no_grad():
            return _Recorder(self.gm, hook or (lambda n, v: None)).run(*xs, *self.extra)

    def _batch(self, a, b):
        return tuple(c[a:b] for c in self.calib)

    def _probe(self):
        rec2, rec3 = {}, {}
        self.run(*self._batch(0, 2), hook=lambda n, v: rec2.__setitem__(n.name, v))
        self.run(*self._batch(0, 3), hook=lambda n, v: rec3.__setitem__(n.name, v))
        self.is_tensor, self.shape, self.value = {}, {}, {}
        for n in self.nodes:
            v2, v3 = rec2.get(n.name), rec3.get(n.name)
            if isinstance(v2, torch.Tensor):
                self.is_tensor[n.name] = True
                self.shape[n.name] = [sym_dim(a, b) for a, b in zip(v2.shape, v3.shape)]
            else:
                self.is_tensor[n.name] = False
                self.value[n.name] = v2 if v2 == v3 else ("batch-dependent", v2)
        self.observed = {n.name for n in self.nodes if self.is_tensor.get(n.name)
                         and not (self.kind(n) in ("layout", "identity") and not self._is_terminal(n))}

    def _calibrate(self, bs):
        self.stats = {k: Observer() for k in self.observed}
        M = len(self.calib[0])
        for pass_ in ("range", "hist"):
            for i in range(0, M, bs):
                def hook(n, v, pass_=pass_):
                    if n.name in self.stats:
                        o = self.stats[n.name]
                        (o.observe_range if pass_ == "range" else o.observe_hist)(_to_np(v))
                self.run(*self._batch(i, i + bs), hook=hook)

    # -- node classification --
    def real_users(self, n):
        return [u for u in n.users if not (u.op == "call_function" and u.target is getattr
                                           and u.args[1] in _SHAPE_ACCESS)
                and not (u.op == "call_method" and u.target in _SHAPE_ACCESS)]

    def kind(self, n):
        if n.op == "placeholder":
            return "input"
        if n.op == "get_attr":
            return "param"
        if n.op == "output":
            return "output"
        if not self.is_tensor.get(n.name, False):
            return "scalar"
        t = n.target
        if n.op == "call_module":
            m = self.modules[t]
            if isinstance(m, nn.Linear): return "linear"
            if isinstance(m, nn.Conv2d): return "conv2d"
            if isinstance(m, nn.LayerNorm): return "layernorm"
            if isinstance(m, (nn.Dropout, nn.Identity)): return "identity"
            if isinstance(m, nn.Softmax): return "softmax"
            if isinstance(m, nn.MaxPool2d): return "maxpool2d"
            if isinstance(m, nn.AdaptiveAvgPool2d): return "mean"
            if isinstance(m, nn.Flatten): return "layout"
            for c, k in _ACT_MODULES.items():
                if isinstance(m, c): return "act:" + k
            return "unsupported"
        if n.op == "call_method":
            if t in _LAYOUT_METHODS: return "layout"
            if t in _IDENTITY_METHODS: return "identity"
            if t == "softmax": return "softmax"
            if t == "matmul": return "matmul"
            if t == "add": return "add"
            if t in ("mul", "div"): return t
            if t == "mean": return "mean"
            if t in _ACT_METHODS: return "act:" + _ACT_METHODS[t]
            return "unsupported"
        if t in _ACT_FUNCS: return "act:" + _ACT_FUNCS[t]
        if t is operator.getitem: return "layout"
        if t in _LAYOUT_FUNCS: return "layout"
        if t in (operator.matmul, torch.matmul, torch.bmm): return "matmul"
        if t in (operator.add, torch.add, operator.iadd): return "add"
        if t in (operator.mul, torch.mul): return "mul"
        if t in (operator.truediv, torch.div): return "div"
        if t in (F.softmax, torch.softmax): return "softmax"
        if t is F.layer_norm: return "layernorm"
        if t is F.dropout: return "identity"
        if t is torch.mean: return "mean"
        if t is F.max_pool2d: return "maxpool2d"
        return "unsupported"

    def _is_terminal(self, n):
        return any(self.kind(u) not in ("layout", "identity") for u in self.real_users(n))

    def build(self, cfg=None):
        return _Lowering(self, cfg or QConfig()).run()


# ------------------------------------------------------------------------------- lowering ----
class _Lowering:
    def __init__(self, cal, cfg):
        self.c, self.cfg = cal, cfg
        self.g = Graph(cal.name)
        self.env = {}               # fx name -> IR tensor name | python value
        self.skip = set()           # fx nodes absorbed into a producer
        self.ln_fold = {}           # fx LN name -> (gamma, beta)
        self.need_sym = self._need_sym()
        self.g.meta["qconfig"] = {k: v for k, v in cfg.__dict__.items()}
        self.g.meta["int_exact"] = True

    # -- qparams --
    def _sym(self, name):
        return self.cfg.act_symmetric or name in self.need_sym

    def _need_sym(self):
        """Tensors feeding act x act matmuls must have zp 0 (the 'weight' side has no zp)."""
        out = set()
        for n in self.c.nodes:
            if self.c.kind(n) == "matmul":
                for a in n.args[:2]:
                    while isinstance(a, fx.Node):
                        out.add(a.name)
                        if self.c.kind(a) in ("layout", "identity"):
                            a = a.args[0]
                        else:
                            break
        return out

    def qp(self, fxname, sym=None):
        if fxname in self.cfg.fixed_qparams:
            s, zp = self.cfg.fixed_qparams[fxname]
            return float(s), int(zp)
        st = self.c.stats[fxname]
        o = make_observer(self.cfg.method_overrides.get(fxname, self.cfg.act_method))
        o.__dict__.update({k: getattr(st, k) for k in ("min", "max", "hist", "count", "nbins")})
        sym = self._sym(fxname) if sym is None else sym
        return act_qparams(*o.range(sym), symmetric=sym)

    def T(self, fxnode):
        return self.g.tensors[self.env[fxnode.name]]

    def val(self, a):
        if isinstance(a, fx.Node):
            if self.c.is_tensor.get(a.name):
                return a
            v = self.c.value[a.name]
            return v
        if isinstance(a, (tuple, list)):
            return type(a)(self.val(x) for x in a)
        return a

    def new_tensor(self, name, shape, scale, zp, lo=-128, hi=127, dtype="int8"):
        return self.g.add_tensor(TensorInfo(name, list(shape), scale, int(zp), dtype, int(lo), int(hi)))

    def uniform(self, fxnode):
        """IR tensor name for fxnode, rescaled to a single scale if it carries a scale array."""
        t = self.T(fxnode)
        if t.uniform:
            return t.name
        s_out = float(np.max(t.scale))
        name = f"{t.name}_rs"
        if name not in self.g.tensors:
            m = I.ew_mult(np.asarray(t.scale), s_out)
            self.new_tensor(name, t.shape, s_out, t.zp)
            self.g.add_node(Node(name, "rescale", [t.name], name, dict(za=t.zp, zo=t.zp),
                                 dict(m=np.asarray(m, np.int64))))
        return name

    # -- chain absorption: relu / relu6 / scalar mul-div after a requantising producer --
    def absorb(self, n):
        end, post, lo, hi = n, 1.0, -np.inf, np.inf
        while True:
            us = self.c.real_users(end)
            if len(us) != 1 or len(end.users) and any(u.op == "output" for u in end.users):
                break
            u = us[0]; k = self.c.kind(u)
            if k in ("act:relu", "act:relu6"):
                lo = max(lo, 0.0)
                if k == "act:relu6":
                    hi = min(hi, 6.0)
            elif k in ("mul", "div"):
                other = [a for a in u.args if a is not end]
                if len(other) != 1:
                    break
                v = self.val(other[0])
                if isinstance(v, fx.Node) or not isinstance(v, (int, float)) or v <= 0 or (k == "div" and u.args[0] is not end):
                    break
                f = float(v) if k == "mul" else 1.0 / float(v)
                post *= f; lo *= f; hi *= f
            else:
                break
            self.skip.add(u.name)
            end = u
        return end, post, (lo, hi)

    def out_qparams(self, end, chan_axis, shape):
        """Output qparams for a requantising producer whose float value is fx node `end`."""
        s, zp = self.qp(end.name)
        if chan_axis is None or not self.cfg.split_scales or zp != 0:
            return s, zp
        # follow layout ops to every terminal consumer, give each element the max terminal scale
        shp1 = resolve_shape(shape, 1)
        ids = np.arange(int(np.prod(shp1))).reshape(shp1)
        s_el = np.zeros(ids.size)
        stack = [(end, ids)]
        any_split = False
        while stack:
            nd, a = stack.pop()
            if self.c._is_terminal(nd):
                st = s if nd is end else self.qp(nd.name)[0]
                s_el[a.ravel()] = np.maximum(s_el[a.ravel()], st)
            for u in self.c.real_users(nd):
                k = self.c.kind(u)
                if k == "identity":
                    stack.append((u, a))
                elif k == "layout":
                    op, attrs = self.layout_attrs(u, nd)
                    stack.append((u, apply_layout(op, attrs, a, 1)))
                    any_split = True
        if not any_split:
            return s, zp
        s_el[s_el == 0] = s
        sc = compress_scale(s_el.reshape(shp1))
        if np.ndim(sc) == 0:
            return sc, 0
        ax = chan_axis % len(shp1)
        if any(sc.shape[i] != 1 for i in range(sc.ndim) if i != ax):
            sc = np.max(sc, axis=tuple(i for i in range(sc.ndim) if i != ax), keepdims=True)
        return sc, 0

    def layout_attrs(self, u, src):
        k_shape = self.c.shape[u.name]
        if u.op == "call_function" and u.target is operator.getitem:
            return "getitem", dict(index=index_to_json(self.val(u.args[1])))
        t = u.target
        nd = len(self.c.shape[src.name])
        if t in ("permute", torch.permute):
            dims = u.args[1:] if t == "permute" else u.args[1]
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = dims[0]
            return "permute", dict(perm=[int(d) % nd for d in self.val(list(dims))])
        if t in ("transpose", torch.transpose):
            a, b = (int(self.val(x)) % nd for x in u.args[1:3])
            perm = list(range(nd)); perm[a], perm[b] = perm[b], perm[a]
            return "permute", dict(perm=perm)
        return "reshape", dict(shape=list(k_shape))

    # -- main --
    def run(self):
        c, g = self.c, self.g
        for n in c.nodes:
            if n.name in self.skip:
                continue
            k = c.kind(n)
            h = getattr(self, "lower_" + k.split(":")[0], None)
            if k == "scalar":
                continue
            if h is None:
                raise NotImplementedError(f"tpuquant: unsupported fx node {n.op} {n.name} target={n.target}")
            h(n, k)
        return g

    def lower_input(self, n, k):
        if n.name not in [p.name for p in self.c.placeholders]:
            return
        shape = self.c.shape[n.name]
        self.g.add_tensor(TensorInfo(n.name, shape, 1.0, 0, "float"))
        self.g.inputs.append(n.name)
        s, zp = self.qp(n.name)
        q = f"{n.name}_q"
        self.new_tensor(q, shape, s, zp)
        self.g.add_node(Node(q, "quantize", [n.name], q))
        self.env[n.name] = q

    def lower_param(self, n, k):
        obj = self.c.gm
        for part in n.target.split("."):
            obj = getattr(obj, part)
        w = _to_np(obj)
        s, zp = self.qp(n.name)
        q = quantize_np(w, s, zp)
        self.new_tensor(n.name, self.c.shape[n.name], s, zp)
        self.g.add_node(Node(n.name, "const", [], n.name, {}, dict(value=q.astype(np.int8)),
                             dict(value_float=w)))
        self.env[n.name] = n.name

    def lower_identity(self, n, k):
        self.env[n.name] = self.env[n.args[0].name]

    def lower_layout(self, n, k):
        src = n.args[0]
        t = self.T(src)
        op, attrs = self.layout_attrs(n, src)
        if t.uniform:
            sc = t.scale
        else:
            full = np.broadcast_to(np.asarray(t.scale), resolve_shape(t.shape, 1))
            sc = compress_scale(apply_layout(op, attrs, full, 1))
        self.new_tensor(n.name, self.c.shape[n.name], sc, t.zp, t.act_min, t.act_max)
        self.g.add_node(Node(n.name, op, [t.name], n.name, attrs))
        self.env[n.name] = n.name

    def _weight_layer(self, n, k, W, b, xname, chan_axis, extra_attrs, wshape_fn):
        cfg = self.cfg
        tx = self.g.tensors[xname]
        s_in, zp_in = tx.scale, tx.zp
        end, post, (flo, fhi) = self.absorb(n)
        shape = self.c.shape[end.name]
        s_out, zp_out = self.out_qparams(end, chan_axis, shape)
        bits = cfg.bits_for(n.name)
        qw = quantize_weight(W, bits, cfg.int4_scheme, cfg.group_size, cfg.weight_method)
        sw = qw["scale"]
        if qw["w"] is None:          # group_float: integer path uses the INT8 projection
            self.g.meta["int_exact"] = False
            sw = np.maximum(np.abs(qw["w_deq"]).max(0), 1e-8) / 127
            w = np.clip(np.round(qw["w_deq"] / sw), -127, 127).astype(np.int64)
        else:
            w = qw["w"]
        s_out_c = np.broadcast_to(np.asarray(s_out, np.float64).reshape(-1), (W.shape[1],)) \
            if np.ndim(s_out) else np.full(W.shape[1], s_out)
        bq = np.round(b / (s_in * sw)).astype(np.int64)
        bias_hw = bq - zp_in * w.sum(0)
        if np.abs(bias_hw).max() >= 2 ** 31:
            raise OverflowError(f"{n.name}: bias does not fit int32")
        m0, sh = I.qmul_vec(s_in * sw * post / s_out_c)
        lo = -128 if flo == -np.inf else max(-128, zp_out + int(np.round(flo / s_out_c.min())))
        hi = 127 if fhi == np.inf else min(127, zp_out + int(np.round(fhi / s_out_c.min())))
        out = end.name
        self.new_tensor(out, shape, s_out, zp_out, lo, hi)
        attrs = dict(zp_in=int(zp_in), zp_out=int(zp_out), act_min=int(lo), act_max=int(hi),
                     weight_bits=int(bits), weight_scheme=qw["scheme"], **extra_attrs)
        params = dict(w=wshape_fn(w).astype(np.int8), bias=bias_hw.astype(np.int32),
                      m0=m0.astype(np.int32), shift=sh.astype(np.int8),
                      w_scale=np.asarray(sw, np.float32))
        if bits == 4:
            attrs["group_size"] = int(qw.get("group_size", 0))
            params["w4"] = wshape_fn(qw["q4"]).astype(np.int8)
            if "gmul" in qw:
                params["gmul"] = qw["gmul"].astype(np.uint8)
        meta = dict(w_float=W, b_float=b, w_deq=qw["w_deq"], post_scale=post,
                    clamp=None if (flo, fhi) == (-np.inf, np.inf) else (flo, fhi), fx=n.name,
                    w_shape=wshape_fn(w).shape)
        self.g.add_node(Node(n.name, k, [xname], out, attrs, params, meta))
        self.env[end.name] = out

    def lower_linear(self, n, k):
        m = self.c.modules[n.target]
        src = n.args[0]
        W = _to_np(m.weight).T.copy()
        b = _to_np(m.bias) if m.bias is not None else np.zeros(W.shape[1])
        if src.name in self.ln_fold:
            gma, bta = self.ln_fold[src.name]
            b = b + bta @ W
            W = gma[:, None] * W
        self._weight_layer(n, "linear", W, b, self.uniform(src), -1, {}, lambda w: w)

    def lower_conv2d(self, n, k):
        m = self.c.modules[n.target]
        if m.groups != 1 or tuple(m.dilation) != (1, 1) or isinstance(m.padding, str) \
                or m.padding_mode != "zeros":
            raise NotImplementedError(f"{n.name}: only groups=1, dilation=1, numeric zero padding")
        Wt = _to_np(m.weight)                                   # [cout, cin, kh, kw]
        cout, cin, kh, kw = Wt.shape
        W = Wt.transpose(2, 3, 1, 0).reshape(-1, cout)          # (kh, kw, cin) x cout
        b = _to_np(m.bias) if m.bias is not None else np.zeros(cout)
        self._weight_layer(n, "conv2d", W, b, self.uniform(n.args[0]), 1,
                           dict(stride=list(m.stride), padding=list(m.padding)),
                           lambda w: w.reshape(kh, kw, cin, cout))

    def lower_matmul(self, n, k):
        a, b = n.args[:2]
        ta, tb = self.g.tensors[self.uniform(a)], self.g.tensors[self.uniform(b)]
        if ta.zp or tb.zp:
            raise ValueError(f"{n.name}: act x act matmul needs symmetric operands")
        end, post, (flo, fhi) = self.absorb(n)
        s_out, zp_out = self.qp(end.name)
        m0, sh = I.quantize_multiplier(ta.scale * tb.scale * post / s_out)
        lo = -128 if flo == -np.inf else max(-128, zp_out + int(round(flo / s_out)))
        hi = 127 if fhi == np.inf else min(127, zp_out + int(round(fhi / s_out)))
        self.new_tensor(end.name, self.c.shape[end.name], s_out, zp_out, lo, hi)
        self.g.add_node(Node(n.name, "matmul", [ta.name, tb.name], end.name,
                             dict(m0=int(m0), shift=int(sh), zp_out=int(zp_out), act_min=lo, act_max=hi),
                             {}, dict(post_scale=post, fx=n.name,
                                      clamp=None if (flo, fhi) == (-np.inf, np.inf) else (flo, fhi))))
        self.env[end.name] = end.name

    def lower_layernorm(self, n, k):
        if n.op == "call_module":
            m = self.c.modules[n.target]
            if len(m.normalized_shape) != 1:
                raise NotImplementedError(f"{n.name}: LayerNorm over the last dim only")
            eps = m.eps
            g_, b_ = (_to_np(m.weight), _to_np(m.bias)) if m.elementwise_affine else (None, None)
            if b_ is None and g_ is not None:
                b_ = np.zeros_like(g_)
        else:
            eps = n.kwargs.get("eps", 1e-5)
            w = n.args[2] if len(n.args) > 2 else n.kwargs.get("weight")
            if w is not None:
                raise NotImplementedError(f"{n.name}: functional layer_norm with affine")
            g_ = b_ = None
        x = self.uniform(n.args[0])
        name = f"{n.name}_norm" if g_ is not None else n.name
        self.new_tensor(name, self.c.shape[n.name], 1.0 / I.LN_Q, 0)
        self.g.add_node(Node(name, "layernorm", [x], name, {}, {}, dict(eps=eps, fx=n.name)))
        self.env[n.name] = name
        if g_ is None:
            return
        users = self.c.real_users(n)
        if self.cfg.fold_ln and users and all(self.c.kind(u) == "linear" for u in users):
            self.ln_fold[n.name] = (g_, b_)
            return
        if self.cfg.output_affine == "host" and users and all(u.op == "output" for u in users):
            self.g.output_post[name] = dict(scale=g_, bias=b_)
            return
        s_out, zp_out = self.qp(n.name)
        s_in = 1.0 / I.LN_Q
        sw = np.maximum(np.abs(g_), 1e-8) / 127
        wq = np.clip(np.round(g_ / sw), -127, 127).astype(np.int64)
        bq = np.round(b_ / (s_in * sw)).astype(np.int64)
        m0, sh = I.qmul_vec(s_in * sw / s_out)
        self.new_tensor(n.name, self.c.shape[n.name], s_out, zp_out)
        self.g.add_node(Node(n.name, "affine", [name], n.name,
                             dict(axis=-1, zp_in=0, zp_out=int(zp_out), act_min=-128, act_max=127),
                             dict(w=wq.astype(np.int8), bias=bq.astype(np.int32),
                                  m0=m0.astype(np.int32), shift=sh.astype(np.int8)),
                             dict(g_float=g_, b_float=b_, fx=n.name)))
        self.env[n.name] = n.name

    def lower_softmax(self, n, k):
        if n.op == "call_module":
            dim = self.c.modules[n.target].dim
        else:
            dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else None)
        nd = len(self.c.shape[n.name])
        if dim is None or int(dim) % nd != nd - 1:
            raise NotImplementedError(f"{n.name}: softmax over the last dim only")
        x = self.T(n.args[0]) if self.T(n.args[0]).uniform else self.g.tensors[self.uniform(n.args[0])]
        sm = self.cfg.softmax_mult
        if sm is None:
            mult = I.SM_OUT
        elif isinstance(sm, (int, np.integer)):
            mult = int(sm)
        else:
            st = self.c.stats[n.name]
            o = make_observer(sm)
            o.__dict__.update({k: getattr(st, k) for k in ("min", "max", "hist", "count", "nbins")})
            mult = int(np.clip(np.floor(127 / max(o.range(True)[1], 1e-6)), I.SM_OUT, 4095))
        self.new_tensor(n.name, self.c.shape[n.name], 1.0 / mult, 0, 0, 127)
        self.g.add_node(Node(n.name, "softmax", [x.name], n.name, dict(axis=-1, out_mult=mult),
                             dict(lut=I.exp_lut(x.scale).astype(np.int32)), dict(fx=n.name)))
        self.env[n.name] = n.name

    def lower_act(self, n, k):
        fn = k.split(":")[1]
        if fn == "gelu" and (n.kwargs.get("approximate") == "tanh" or
                             (n.op == "call_module" and getattr(self.c.modules[n.target], "approximate", "none") == "tanh")):
            fn = "gelu_tanh"
        x = self.g.tensors[self.uniform(n.args[0])]
        s_out, zp_out = self.qp(n.name)
        lut = I.make_lut(fn, x.scale, s_out, x.zp, zp_out)
        self.new_tensor(n.name, self.c.shape[n.name], s_out, zp_out, int(lut.min()), int(lut.max()))
        self.g.add_node(Node(n.name, "lut", [x.name], n.name, dict(fn=fn),
                             dict(lut=lut.astype(np.int8)), dict(fx=n.name)))
        self.env[n.name] = n.name

    def lower_add(self, n, k):
        a, b = n.args[:2]
        if not (isinstance(a, fx.Node) and isinstance(b, fx.Node) and self.c.is_tensor[a.name]
                and self.c.is_tensor[b.name]):
            raise NotImplementedError(f"{n.name}: add needs two tensors")
        ta, tb = self.g.tensors[self.uniform(a)], self.g.tensors[self.uniform(b)]
        s, zp = self.qp(n.name)
        self.new_tensor(n.name, self.c.shape[n.name], s, zp)
        self.g.add_node(Node(n.name, "add", [ta.name, tb.name], n.name,
                             dict(ma=I.ew_mult(ta.scale, s), mb=I.ew_mult(tb.scale, s),
                                  za=ta.zp, zb=tb.zp, zo=int(zp)), {}, dict(fx=n.name)))
        self.env[n.name] = n.name

    def lower_mul(self, n, k):
        raise NotImplementedError(f"{n.name}: scalar mul/div is only supported directly after "
                                  "linear/conv/matmul (folded into the requant multiplier)")
    lower_div = lower_mul

    def lower_mean(self, n, k):
        shp = self.c.shape[n.args[0].name]
        nd = len(shp)
        if n.op == "call_module":
            m = self.c.modules[n.target]
            if m.output_size not in (1, (1, 1)):
                raise NotImplementedError("AdaptiveAvgPool2d(1) only")
            axes, keep = [nd - 2, nd - 1], True
        else:
            dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else None)
            keep = bool(n.kwargs.get("keepdim", n.args[2] if len(n.args) > 2 else False))
            dim = list(range(nd)) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
            axes = [int(d) % nd for d in dim]
        if any(isinstance(shp[a], str) for a in axes):
            raise NotImplementedError("mean over the batch dim")
        cnt = int(np.prod([shp[a] for a in axes]))
        x = self.g.tensors[self.uniform(n.args[0])]
        s, zp = self.qp(n.name)
        m0, sh = I.quantize_multiplier(x.scale / (cnt * s))
        self.new_tensor(n.name, self.c.shape[n.name], s, zp)
        self.g.add_node(Node(n.name, "mean", [x.name], n.name,
                             dict(axes=axes, keepdim=keep, m0=int(m0), shift=int(sh),
                                  zp_in=int(x.zp), zp_out=int(zp)), {}, dict(fx=n.name)))
        self.env[n.name] = n.name

    def lower_maxpool2d(self, n, k):
        if n.op == "call_module":
            m = self.c.modules[n.target]
            ks, st, pd = m.kernel_size, m.stride, m.padding
        else:
            ks = n.args[1]; st = n.kwargs.get("stride", n.args[2] if len(n.args) > 2 else None) or ks
            pd = n.kwargs.get("padding", 0)
        two = lambda v: [int(v), int(v)] if isinstance(v, int) else [int(x) for x in v]
        x = self.T(n.args[0])
        self.new_tensor(n.name, self.c.shape[n.name], x.scale, x.zp, x.act_min, x.act_max)
        self.g.add_node(Node(n.name, "maxpool2d", [x.name], n.name,
                             dict(kernel=two(ks), stride=two(st), padding=two(pd))))
        self.env[n.name] = n.name

    def lower_output(self, n, k):
        outs = n.args[0]
        outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
        for o in outs:
            self.g.outputs.append(self.uniform(o))

    def lower_unsupported(self, n, k):
        raise NotImplementedError(f"tpuquant: unsupported fx node {n.op} {n.name} target={n.target}")


def quantize_model(model, calib, cfg=None, concrete_args=None, **kw):
    """One-shot helper: trace + calibrate + lower. Returns (graph, calibrator)."""
    cal = Calibrator(model, calib, concrete_args=concrete_args, **kw)
    return cal.build(cfg), cal
