"""A small dataflow IR for INT8 inference graphs.

Every runtime tensor is 2-D ``[rows, cols]``: rows are tokens / batch items / pixels, cols are
channels. On the TPU a tensor lives in the Unified Buffer "k-tile major": UB row
``base + (c // 16) * rows + r`` holds channels ``16*(c//16) .. +15`` of row ``r`` (pad lanes are
don't-care). Conv feature maps are ``[H*W, C]`` with ``attrs['hw'] = (H, W)``.

Op kinds (``OPS`` below) fall into four classes:

* ``tpu``  - systolic-array work: ``matmul`` / ``conv2d`` (int8 x int8 -> int32) and the ops the
  activation unit can absorb (``requant``, ``clamp``, ``pool2``). After ``passes.fuse`` they
  become ``tpu_matmul`` / ``tpu_conv`` (one op = matmul + requant + activation (+ pool)).
* ``host`` - integer ops the current tpuv1 cannot run: ``layernorm``, ``softmax``, ``lut``
  (GELU), ``add`` (residual), and ``pack_weights`` (turn a runtime tensor into a weight-tile
  stream; inserted by ``passes.legalize`` for activation x activation matmuls).
* ``view`` - zero-cost re-interpretations of UB rows: ``slice`` (16-aligned column range),
  ``concat`` (columns), ``flatten_planar`` ([P, C] feature map -> [1, P*C] in planar order).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

OPS = {
    # kind: (class, fusable-into-activation-unit)
    "matmul": "tpu", "conv2d": "tpu", "requant": "tpu", "clamp": "tpu", "pool2": "tpu",
    "tpu_matmul": "tpu", "tpu_conv": "tpu",
    "layernorm": "host", "softmax": "host", "lut": "host", "add": "host", "pack_weights": "host",
    "slice": "view", "concat": "view", "flatten_planar": "view",
}


@dataclass
class Tensor:
    name: str
    shape: tuple                       # (rows, cols)
    dtype: str = "int8"                # int8 | int32
    const: Optional[np.ndarray] = None  # constant data (weights, LUT-free constants)
    attrs: dict = field(default_factory=dict)

    @property
    def is_const(self):
        return self.const is not None

    @property
    def ktiles(self):
        return (self.shape[1] + 15) // 16

    @property
    def ub_rows(self):
        return self.ktiles * self.shape[0]


@dataclass
class Op:
    name: str
    kind: str
    inputs: list
    outputs: list
    attrs: dict = field(default_factory=dict)

    @property
    def cls(self):
        return OPS[self.kind]


class Graph:
    def __init__(self, name="graph"):
        self.name = name
        self.tensors = {}
        self.ops = []
        self.inputs, self.outputs = [], []

    # ---- construction ------------------------------------------------------------------------
    def tensor(self, name, shape, dtype="int8", const=None, **attrs):
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name}")
        if const is not None:
            const = np.asarray(const, np.int64)
        t = Tensor(name, tuple(int(s) for s in shape), dtype, const, dict(attrs))
        self.tensors[name] = t
        return name

    def input(self, name, shape, **attrs):
        self.tensor(name, shape, **attrs)
        self.inputs.append(name)
        return name

    def const(self, name, data):
        data = np.asarray(data, np.int64)
        return self.tensor(name, data.shape, "int8", data)

    def op(self, kind, name, inputs, out_shape, out_dtype="int8", out_name=None, out_attrs=None, **attrs):
        if kind not in OPS:
            raise ValueError(f"unknown op kind {kind}")
        if any(o.name == name for o in self.ops):
            raise ValueError(f"duplicate op {name}")
        out = self.tensor(out_name or name, out_shape, out_dtype, **(out_attrs or {}))
        self.ops.append(Op(name, kind, list(inputs), [out], dict(attrs)))
        return out

    def mark_output(self, *names):
        self.outputs += list(names)

    # ---- queries -----------------------------------------------------------------------------
    def T(self, name):
        return self.tensors[name]

    def producer(self, tname):
        for o in self.ops:
            if tname in o.outputs:
                return o
        return None

    def consumers(self, tname):
        return [o for o in self.ops if tname in o.inputs]

    def op_by_name(self, name):
        for o in self.ops:
            if o.name == name:
                return o
        raise KeyError(name)

    def validate(self):
        """Ops are topologically ordered, every input defined, shapes consistent."""
        defined = {n for n, t in self.tensors.items() if t.is_const} | set(self.inputs)
        for o in self.ops:
            for i in o.inputs:
                if i not in defined:
                    raise ValueError(f"{o.name}: input {i} used before definition")
            for out in o.outputs:
                if out in defined:
                    raise ValueError(f"{o.name}: {out} defined twice")
                defined.add(out)
            _check_shapes(self, o)
        for out in self.outputs:
            if out not in defined:
                raise ValueError(f"graph output {out} undefined")
        return True

    def summary(self):
        lines = [f"graph {self.name}: {len(self.ops)} ops, {len(self.tensors)} tensors"]
        for o in self.ops:
            ins = ", ".join(f"{i}{list(self.T(i).shape)}" for i in o.inputs)
            outs = ", ".join(f"{i}{list(self.T(i).shape)}" for i in o.outputs)
            lines.append(f"  [{o.cls:4s}] {o.kind:14s} {o.name:12s} {ins} -> {outs}")
        return "\n".join(lines)


def _check_shapes(g, o):
    T = g.T
    out = T(o.outputs[0])
    if o.kind in ("matmul", "tpu_matmul"):
        x, w = T(o.inputs[0]), T(o.inputs[1])
        K, Nn = (w.shape[1], w.shape[0]) if o.attrs.get("transpose_b") else w.shape
        if x.shape[1] != K or out.shape != (x.shape[0], Nn):
            raise ValueError(f"{o.name}: matmul shapes {x.shape} x {w.shape} -> {out.shape}")
    elif o.kind == "slice":
        c0, c1 = o.attrs["c0"], o.attrs["c1"]
        if c0 % 16 or (c1 % 16 and c1 != T(o.inputs[0]).shape[1]):
            raise ValueError(f"{o.name}: slice must be 16-column aligned")
        if out.shape != (T(o.inputs[0]).shape[0], c1 - c0):
            raise ValueError(f"{o.name}: slice shape")
    elif o.kind == "concat":
        if any(T(i).shape[1] % 16 for i in o.inputs[:-1]):
            raise ValueError(f"{o.name}: concat inputs must be 16-column aligned")
        if out.shape != (T(o.inputs[0]).shape[0], sum(T(i).shape[1] for i in o.inputs)):
            raise ValueError(f"{o.name}: concat shape")
