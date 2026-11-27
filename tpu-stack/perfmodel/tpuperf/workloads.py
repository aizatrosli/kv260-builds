"""Workload descriptions: a network is an ordered list of GEMMs and vector (VPU) ops.

Shapes follow the tpuv1 mapping: a GEMM streams M activation rows (tokens x batch, or output
pixels) through a weight tile of R x C; K is tiled by R, N by C. Convolutions use tpuv1's
CONVOLVE mapping (one K tile per kernel tap and per R input channels, read straight from the
UB, no im2col copy). Attention Q.K^T and P.V are GEMMs whose "weights" are activations
(``dynamic=True``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import List, Optional, Union


@dataclass
class Gemm:
    name: str
    M: int                      # activation rows streamed per weight tile
    K: int
    N: int
    count: int = 1              # identical independent instances (heads x batch)
    dynamic: bool = False       # "weights" are activations (Q.K^T, P.V)
    taps: int = 1               # conv kernel taps (kh*kw); K = taps * cin
    depthwise: bool = False     # depthwise conv: N == channels, K = taps
    host_im2col: bool = False   # stem conv: im2col done when the image is DMA'd in (K dense)
    in_bytes: Optional[int] = None    # UB-resident input bytes (default M*K)
    out_bytes: Optional[int] = None   # UB-resident output bytes (default M*N)

    @property
    def cin(self) -> int:
        return self.K // self.taps

    def k_tiles(self, R: int, C: int) -> int:
        if self.depthwise:
            return self.taps * ceil(min(C, self.N) / R)
        if self.taps > 1 and not self.host_im2col:
            return self.taps * ceil(self.cin / R)
        return ceil(self.K / R)

    def n_tiles(self, C: int) -> int:
        return ceil(self.N / C)

    @property
    def macs(self) -> int:
        if self.depthwise:
            return self.M * self.taps * self.N * self.count
        return self.M * self.K * self.N * self.count

    @property
    def weight_bytes(self) -> int:
        """INT8 parameter bytes (0 for dynamic operands)."""
        if self.dynamic:
            return 0
        return (self.taps * self.N if self.depthwise else self.K * self.N) * self.count

    @property
    def operand_bytes(self) -> int:
        """Bytes that travel down the weight path (params or dynamic operands)."""
        return (self.taps * self.N if self.depthwise else self.K * self.N) * self.count

    @property
    def ub_in(self) -> int:
        return self.in_bytes if self.in_bytes is not None else self.M * self.K * self.count

    @property
    def ub_out(self) -> int:
        return self.out_bytes if self.out_bytes is not None else self.M * self.N * self.count


@dataclass
class VecOp:
    """A VPU job type: ``jobs`` vectors of ``elems`` INT8 elements each."""
    name: str
    kind: str                   # "ln", "softmax", "gelu", "add"
    jobs: int
    elems: int


Layer = Union[Gemm, VecOp]


@dataclass
class Workload:
    name: str
    layers: List[Layer]
    batch: int = 1
    input_bytes: int = 0
    output_bytes: int = 0
    description: str = ""
    separate_programs: bool = False   # every GEMM is its own job (jepa-tpu Verilator flow)

    @property
    def gemms(self) -> List[Gemm]:
        return [l for l in self.layers if isinstance(l, Gemm)]

    @property
    def vecops(self) -> List[VecOp]:
        return [l for l in self.layers if isinstance(l, VecOp)]

    @property
    def macs(self) -> int:
        return sum(g.macs for g in self.gemms)

    @property
    def param_bytes(self) -> int:
        return sum(g.weight_bytes for g in self.gemms)

    @property
    def n_matmuls(self) -> int:
        return sum(g.count for g in self.gemms)


# ---------------------------------------------------------------------------------------------
# MLPs (tpuv1 benchmarks)
# ---------------------------------------------------------------------------------------------
def pad16(x: int) -> int:
    return -(-x // 16) * 16


def mlp(name: str, dims, batch: int) -> Workload:
    layers = []
    for i, (k, n) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(Gemm(f"fc{i}", batch, k, n))
    return Workload(name, layers, batch=batch, input_bytes=batch * pad16(dims[0]),
                    output_bytes=batch * pad16(dims[-1]),
                    description=f"MLP {'-'.join(map(str, dims))}, batch {batch}")


def mlp_wide(batch=100):
    return mlp("mlp_wide", [256, 256], batch)


def mlp3(batch=32):
    return mlp("mlp3", [64, 128, 64, 16], batch)


def digits(batch=128):
    return mlp(f"digits_b{batch}", [64, 128, 64, 10], batch)


# ---------------------------------------------------------------------------------------------
# Vision transformers
# ---------------------------------------------------------------------------------------------
def vit(name: str, dim: int, depth: int, heads: int, hidden: int, tokens: int, patch_dim: int,
        batch: int = 1, cls: bool = False, head_classes: int = 0, pos_add: bool = True,
        description: str = "") -> Workload:
    T = tokens + (1 if cls else 0)
    hd = dim // heads
    M = T * batch
    L: List[Layer] = [Gemm("embed", tokens * batch, patch_dim, dim)]
    if pos_add:
        L.append(VecOp("pos_add", "add", M, dim))
    for b in range(depth):
        p = f"b{b}."
        L += [
            VecOp(p + "ln1", "ln", M, dim),
            Gemm(p + "qkv", M, dim, 3 * dim),
            Gemm(p + "qk", T, hd, T, count=heads * batch, dynamic=True),
            VecOp(p + "softmax", "softmax", heads * M, T),
            Gemm(p + "pv", T, T, hd, count=heads * batch, dynamic=True),
            Gemm(p + "proj", M, dim, dim),
            VecOp(p + "add1", "add", M, dim),
            VecOp(p + "ln2", "ln", M, dim),
            Gemm(p + "fc1", M, dim, hidden),
            VecOp(p + "gelu", "gelu", M, hidden),
            Gemm(p + "fc2", M, hidden, dim),
            VecOp(p + "add2", "add", M, dim),
        ]
    L.append(VecOp("ln_f", "ln", M, dim))
    if head_classes:
        L.append(Gemm("head", batch, dim, head_classes))
    return Workload(name, L, batch=batch, input_bytes=tokens * patch_dim * batch,
                    output_bytes=(batch * head_classes if head_classes else M * dim),
                    description=description or f"ViT d{dim} x{depth} h{heads} mlp{hidden} T{T}")


def jepa_tiny(batch=1) -> Workload:
    """jepa-tpu encoder: dim 128, 6 blocks, 4 heads, MLP 512, 64 tokens (32x32, patch 4)."""
    w = vit("jepa_tiny", 128, 6, 4, 512, 64, 16, batch=batch,
            description="tiny I-JEPA ViT encoder (jepa-tpu), 64 tokens")
    if batch == 1:
        w.separate_programs = True
    return w


def vit_tiny(batch=1) -> Workload:
    return vit("vit_tiny", 192, 12, 3, 768, 196, 768, batch=batch, cls=True, head_classes=1000,
               description="ViT-Ti/16 224px (5.7M params)")


def deit_small(batch=1) -> Workload:
    return vit("deit_small", 384, 12, 6, 1536, 196, 768, batch=batch, cls=True,
               head_classes=1000, description="DeiT-S/16 224px (22M params)")


def ijepa_vits(batch=1) -> Workload:
    return vit("ijepa_vits14", 384, 12, 6, 1536, 256, 588, batch=batch,
               description="I-JEPA-style ViT-S/14 224px context encoder, 256 tokens, no CLS")


# ---------------------------------------------------------------------------------------------
# CNNs (tpuv1 CONVOLVE mapping)
# ---------------------------------------------------------------------------------------------
def conv(name, hw_in, cin, cout, k, stride, batch=1, depthwise=False, host_im2col=False) -> Gemm:
    hw_out = -(-hw_in // stride)
    M = hw_out * hw_out * batch
    taps = k * k
    if host_im2col:
        return Gemm(name, M, taps * cin, cout, taps=taps, host_im2col=True, out_bytes=M * cout)
    if depthwise:
        return Gemm(name, M, taps, cout, taps=taps, depthwise=True,
                    in_bytes=hw_in * hw_in * cin * batch, out_bytes=M * cout)
    return Gemm(name, M, taps * cin, cout, taps=taps,
                in_bytes=hw_in * hw_in * cin * batch, out_bytes=M * cout)


def resnet18(batch=1, res=224) -> Workload:
    L: List[Layer] = [conv("conv1", res, 3, 64, 7, 2, batch, host_im2col=True)]
    hw = res // 4                       # after conv1 stride 2 + maxpool stride 2
    cin = 64
    for stage, (c, s) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)]):
        for blk in range(2):
            stride = s if blk == 0 else 1
            p = f"l{stage + 1}.{blk}."
            L.append(conv(p + "c1", hw, cin, c, 3, stride, batch))
            hw_o = -(-hw // stride)
            L.append(conv(p + "c2", hw_o, c, c, 3, 1, batch))
            if stride != 1 or cin != c:
                L.append(conv(p + "down", hw, cin, c, 1, stride, batch))
            L.append(VecOp(p + "add", "add", hw_o * hw_o * batch, c))
            hw, cin = hw_o, c
    L.append(Gemm("fc", batch, 512, 1000))
    return Workload("resnet18", L, batch=batch, input_bytes=(res // 2) ** 2 * 147 * batch,
                    output_bytes=1000 * batch, description=f"ResNet-18 {res}px (11.7M params)")


def mobilenet_v2(batch=1, res=224) -> Workload:
    L: List[Layer] = [conv("conv1", res, 3, 32, 3, 2, batch, host_im2col=True)]
    hw, cin = res // 2, 32
    cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1),
           (6, 160, 3, 2), (6, 320, 1, 1)]
    i = 0
    for t, c, n, s in cfg:
        for j in range(n):
            stride = s if j == 0 else 1
            p = f"ir{i}."
            mid = cin * t
            if t != 1:
                L.append(conv(p + "expand", hw, cin, mid, 1, 1, batch))
            L.append(conv(p + "dw", hw, mid, mid, 3, stride, batch, depthwise=True))
            hw = -(-hw // stride)
            L.append(conv(p + "project", hw, mid, c, 1, 1, batch))
            if stride == 1 and cin == c:
                L.append(VecOp(p + "add", "add", hw * hw * batch, c))
            cin = c
            i += 1
    L.append(conv("conv_last", hw, cin, 1280, 1, 1, batch))
    L.append(Gemm("fc", batch, 1280, 1000))
    return Workload("mobilenet_v2", L, batch=batch, input_bytes=(res // 2) ** 2 * 27 * batch,
                    output_bytes=1000 * batch, description=f"MobileNetV2 {res}px (3.5M params)")


ZOO = {
    "mlp_wide": mlp_wide, "mlp3": mlp3, "digits_b128": lambda batch=128: digits(batch),
    "jepa_tiny": jepa_tiny, "vit_tiny": vit_tiny, "deit_small": deit_small,
    "ijepa_vits14": ijepa_vits, "resnet18": resnet18, "mobilenet_v2": mobilenet_v2,
}


def get(name: str, batch: Optional[int] = None) -> Workload:
    f = ZOO[name]
    return f() if batch is None else f(batch=batch)
