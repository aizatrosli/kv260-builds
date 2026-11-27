"""Tiny I-JEPA ViT encoder sized for the KV260's on-chip memory.

Pre-LN ViT, no CLS token, learned positional embedding, GELU MLP. The encoder (context/target)
is what gets hard-coded into RTL; the predictor is only needed for training.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    img: int = 32          # MNIST padded 28 -> 32
    chans: int = 1
    patch: int = 4         # 8x8 = 64 tokens
    dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp: int = 512
    pred_dim: int = 64
    pred_depth: int = 2

    @property
    def tokens(self):
        return (self.img // self.patch) ** 2

    @property
    def patch_dim(self):
        return self.chans * self.patch * self.patch


class Block(nn.Module):
    def __init__(self, d, h, m):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, d)

    def attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        o = s.softmax(-1) @ v
        return self.proj(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def patchify(img, p):
    B, C, H, W = img.shape
    x = img.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


class Encoder(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.embed = nn.Linear(c.patch_dim, c.dim)
        self.pos = nn.Parameter(torch.randn(1, c.tokens, c.dim) * 0.02)
        self.blocks = nn.ModuleList(Block(c.dim, c.heads, c.mlp) for _ in range(c.depth))
        self.ln = nn.LayerNorm(c.dim)

    def forward(self, img, keep=None):
        x = self.embed(patchify(img, self.c.patch)) + self.pos
        if keep is not None:                      # context encoder: visible tokens only
            x = torch.gather(x, 1, keep[..., None].expand(-1, -1, x.shape[-1]))
        for b in self.blocks:
            x = b(x)
        return self.ln(x)


class Predictor(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.inp = nn.Linear(c.dim, c.pred_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, c.pred_dim))
        self.pos = nn.Parameter(torch.randn(1, c.tokens, c.pred_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(c.pred_dim, 2, 4 * c.pred_dim) for _ in range(c.pred_depth))
        self.ln = nn.LayerNorm(c.pred_dim)
        self.out = nn.Linear(c.pred_dim, c.dim)

    def forward(self, ctx, keep, tgt):
        B = ctx.shape[0]
        pos = self.pos.expand(B, -1, -1)
        g = lambda idx: torch.gather(pos, 1, idx[..., None].expand(-1, -1, pos.shape[-1]))
        x = self.inp(ctx) + g(keep)
        m = self.mask_tok.expand(B, tgt.shape[1], -1) + g(tgt)
        x = torch.cat([x, m], 1)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln(x[:, -tgt.shape[1]:]))
