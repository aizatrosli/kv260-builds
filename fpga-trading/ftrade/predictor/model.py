"""Tiny direction MLP: 16 int8-friendly inputs -> 32 -> 16 -> 3 logits (down, flat, up), ReLU6 hidden.

All widths except the 3 logits are multiples of 16, so every layer maps onto whole 16x16 tiles
of the tpuv1 systolic array (the logits are padded to 16 lanes on the TPU).
"""
from __future__ import annotations

import torch
import torch.nn as nn

DIMS = (16, 32, 16, 3)


class DirectionMLP(nn.Module):
    def __init__(self, dims=DIMS, act="relu6"):
        super().__init__()
        self.dims = tuple(dims)
        self.act = act
        self.fc0 = nn.Linear(dims[0], dims[1])
        self.fc1 = nn.Linear(dims[1], dims[2])
        self.fc2 = nn.Linear(dims[2], dims[3])
        A = {"relu": nn.ReLU, "relu6": nn.ReLU6}[act]
        self.act0 = A()        # ReLU6: bounded hidden ranges -> fixed int8 scales, no outlier clipping
        self.act1 = A()

    def forward(self, x):
        return self.fc2(self.act1(self.fc1(self.act0(self.fc0(x)))))


def save(model: DirectionMLP, path):
    torch.save({"dims": model.dims, "act": model.act, "state": model.state_dict()}, path)


def load(path) -> DirectionMLP:
    d = torch.load(path, map_location="cpu", weights_only=True)
    m = DirectionMLP(tuple(d["dims"]), d.get("act", "relu"))
    m.load_state_dict(d["state"])
    return m.eval()
