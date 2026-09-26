#!/usr/bin/env python3
"""Trivial ANE-placeable graph: marshalling, not arithmetic, dominates.

Input and output are the real activation shape [1, 1024, 5120] fp16 (~10.5 MB).
The body is a bias add plus a scale, so the timed predict is a round-trip of
that tensor through a Core ML ANE model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import SEQ, load_geometry


class CrossingNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Tiny learned affine so the graph is not constant-folded away.
        self.scale = nn.Parameter(torch.tensor(1.0001, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((hidden,), 0.0001, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale + self.bias


def build_crossing() -> tuple[CrossingNet, dict]:
    geo = load_geometry()
    net = CrossingNet(geo["hidden_size"]).eval()
    return net, geo


if __name__ == "__main__":
    net, geo = build_crossing()
    x = torch.randn(1, SEQ, geo["hidden_size"])
    with torch.no_grad():
        y = net(x)
    print("geo.hidden", geo["hidden_size"], "out", tuple(y.shape))
