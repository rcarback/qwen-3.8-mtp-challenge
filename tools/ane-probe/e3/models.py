#!/usr/bin/env python3
"""Isolated op-class graphs at Qwen 3.8 27B text-tower geometry.

Random weights, fixed seed. Shapes match the real tower; values do not.
No 27B checkpoint is loaded.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
RMS_EPS = 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(orig)


def init_linear_(module: nn.Module, seed: int) -> None:
    g = torch.Generator()
    g.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim == 1:
                p.fill_(1.0)
            else:
                fan_in = p.shape[1]
                std = (1.0 / max(fan_in, 1)) ** 0.5
                p.copy_(torch.randn(p.shape, generator=g) * std)
    module.eval()


class LinearGEMM(nn.Module):
    """y = x @ W^T, x is [1, S, in], W is [out, in], no bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AttentionProjections(nn.Module):
    """Q (fused output-gate), K, V, O at full-attention geometry.

    Q rows = 2 * n_q * head_dim because attn_output_gate packs the gate.
    O consumes n_q * head_dim. K/V consume n_kv * head_dim.
    O's input is a separate tensor so the four GEMMs stay independent.
    """

    def __init__(self, hidden: int, n_q: int, n_kv: int, head_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden, 2 * n_q * head_dim, bias=False)
        self.k = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o = nn.Linear(n_q * head_dim, hidden, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, attn_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        o = self.o(attn_out)
        return q, k, v, o


class ScoreMatmuls(nn.Module):
    """QK^T and AV at GQA geometry, no softmax.

    Q is [1, n_q, S, D]; K/V are cached [1, n_kv, KV, D]. K/V are repeated
    to n_q heads the same way the Swift GQA path expands them. Softmax is
    omitted on purpose: this cell isolates the two score matmuls.
    """

    def __init__(self, n_q: int, n_kv: int, head_dim: int, seq: int, kv: int):
        super().__init__()
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.seq = seq
        self.kv = kv
        self.n_rep = n_q // n_kv
        self.scale = head_dim**-0.5

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        k = (
            k.unsqueeze(2)
            .expand(1, self.n_kv, self.n_rep, self.kv, self.head_dim)
            .reshape(1, self.n_q, self.kv, self.head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(1, self.n_kv, self.n_rep, self.kv, self.head_dim)
            .reshape(1, self.n_q, self.kv, self.head_dim)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        return torch.matmul(scores, v)


class RMSSiLUSwiGLU(nn.Module):
    """RMSNorm + SiLU + multiply at a single width.

    Inputs `gate` and `up` are both [1, S, width]. This is the elementwise
    body of SwiGLU after the two GEMMs, plus an RMSNorm on `gate`.
    """

    def __init__(self, width: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        x = rms_norm(gate, self.weight, self.eps)
        return F.silu(x) * up


def gemm_flops(batch_seq: int, out_features: int, in_features: int) -> int:
    return 2 * batch_seq * out_features * in_features


def score_flops(n_q: int, seq: int, kv: int, head_dim: int) -> int:
    # Two matmuls, QK^T and AV, counted at the expanded GQA head count.
    return 2 * (2 * n_q * seq * kv * head_dim)


def elem_flops(batch_seq: int, width: int) -> int:
    # RMSNorm ~4, SiLU ~4, mul 1, per element. Order-of-magnitude only.
    return batch_seq * width * 9
