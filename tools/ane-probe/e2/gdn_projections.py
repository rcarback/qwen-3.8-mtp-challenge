#!/usr/bin/env python3
"""Gated-delta linear-attention layer minus the sequential recurrence.

Includes the projection GEMMs that would go to ANE under the split:
  input RMSNorm
  in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
  gated RMSNorm on a recurrence stand-in (value) * silu(z)
  out_proj + residual
  post RMSNorm + SwiGLU MLP

Excludes the sequential gated-delta state update (known ANECCompile failure).
Excludes the depthwise conv1d (not a GEMM; not in the 114 ms projection budget).

Query, key, a, and b would otherwise be dead-code-eliminated without the
recurrence. They stay live through a tiny data-dependent add so the converted
graph still contains every in_proj GEMM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RMS_EPS, SEED, SEQ, load_geometry


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(orig)


class GatedDeltaProjections(nn.Module):
    def __init__(self, geo: dict, seq_len: int):
        super().__init__()
        hidden = geo["hidden_size"]
        intermediate = geo["intermediate_size"]
        n_k = geo["linear_num_key_heads"]
        n_v = geo["linear_num_value_heads"]
        head_k = geo["linear_key_head_dim"]
        head_v = geo["linear_value_head_dim"]
        key_dim = geo["key_dim"]
        value_dim = geo["value_dim"]
        conv_dim = geo["conv_dim"]
        self.seq_len = seq_len
        self.n_k = n_k
        self.n_v = n_v
        self.head_k = head_k
        self.head_v = head_v
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden = hidden
        self.eps = float(geo.get("rms_norm_eps", RMS_EPS))

        self.input_norm = nn.Parameter(torch.ones(hidden))
        self.post_norm = nn.Parameter(torch.ones(hidden))
        self.gdn_norm = nn.Parameter(torch.ones(head_v))
        self.qkv = nn.Parameter(torch.empty(conv_dim, hidden))
        self.z = nn.Parameter(torch.empty(value_dim, hidden))
        self.b = nn.Parameter(torch.empty(n_v, hidden))
        self.a = nn.Parameter(torch.empty(n_v, hidden))
        self.out = nn.Parameter(torch.empty(hidden, value_dim))
        self.gate = nn.Parameter(torch.empty(intermediate, hidden))
        self.up = nn.Parameter(torch.empty(intermediate, hidden))
        self.down = nn.Parameter(torch.empty(hidden, intermediate))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = rms_norm(hidden_states, self.input_norm, self.eps)
        seq_len = self.seq_len

        qkv = F.linear(x, self.qkv)
        query, key, value = qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(1, seq_len, self.n_k, self.head_k)
        key = key.reshape(1, seq_len, self.n_k, self.head_k)
        value = value.reshape(1, seq_len, self.n_v, self.head_v)
        z = F.linear(x, self.z).reshape(1, seq_len, self.n_v, self.head_v)
        beta = torch.sigmoid(F.linear(x, self.b))
        a = F.linear(x, self.a)

        # Recurrence stand-in: value has the same [B,S,Hv,Dv] shape as the
        # gated-delta scan output. Not a numerical model of the scan.
        core = value.reshape(seq_len * self.n_v, self.head_v).float()
        var = core.pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(var + self.eps)
        core = self.gdn_norm.float() * core
        core = core * F.silu(z.reshape(seq_len * self.n_v, self.head_v).float())
        core = core.reshape(1, seq_len, self.value_dim).to(dtype=x.dtype)
        mixed = F.linear(core, self.out)

        # Keep query/key/a/b live so MIL cannot drop those GEMM rows.
        live = (
            query.mean() + key.mean() + a.mean() + beta.mean()
        ) * (1.0 / 4096.0)
        mixed = mixed + live.to(dtype=mixed.dtype)

        hidden = residual + mixed
        mlp_in = rms_norm(hidden, self.post_norm, self.eps)
        mlp = F.linear(
            F.silu(F.linear(mlp_in, self.gate)) * F.linear(mlp_in, self.up),
            self.down,
        )
        return hidden + mlp


def init_random(block: GatedDeltaProjections, seed: int = SEED) -> GatedDeltaProjections:
    g = torch.Generator()
    g.manual_seed(seed)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "norm" in name:
                p.copy_(torch.ones_like(p))
            else:
                fan_in = p.shape[1] if p.ndim == 2 else p.numel()
                std = (1.0 / max(fan_in, 1)) ** 0.5
                p.copy_(torch.randn(p.shape, generator=g) * std)
    block.eval()
    return block


def build_block(seq_len: int = SEQ, seed: int = SEED) -> tuple[GatedDeltaProjections, dict]:
    geo = load_geometry()
    block = GatedDeltaProjections(geo, seq_len)
    init_random(block, seed=seed)
    return block, geo


def parameter_count(block: nn.Module) -> int:
    return sum(p.numel() for p in block.parameters())


def gemm_param_breakdown(block: GatedDeltaProjections) -> dict:
    named = dict(block.named_parameters())
    keys = ("qkv", "z", "b", "a", "out", "gate", "up", "down")
    return {k: int(named[k].numel()) for k in keys}


if __name__ == "__main__":
    block, geo = build_block(SEQ)
    x = torch.randn(1, SEQ, geo["hidden_size"])
    with torch.no_grad():
        y = block(x)
    print("geo", {k: geo[k] for k in (
        "hidden_size", "intermediate_size", "conv_dim", "value_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim",
    )})
    print("params_M", parameter_count(block) / 1e6)
    print("gemm", gemm_param_breakdown(block))
    print("out", tuple(y.shape), "finite", bool(torch.isfinite(y).all()))
