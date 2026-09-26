#!/usr/bin/env python3
"""Single full-attention block at Qwen3.8-27B text-tower geometry.

Random weights, fixed seed. Matches shapes/dtypes of the real block, not values.
No 27B checkpoint is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_PATH = Path(
    "/Users/carback1/Code/llms/qwen-3.8-mtp-challenge/weights/config.json"
)

RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0
SEED = 42


def load_geometry() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    return {
        "hidden_size": int(cfg["hidden_size"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(cfg["head_dim"]),
        "partial_rotary_factor": float(cfg["partial_rotary_factor"]),
        "attn_output_gate": bool(cfg["attn_output_gate"]),
        "output_gate_type": str(cfg["output_gate_type"]),
        "model_type": str(cfg["model_type"]),
        "attention_bias": bool(cfg["attention_bias"]),
        "rope_theta": float(cfg.get("rope_theta", ROPE_THETA)),
    }


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(orig)


def rotate_half(x: torch.Tensor, half_dim: int) -> torch.Tensor:
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope(
    seq_len: int, rotary_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_gate(gate: torch.Tensor, gate_type: str) -> torch.Tensor:
    if gate_type == "swish":
        return F.silu(gate)
    if gate_type == "sigmoid":
        return torch.sigmoid(gate)
    raise ValueError(f"unsupported output_gate_type={gate_type}")


class FullAttentionBlock(nn.Module):
    """One full-attention transformer sublayer (attention path only, no MLP).

    Included (matches real qwen3_5_text full-attention layer attention path):
      - input RMSNorm
      - Q projection with fused output-gate half (attn_output_gate)
      - K, V projections (GQA: 24 Q heads over 4 KV heads, head_dim 256)
      - per-head Q/K RMSNorm
      - partial RoPE (partial_rotary_factor=0.25)
      - causal scaled-dot-product attention with KV repeat for GQA
      - output gate (swish) applied to attention output
      - output projection
      - residual add

    Simplified / omitted (stated so the number is interpretable):
      - NO MLP / feed-forward
      - NO post_attention_layernorm
      - NO KV cache (static full-window prefill graph)
      - NO attention dropout (config dropout=0 anyway)
      - NO bias on Q/K/V/O (attention_bias=false)
      - random weights (seeded), not checkpoint values
    """

    def __init__(self, geo: dict, seq_len: int):
        super().__init__()
        hidden = geo["hidden_size"]
        n_heads = geo["num_attention_heads"]
        n_kv = geo["num_key_value_heads"]
        head_dim = geo["head_dim"]
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv
        self.seq_len = seq_len
        self.hidden = hidden
        self.rotary_dim = int(head_dim * geo["partial_rotary_factor"])
        self.scale = head_dim**-0.5
        self.gate_type = geo["output_gate_type"]
        self.use_gate = geo["attn_output_gate"]

        self.input_norm = nn.Parameter(torch.ones(hidden))
        q_out = n_heads * head_dim * (2 if self.use_gate else 1)
        self.q = nn.Parameter(torch.empty(q_out, hidden))
        self.k = nn.Parameter(torch.empty(n_kv * head_dim, hidden))
        self.v = nn.Parameter(torch.empty(n_kv * head_dim, hidden))
        self.o = nn.Parameter(torch.empty(hidden, n_heads * head_dim))
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

        cos, sin = build_rope(seq_len, self.rotary_dim, geo["rope_theta"])
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = rms_norm(hidden_states, self.input_norm)
        seq_len = self.seq_len
        q_out = F.linear(x, self.q)
        if self.use_gate:
            q_out = q_out.reshape(1, seq_len, self.n_heads, self.head_dim * 2)
            query, gate = q_out.chunk(2, dim=-1)
            gate = gate.reshape(1, seq_len, self.n_heads * self.head_dim)
        else:
            query = q_out.reshape(1, seq_len, self.n_heads, self.head_dim)
            gate = None
        key = F.linear(x, self.k).reshape(1, seq_len, self.n_kv, self.head_dim)
        value = F.linear(x, self.v).reshape(1, seq_len, self.n_kv, self.head_dim)
        query = rms_norm(query, self.q_norm).transpose(1, 2)
        key = rms_norm(key, self.k_norm).transpose(1, 2)
        value = value.transpose(1, 2)

        cos = self.rope_cos.to(dtype=query.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.to(dtype=query.dtype).unsqueeze(0).unsqueeze(0)
        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        query = torch.cat(
            (q_rot * cos + rotate_half(q_rot, half) * sin, q_pass), dim=-1
        )
        key = torch.cat(
            (k_rot * cos + rotate_half(k_rot, half) * sin, k_pass), dim=-1
        )

        key = (
            key.unsqueeze(2)
            .expand(1, self.n_kv, self.n_rep, seq_len, self.head_dim)
            .reshape(1, self.n_heads, seq_len, self.head_dim)
        )
        value = (
            value.unsqueeze(2)
            .expand(1, self.n_kv, self.n_rep, seq_len, self.head_dim)
            .reshape(1, self.n_heads, seq_len, self.head_dim)
        )

        attn = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn = attn + self.causal_mask.to(dtype=attn.dtype)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        out = torch.matmul(attn, value).transpose(1, 2).reshape(
            1, seq_len, self.n_heads * self.head_dim
        )
        if gate is not None:
            out = out * apply_gate(gate, self.gate_type)
        return residual + F.linear(out, self.o)


def init_random(block: FullAttentionBlock, seed: int = SEED) -> FullAttentionBlock:
    g = torch.Generator()
    g.manual_seed(seed)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.endswith("norm") or name.endswith("_norm"):
                p.copy_(torch.ones_like(p))
            else:
                # Xavier-ish scale so outputs stay finite at large hidden.
                fan_in = p.shape[1] if p.ndim == 2 else p.numel()
                std = (1.0 / max(fan_in, 1)) ** 0.5
                p.copy_(torch.randn(p.shape, generator=g) * std)
    block.eval()
    return block


def build_block(seq_len: int, seed: int = SEED) -> tuple[FullAttentionBlock, dict]:
    geo = load_geometry()
    block = FullAttentionBlock(geo, seq_len)
    init_random(block, seed=seed)
    return block, geo


def parameter_count(block: nn.Module) -> int:
    return sum(p.numel() for p in block.parameters())


if __name__ == "__main__":
    block, geo = build_block(16)
    x = torch.randn(1, 16, geo["hidden_size"])
    with torch.no_grad():
        y = block(x)
    print("geo", geo)
    print("params_M", parameter_count(block) / 1e6)
    print("out", tuple(y.shape), "finite", bool(torch.isfinite(y).all()))
