#!/usr/bin/env python3
"""Traceable PyTorch text tower for Qwen3.5-0.8B, loaded from MLX 4-bit weights.

Single static window, no KV cache, no vision. Gated DeltaNet is unrolled over
SEQ_LEN so torch.jit.trace / coremltools can convert it.

RMSNorm matches the MLX 4-bit checkpoint: multiply by the stored weight
directly (the converter already baked HuggingFace's (1+w) into the tensor).
GatedDeltaNet Q/K use L2-norm, matching transformers.models.qwen3_5.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open

SOURCE_DIR = Path("/Users/carback1/.cache/mlxfast/qwen3.5-0.8b-src")
SOURCE_WEIGHTS = SOURCE_DIR / "model.safetensors"
SOURCE_CONFIG = SOURCE_DIR / "config.json"
TOKENIZER_JSON = SOURCE_DIR / "tokenizer.json"

SEQ_LEN = 16
GROUP_SIZE = 64
BITS = 4
RMS_EPS = 1e-6
ROPE_THETA = 10_000_000.0
PARTIAL_ROTARY = 0.25
FIXED_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin."
)


def load_text_config() -> dict:
    cfg = json.loads(SOURCE_CONFIG.read_text())
    return cfg["text_config"]


def affine_dequant_u32(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> torch.Tensor:
    """MLX affine 4-bit: w_i = s * hat_w_i + beta, packed low-nibble-first in U32."""
    packed = packed.to(torch.int64) & 0xFFFFFFFF
    scales = scales.float()
    biases = biases.float()
    out_features, n_groups = scales.shape
    in_features = n_groups * group_size
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(pack_factor, device=packed.device, dtype=torch.int64) * bits
    unpacked = (packed.unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(out_features, -1)[:, :in_features]
    unpacked = unpacked.reshape(out_features, n_groups, group_size).float()
    return (unpacked * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(
        out_features, in_features
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Llama-style RMSNorm used by the MLX 4-bit dump (weight already includes +1)."""
    orig = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * weight.float()).to(orig)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rotate_half(x: torch.Tensor, half_dim: int) -> torch.Tensor:
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope(seq_len: int, rotary_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    seq_len: int,
    n_heads: int,
    k_dim: int,
    v_dim: int,
    qk_scale: float,
) -> torch.Tensor:
    """Unrolled gated delta rule. Shapes [B, S, H, D]. Trace-friendly."""
    query = query.transpose(1, 2).float()
    key = key.transpose(1, 2).float()
    value = value.transpose(1, 2).float()
    decay = decay.transpose(1, 2).float()
    beta = beta.transpose(1, 2).float()
    query = l2norm(query, dim=-1) * qk_scale
    key = l2norm(key, dim=-1)

    state = torch.zeros(
        1, n_heads, k_dim, v_dim, dtype=torch.float32, device=query.device
    )
    outs = []
    for t in range(seq_len):
        q_t = query[:, :, t]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        state = state * decay[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outs, dim=2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(intermediate, hidden))
        self.up = nn.Parameter(torch.empty(intermediate, hidden))
        self.down = nn.Parameter(torch.empty(hidden, intermediate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


class GDNBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        n_v = int(cfg["linear_num_value_heads"])
        n_k = int(cfg["linear_num_key_heads"])
        head_k = int(cfg["linear_key_head_dim"])
        head_v = int(cfg["linear_value_head_dim"])
        conv_k = int(cfg["linear_conv_kernel_dim"])
        key_dim = n_k * head_k
        value_dim = n_v * head_v
        conv_dim = key_dim * 2 + value_dim
        self.n_k = n_k
        self.n_v = n_v
        self.head_k = head_k
        self.head_v = head_v
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.conv_dim = conv_dim
        self.conv_k = conv_k
        self.input_norm = nn.Parameter(torch.ones(hidden))
        self.post_norm = nn.Parameter(torch.ones(hidden))
        self.qkv = nn.Parameter(torch.empty(conv_dim, hidden))
        self.z = nn.Parameter(torch.empty(value_dim, hidden))
        self.b = nn.Parameter(torch.empty(n_v, hidden))
        self.a = nn.Parameter(torch.empty(n_v, hidden))
        self.out = nn.Parameter(torch.empty(hidden, value_dim))
        self.conv = nn.Parameter(torch.empty(conv_dim, 1, conv_k))
        self.dt = nn.Parameter(torch.ones(n_v))
        self.alog = nn.Parameter(torch.zeros(n_v))
        self.gdn_norm = nn.Parameter(torch.ones(head_v))
        self.mlp = MLP(hidden, int(cfg["intermediate_size"]))
        self.seq_len = SEQ_LEN
        self.qk_scale = head_k ** -0.5

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        x = rms_norm(hidden, self.input_norm)
        seq_len = self.seq_len
        mixed = F.linear(x, self.qkv).transpose(1, 2)
        conv = F.conv1d(mixed, self.conv, padding=self.conv_k - 1, groups=self.conv_dim)
        conv = conv[:, :, :seq_len]
        mixed = F.silu(conv).transpose(1, 2)
        query, key, value = mixed.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(1, seq_len, self.n_k, self.head_k)
        key = key.reshape(1, seq_len, self.n_k, self.head_k)
        value = value.reshape(1, seq_len, self.n_v, self.head_v)
        z = F.linear(x, self.z).reshape(1, seq_len, self.n_v, self.head_v)
        beta = torch.sigmoid(F.linear(x, self.b))
        a = F.linear(x, self.a)
        g = -self.alog.float().exp() * F.softplus(a.float() + self.dt.float())
        core = gated_delta_recurrent(
            query,
            key,
            value,
            g,
            beta,
            seq_len=seq_len,
            n_heads=self.n_v,
            k_dim=self.head_k,
            v_dim=self.head_v,
            qk_scale=self.qk_scale,
        )
        core = core.reshape(seq_len * self.n_v, self.head_v)
        var = core.pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(var + RMS_EPS)
        core = self.gdn_norm.float() * core
        core = core * F.silu(z.reshape(seq_len * self.n_v, self.head_v).float())
        core = core.reshape(1, seq_len, self.value_dim).to(dtype=x.dtype)
        hidden = residual + F.linear(core, self.out)
        return hidden + self.mlp(rms_norm(hidden, self.post_norm))


class AttnBlock(nn.Module):
    def __init__(self, cfg: dict, seq_len: int):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        n_heads = int(cfg["num_attention_heads"])
        n_kv = int(cfg["num_key_value_heads"])
        head_dim = int(cfg["head_dim"])
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv
        self.rotary_dim = int(head_dim * PARTIAL_ROTARY)
        self.scale = head_dim ** -0.5
        self.input_norm = nn.Parameter(torch.ones(hidden))
        self.post_norm = nn.Parameter(torch.ones(hidden))
        self.q = nn.Parameter(torch.empty(n_heads * head_dim * 2, hidden))
        self.k = nn.Parameter(torch.empty(n_kv * head_dim, hidden))
        self.v = nn.Parameter(torch.empty(n_kv * head_dim, hidden))
        self.o = nn.Parameter(torch.empty(hidden, n_heads * head_dim))
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        self.mlp = MLP(hidden, int(cfg["intermediate_size"]))
        cos, sin = build_rope(seq_len, self.rotary_dim, ROPE_THETA)
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        x = rms_norm(hidden, self.input_norm)
        seq_len = SEQ_LEN
        q_out = F.linear(x, self.q).reshape(1, seq_len, self.n_heads, self.head_dim * 2)
        query, gate = q_out.chunk(2, dim=-1)
        gate = gate.reshape(1, seq_len, self.n_heads * self.head_dim)
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
        query = torch.cat((q_rot * cos + rotate_half(q_rot, half) * sin, q_pass), dim=-1)
        key = torch.cat((k_rot * cos + rotate_half(k_rot, half) * sin, k_pass), dim=-1)
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
        out = out * torch.sigmoid(gate)
        hidden = residual + F.linear(out, self.o)
        return hidden + self.mlp(rms_norm(hidden, self.post_norm))


class Qwen35Text(nn.Module):
    def __init__(self, text_cfg: dict, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = int(text_cfg["hidden_size"])
        self.vocab = int(text_cfg["vocab_size"])
        self.n_layers = int(text_cfg["num_hidden_layers"])
        self.layer_types = list(text_cfg["layer_types"])
        self.embed = nn.Embedding(self.vocab, self.hidden)
        self.final_norm = nn.Parameter(torch.ones(self.hidden))
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if self.layer_types[i] == "linear_attention":
                self.layers.append(GDNBlock(text_cfg))
            else:
                self.layers.append(AttnBlock(text_cfg, seq_len))

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return rms_norm(hidden, self.final_norm)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.linear(self.forward_hidden(input_ids), self.embed.weight)


def _load_linear(f, prefix: str) -> torch.Tensor:
    weight = f.get_tensor(prefix + "weight")
    scales_key = prefix + "scales"
    if scales_key in f.keys():
        scales = f.get_tensor(scales_key)
        biases = f.get_tensor(prefix + "biases")
        return affine_dequant_u32(weight, scales, biases)
    return weight.float()


def load_from_mlx(model: Qwen35Text, path: Path = SOURCE_WEIGHTS) -> Qwen35Text:
    with safe_open(str(path), framework="pt") as f:
        model.embed.weight.data.copy_(_load_linear(f, "language_model.model.embed_tokens."))
        model.final_norm.data.copy_(f.get_tensor("language_model.model.norm.weight").float())
        for i, layer in enumerate(model.layers):
            lp = f"language_model.model.layers.{i}."
            layer.input_norm.data.copy_(f.get_tensor(lp + "input_layernorm.weight").float())
            layer.post_norm.data.copy_(
                f.get_tensor(lp + "post_attention_layernorm.weight").float()
            )
            layer.mlp.gate.data.copy_(_load_linear(f, lp + "mlp.gate_proj."))
            layer.mlp.up.data.copy_(_load_linear(f, lp + "mlp.up_proj."))
            layer.mlp.down.data.copy_(_load_linear(f, lp + "mlp.down_proj."))
            if isinstance(layer, GDNBlock):
                layer.qkv.data.copy_(_load_linear(f, lp + "linear_attn.in_proj_qkv."))
                layer.z.data.copy_(_load_linear(f, lp + "linear_attn.in_proj_z."))
                layer.b.data.copy_(_load_linear(f, lp + "linear_attn.in_proj_b."))
                layer.a.data.copy_(_load_linear(f, lp + "linear_attn.in_proj_a."))
                layer.out.data.copy_(_load_linear(f, lp + "linear_attn.out_proj."))
                conv = f.get_tensor(lp + "linear_attn.conv1d.weight").float()
                if conv.ndim == 3 and conv.shape[-1] == 1:
                    conv = conv.permute(0, 2, 1).contiguous()
                layer.conv.data.copy_(conv)
                layer.dt.data.copy_(f.get_tensor(lp + "linear_attn.dt_bias").float())
                layer.alog.data.copy_(f.get_tensor(lp + "linear_attn.A_log").float())
                layer.gdn_norm.data.copy_(
                    f.get_tensor(lp + "linear_attn.norm.weight").float()
                )
            else:
                layer.q.data.copy_(_load_linear(f, lp + "self_attn.q_proj."))
                layer.k.data.copy_(_load_linear(f, lp + "self_attn.k_proj."))
                layer.v.data.copy_(_load_linear(f, lp + "self_attn.v_proj."))
                layer.o.data.copy_(_load_linear(f, lp + "self_attn.o_proj."))
                layer.q_norm.data.copy_(f.get_tensor(lp + "self_attn.q_norm.weight").float())
                layer.k_norm.data.copy_(f.get_tensor(lp + "self_attn.k_norm.weight").float())
    return model


def tokenize_fixed_prompt(seq_len: int = SEQ_LEN) -> torch.Tensor:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = tok.encode(FIXED_PROMPT).ids
    if len(ids) >= seq_len:
        ids = ids[:seq_len]
    else:
        ids = ids + [0] * (seq_len - len(ids))
    return torch.tensor([ids], dtype=torch.int32)


def build_loaded_model(seq_len: int = SEQ_LEN) -> Qwen35Text:
    cfg = load_text_config()
    model = Qwen35Text(cfg, seq_len=seq_len)
    load_from_mlx(model)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.set_num_threads(4)
    model = build_loaded_model()
    ids = tokenize_fixed_prompt()
    print("input_ids", ids.tolist())
    print("n_tokens", int((ids != 0).sum()))
    print("params_M", parameter_count(model) / 1e6)
    with torch.no_grad():
        logits = model(ids)
        logits2 = model(ids)
    print("logits", tuple(logits.shape), "dtype", logits.dtype)
    print("finite", bool(torch.isfinite(logits).all()))
    print("has_nan", bool(torch.isnan(logits).any()))
    print("repeat_max_abs_diff", float((logits - logits2).abs().max()))
    last = logits[0, -1]
    top = torch.topk(last, 8)
    print("last_argmax", int(last.argmax()))
    print("top8_ids", top.indices.tolist())
    print("top8_logits", [float(v) for v in top.values])
    print("last_logit_min/max", float(last.min()), float(last.max()))
