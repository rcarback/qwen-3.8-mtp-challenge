"""Dump a DFlash2 reference fixture for the Swift port to check itself against.

Runs the upstream MLX drafter on deterministic synthetic inputs and writes
every input and every intermediate to one safetensors file. The Swift port
loads the same file, binds the saved embedding output in place of the target's
embedding table, and compares.

The target model is never loaded. The embedding table and lm_head are the only
things the drafter borrows from it, and both are replaced here by saved
arrays, so a 15 GB residency buys nothing.
"""
import os, sys, json
sys.path.insert(0, os.environ.get("DFLASH2_REFERENCE_PATH", "."))
import mlx.core as mx
from mlx import nn
from dflash.model_mlx import load_draft

DRAFT = sys.argv[1]
OUT = sys.argv[2]
CONTEXT_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 5
BLOCK = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LOGITS_START = 1

QUANT = sys.argv[5] if len(sys.argv) > 5 else None

draft = load_draft(DRAFT)
if QUANT:
    bits, group = (int(x) for x in QUANT.split(","))
    nn.quantize(draft, group_size=group, bits=bits)
    mx.eval(draft.parameters())
    print(f"quantized reference to affine {bits}-bit group-{group}")
cfg = draft.config
print(f"loaded drafter: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
      f"context width {len(cfg.target_layer_ids) * cfg.hidden_size}")

mx.random.seed(20260824)

# The block the session hands the drafter: the anchor token, then masks.
anchor = 1234
inputs = mx.array([[anchor] + [cfg.mask_token_id] * (BLOCK - 1)], dtype=mx.int32)

# Stand-ins for the two things the drafter borrows from the target.
embed_out = (mx.random.normal((1, BLOCK, cfg.hidden_size)) * 0.02).astype(mx.bfloat16)
target_hidden = (mx.random.normal(
    (1, CONTEXT_LEN, len(cfg.target_layer_ids) * cfg.hidden_size)) * 0.02
).astype(mx.bfloat16)
logits = (mx.random.normal((1, BLOCK - LOGITS_START, cfg.vocab_size)) * 2.0).astype(mx.bfloat16)

cache = draft.make_cache()

# `hidden_states` inlined so the borrowed embedding can be a saved array.
h = embed_out
context = draft.hidden_norm(draft.fc(target_hidden))
per_layer = {}
for i, (layer, c) in enumerate(zip(draft.layers, cache)):
    h = layer(h, context, draft.rope, c)
    per_layer[f"ref_layer_{i}"] = h
hidden = draft.norm(h)[:, LOGITS_START:]

path, candidates, _ = draft.candidate_selector.select(hidden, logits, inputs[:, 0], 0.0)
mx.eval(hidden, path, candidates, context)

out = {
    "inputs": inputs,
    "embed_out": embed_out,
    "target_hidden": target_hidden,
    "logits": logits,
    "ref_context": context,
    "ref_hidden": hidden,
    "ref_path": path.astype(mx.int32),
    "ref_candidates": candidates.astype(mx.int32),
    **per_layer,
}
mx.save_safetensors(OUT, out, metadata={
    "context_len": str(CONTEXT_LEN),
    "block": str(BLOCK),
    "logits_start": str(LOGITS_START),
    "cache_offset_after": str(cache[0].offset),
})
for k, v in out.items():
    print(f"  {k:18s} {str(v.shape):20s} {v.dtype}")
print("path:", path.tolist())
print("draft cache offset after one round:", cache[0].offset)
