# Reference logits for a tiny qwen4_exp source tree; requires mlx and mlx-lm.
# Usage: uv run --with mlx==0.32.0 --with mlx-lm python3 tools/qwen38-flash/parity.py <source-dir>
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_exp  # noqa: E402

src = sys.argv[1]
cfg = json.load(open(f"{src}/config.json"))
args = qwen4_exp.ModelArgs.from_dict(cfg)
model = qwen4_exp.Model(args)
weights = {}
for f in sorted(p for p in os.listdir(src) if p.endswith(".safetensors")):
    weights.update(mx.load(f"{src}/{f}"))
model.load_weights(list(model.sanitize(weights).items()), strict=True)
model.set_dtype(mx.float32)
ple = model.model.layers[1].ple.ple_embedding
print(json.dumps({
    "multipliers": ple._mults.tolist(),
    "sizes": ple._sizes.tolist(),
    "offsets": ple._offsets.tolist(),
    "logits": model(mx.array([[1, 2, 3, 4, 5]])).tolist(),
}))
