# Reference logits for a tiny qwen4_exp source tree; requires mlx and mlx-lm.
# Usage: uv run --with mlx==0.32.0 --with mlx-lm python3 tools/qwen38-flash/parity.py <source-dir>
import json
import os
import sys

import mlx.core as mx

# The PR file is a package module: it imports its siblings relatively
# ("from .base import ..."), so it cannot be imported as a standalone file.
# Load it under its fully qualified name instead, which makes those relative
# imports resolve against the installed mlx_lm.models package.
import importlib.util  # noqa: E402

import mlx_lm.models  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location(
    "mlx_lm.models.qwen4_exp",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen4_exp.py"),
)
qwen4_exp = importlib.util.module_from_spec(_spec)
sys.modules["mlx_lm.models.qwen4_exp"] = qwen4_exp
_spec.loader.exec_module(qwen4_exp)

src = sys.argv[1]
cfg = json.load(open(f"{src}/config.json"))
args = qwen4_exp.ModelArgs.from_dict(cfg)
model = qwen4_exp.Model(args)
weights = {}
for f in sorted(p for p in os.listdir(src) if p.endswith(".safetensors")):
    weights.update(mx.load(f"{src}/{f}"))
model.load_weights(list(model.sanitize(weights).items()), strict=True)
model.set_dtype(mx.float32)

# Match the Swift transform, which stores the routed experts as 4-bit affine at
# group size 32 and leaves every other tensor alone. Comparing a quantized tree
# against an unquantized reference measures quantization noise, not the port.
import mlx.nn as nn  # noqa: E402

from mlx_lm.models.switch_layers import SwitchLinear  # noqa: E402

nn.quantize(
    model, group_size=32, bits=4,
    class_predicate=lambda _p, m: isinstance(m, SwitchLinear),
)

# The Swift tree stores every float tensor as bf16, including the quantization
# scales and biases that nn.quantize just produced in float32. Round the
# reference's the same way. Tensors that came from the bf16 source are already
# bf16 values, so this only touches the scales and biases.
from mlx.utils import tree_map  # noqa: E402

model.update(
    tree_map(
        lambda a: a.astype(mx.bfloat16).astype(mx.float32)
        if a.dtype == mx.float32 else a,
        model.parameters(),
    )
)
ple = model.model.layers[1].ple.ple_embedding
print(json.dumps({
    "multipliers": ple._mults.tolist(),
    "sizes": ple._sizes.tolist(),
    "offsets": ple._offsets.tolist(),
    "logits": model(mx.array([[1, 2, 3, 4, 5]])).tolist(),
}))
