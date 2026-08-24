"""Produce an affine group-64 quantized DFlash2 head from the published BF16 tree."""
import os, sys, json
sys.path.insert(0, os.environ.get("DFLASH2_REFERENCE_PATH", "."))
from pathlib import Path
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from dflash.model_mlx import load_draft

SRC, DST, BITS, GROUP = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
draft = load_draft(str(SRC))
nn.quantize(draft, group_size=GROUP, bits=BITS)
mx.eval(draft.parameters())

DST.mkdir(parents=True, exist_ok=True)
weights = dict(tree_flatten(draft.parameters()))
# Undo the load-time codebook rename so the artifact keeps the published layout.
for name in ("predecessor_codebook", "successor_codebook"):
    key = f"candidate_selector.{name}.weight"
    if key in weights:
        weights[key.removesuffix(".weight")] = weights.pop(key)
mx.save_safetensors(str(DST / "model.safetensors"), weights)

config = json.loads((SRC / "config.json").read_text())
config["quantization"] = {"group_size": GROUP, "bits": BITS, "mode": "affine"}
(DST / "config.json").write_text(json.dumps(config, indent=2) + "\n")
size = (DST / "model.safetensors").stat().st_size
print(f"{len(weights)} tensors, {size / 2**30:.2f} GiB -> {DST}")
