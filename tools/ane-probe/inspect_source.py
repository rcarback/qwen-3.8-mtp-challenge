#!/usr/bin/env python3
"""Inventory the MLX 4-bit Qwen3.5-0.8B source checkpoint."""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

SOURCE = Path("/Users/carback1/.cache/mlxfast/qwen3.5-0.8b-src/model.safetensors")
CONFIG = Path("/Users/carback1/.cache/mlxfast/qwen3.5-0.8b-src/config.json")


def n_elems(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def main() -> None:
    with SOURCE.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)

    config = json.loads(CONFIG.read_text())
    text_cfg = config["text_config"]

    dtype_tensors: Counter[str] = Counter()
    prefix: Counter[str] = Counter()
    packed_u32 = 0
    unquant = 0
    scale_elems = 0
    bias_elems = 0
    text_logical = 0
    vision_logical = 0
    text_on_disk = 0
    vision_on_disk = 0

    for name, info in header.items():
        dt = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]
        nbytes = int(offsets[1]) - int(offsets[0])
        dtype_tensors[dt] += 1
        prefix[name.split(".")[0]] += 1
        e = n_elems(shape)
        is_text = name.startswith("language_model")
        if is_text:
            text_on_disk += nbytes
        else:
            vision_on_disk += nbytes

        if name.endswith(".weight") and dt == "U32":
            packed_u32 += e
            logical = e * 8  # 4-bit, 8 values per uint32
        elif name.endswith(".scales"):
            scale_elems += e
            logical = 0
        elif name.endswith(".biases"):
            bias_elems += e
            logical = 0
        else:
            unquant += e
            logical = e

        if is_text:
            text_logical += logical
        else:
            vision_logical += logical

    report = {
        "source": str(SOURCE),
        "file_bytes": SOURCE.stat().st_size,
        "n_tensors": len(header),
        "dtype_tensor_counts": dict(dtype_tensors),
        "top_level_prefixes": dict(prefix),
        "config_declared": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "quantization": config.get("quantization"),
            "hidden_size": text_cfg["hidden_size"],
            "num_hidden_layers": text_cfg["num_hidden_layers"],
            "vocab_size": text_cfg["vocab_size"],
            "text_dtype": text_cfg.get("dtype"),
            "tie_word_embeddings": text_cfg.get("tie_word_embeddings"),
        },
        "on_disk": {
            "packed_u32_elems": packed_u32,
            "unquantized_elems": unquant,
            "scale_elems": scale_elems,
            "bias_elems": bias_elems,
            "text_bytes": text_on_disk,
            "vision_bytes": vision_on_disk,
        },
        "logical_params": {
            "text": text_logical,
            "vision": vision_logical,
            "total": text_logical + vision_logical,
            "note": (
                "U32 packed weights counted as 8 logical 4-bit values each; "
                "scales/biases are quantization metadata, not parameters."
            ),
        },
        "actual_dtype": (
            "MLX affine 4-bit (group_size=64) packed in U32, with BF16 scales/"
            "biases and a mix of BF16 (norms, conv1d, dt_bias) and F32 (A_log) "
            "unquantized tensors. config.json text_config.dtype is bfloat16; "
            "that is the dequantized compute dtype, not the on-disk weight dtype."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
