#!/usr/bin/env python3
"""Convert the 27B-geometry attention block to Core ML fp16 and int8 packages."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

from attn_block import SEED, build_block, parameter_count

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
SEQS_DEFAULT = (16, 128, 512, 1024, 2048)


def _convert_traced(traced, inputs, outputs, label: str) -> ct.models.MLModel:
    print(f"coremltools.convert {label} ...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )
    elapsed = time.time() - t0
    print(f"  converted {label} in {elapsed:.1f}s", flush=True)
    return mlmodel, elapsed


def _quantize_int8(mlmodel: ct.models.MLModel, label: str) -> tuple[ct.models.MLModel, float]:
    print(f"linear_quantize_weights int8 {label} ...", flush=True)
    t0 = time.time()
    cfg = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    q = linear_quantize_weights(mlmodel, cfg)
    elapsed = time.time() - t0
    print(f"  quantized {label} in {elapsed:.1f}s", flush=True)
    return q, elapsed


def _save(mlmodel: ct.models.MLModel, path: Path) -> int:
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"  saved {path.name} ({size / 1e6:.1f} MB)", flush=True)
    return size


def convert_seq(seq_len: int, artifacts: Path, seed: int = SEED) -> dict:
    block, geo = build_block(seq_len, seed=seed)
    example = torch.randn(1, seq_len, geo["hidden_size"], dtype=torch.float32)
    # Keep a deterministic example for tracing (weights already seeded).
    g = torch.Generator().manual_seed(seed + seq_len)
    example = torch.randn(1, seq_len, geo["hidden_size"], generator=g)

    print(f"tracing attn block seq={seq_len} ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(block, example, strict=False)
        _ = traced(example)
    trace_s = time.time() - t0
    print(f"  traced in {trace_s:.1f}s", flush=True)

    result = {
        "seq_len": seq_len,
        "geo": geo,
        "params": parameter_count(block),
        "seed": seed,
        "trace_s": trace_s,
        "fp16": None,
        "int8": None,
        "errors": {},
    }

    tag = f"attn27b_s{seq_len}"
    try:
        fp16, convert_s = _convert_traced(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, seq_len, geo["hidden_size"]),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden", dtype=np.float16)],
            label=f"{tag}-fp16",
        )
        fp16_path = artifacts / f"{tag}_fp16.mlpackage"
        size = _save(fp16, fp16_path)
        result["fp16"] = {
            "path": str(fp16_path),
            "convert_s": convert_s,
            "bytes": size,
        }
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["fp16"] = f"{type(exc).__name__}: {exc}"
        graph_path = artifacts / f"traced_{tag}.graph.txt"
        graph_path.write_text(str(traced.graph))
        print("wrote traced graph to", graph_path, flush=True)
        return result

    try:
        int8, quant_s = _quantize_int8(fp16, tag)
        int8_path = artifacts / f"{tag}_int8.mlpackage"
        size = _save(int8, int8_path)
        result["int8"] = {
            "path": str(int8_path),
            "quantize_s": quant_s,
            "bytes": size,
        }
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["int8"] = f"{type(exc).__name__}: {exc}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqs",
        default=",".join(str(s) for s in SEQS_DEFAULT),
        help="comma-separated sequence lengths",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    seqs = [int(x) for x in args.seqs.split(",") if x.strip()]
    artifacts = ARTIFACTS
    artifacts.mkdir(parents=True, exist_ok=True)

    meta = {"seqs": seqs, "seed": args.seed, "results": {}}
    for seq in seqs:
        print("=" * 60, flush=True)
        print(f"CONVERT seq={seq}", flush=True)
        meta["results"][str(seq)] = convert_seq(seq, artifacts, seed=args.seed)
        (artifacts / "convert_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", artifacts / "convert_meta.json", flush=True)


if __name__ == "__main__":
    main()
