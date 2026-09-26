#!/usr/bin/env python3
"""Convert gated-delta projection GEMMs (no recurrence) to Core ML fp16 and int8."""

from __future__ import annotations

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

from common import ARTIFACTS, SEED, SEQ
from gdn_projections import (
    build_block,
    gemm_param_breakdown,
    parameter_count,
)


def _save(mlmodel: ct.models.MLModel, path: Path) -> int:
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"  saved {path.name} ({size / 1e6:.1f} MB)", flush=True)
    return size


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    block, geo = build_block(SEQ, seed=SEED)
    g = torch.Generator().manual_seed(SEED + SEQ)
    example = torch.randn(1, SEQ, geo["hidden_size"], generator=g)
    print(
        f"tracing GDN projections seq={SEQ} params_M={parameter_count(block)/1e6:.2f} ...",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(block, example, strict=False)
        _ = traced(example)
    trace_s = time.time() - t0
    print(f"  traced in {trace_s:.1f}s", flush=True)

    result = {
        "seq_len": SEQ,
        "geo": geo,
        "params": parameter_count(block),
        "gemm_params": gemm_param_breakdown(block),
        "seed": SEED,
        "trace_s": trace_s,
        "fp16": None,
        "int8": None,
        "errors": {},
    }
    tag = f"gdnproj27b_s{SEQ}"

    try:
        print(f"coremltools.convert {tag}-fp16 ...", flush=True)
        t0 = time.time()
        fp16 = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, SEQ, geo["hidden_size"]),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )
        convert_s = time.time() - t0
        print(f"  converted {tag}-fp16 in {convert_s:.1f}s", flush=True)
        fp16_path = ARTIFACTS / f"{tag}_fp16.mlpackage"
        size = _save(fp16, fp16_path)
        result["fp16"] = {
            "path": str(fp16_path),
            "convert_s": convert_s,
            "bytes": size,
        }
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["fp16"] = f"{type(exc).__name__}: {exc}"
        graph_path = ARTIFACTS / f"traced_{tag}.graph.txt"
        graph_path.write_text(str(traced.graph))
        print("wrote traced graph to", graph_path, flush=True)
        (ARTIFACTS / "convert_gdn_meta.json").write_text(json.dumps(result, indent=2, default=str))
        return

    try:
        print(f"linear_quantize_weights int8 {tag} ...", flush=True)
        t0 = time.time()
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        int8 = linear_quantize_weights(fp16, cfg)
        quant_s = time.time() - t0
        print(f"  quantized {tag} in {quant_s:.1f}s", flush=True)
        int8_path = ARTIFACTS / f"{tag}_int8.mlpackage"
        size = _save(int8, int8_path)
        result["int8"] = {
            "path": str(int8_path),
            "quantize_s": quant_s,
            "bytes": size,
        }
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["int8"] = f"{type(exc).__name__}: {exc}"

    (ARTIFACTS / "convert_gdn_meta.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print("wrote", ARTIFACTS / "convert_gdn_meta.json", flush=True)


if __name__ == "__main__":
    main()
