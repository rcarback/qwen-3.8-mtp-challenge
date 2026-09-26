#!/usr/bin/env python3
"""Convert isolated op-class graphs to Core ML fp16 and int8 packages."""

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

from catalog import ARTIFACTS, SEED, build_cells, load_geometry, package_path

OP_SEED = {
    "large_gemm_up": 1,
    "large_gemm_down": 2,
    "attn_proj": 3,
    "attn_scores": 4,
    "elem_swiglu": 5,
    "mtp_fc": 6,
    "mtp_mlp_up": 7,
}
from models import (
    AttentionProjections,
    LinearGEMM,
    RMSSiLUSwiGLU,
    ScoreMatmuls,
    init_linear_,
)

GEO = load_geometry()


def _save(mlmodel: ct.models.MLModel, path: Path) -> int:
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"  saved {path.name} ({size / 1e6:.1f} MB)", flush=True)
    return size


def build_module(cell):
    geo = GEO
    if cell.op in ("large_gemm_up", "large_gemm_down", "mtp_fc", "mtp_mlp_up"):
        mod = LinearGEMM(cell.in_features, cell.out_features)
        init_linear_(mod, SEED + OP_SEED[cell.op])
        example = (
            torch.randn(1, cell.seq, cell.in_features),
        )
        inputs = [
            ct.TensorType(
                name="x",
                shape=(1, cell.seq, cell.in_features),
                dtype=np.float16,
            )
        ]
        outputs = [ct.TensorType(name="y", dtype=np.float16)]
        return mod, example, inputs, outputs
    if cell.op == "attn_proj":
        mod = AttentionProjections(
            geo["hidden_size"],
            geo["num_attention_heads"],
            geo["num_key_value_heads"],
            geo["head_dim"],
        )
        init_linear_(mod, SEED)
        example = (
            torch.randn(1, cell.seq, geo["hidden_size"]),
            torch.randn(1, cell.seq, geo["o_in"]),
        )
        inputs = [
            ct.TensorType(
                name="hidden_states",
                shape=(1, cell.seq, geo["hidden_size"]),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="attn_out",
                shape=(1, cell.seq, geo["o_in"]),
                dtype=np.float16,
            ),
        ]
        outputs = [
            ct.TensorType(name="q", dtype=np.float16),
            ct.TensorType(name="k", dtype=np.float16),
            ct.TensorType(name="v", dtype=np.float16),
            ct.TensorType(name="o", dtype=np.float16),
        ]
        return mod, example, inputs, outputs
    if cell.op == "attn_scores":
        mod = ScoreMatmuls(
            geo["num_attention_heads"],
            geo["num_key_value_heads"],
            geo["head_dim"],
            cell.seq,
            cell.kv,
        )
        mod.eval()
        example = (
            torch.randn(1, geo["num_attention_heads"], cell.seq, geo["head_dim"]),
            torch.randn(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
            torch.randn(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
        )
        inputs = [
            ct.TensorType(
                name="q",
                shape=(1, geo["num_attention_heads"], cell.seq, geo["head_dim"]),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="k",
                shape=(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="v",
                shape=(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
                dtype=np.float16,
            ),
        ]
        outputs = [ct.TensorType(name="y", dtype=np.float16)]
        return mod, example, inputs, outputs
    if cell.op == "elem_swiglu":
        mod = RMSSiLUSwiGLU(cell.width, eps=float(geo["rms_norm_eps"]))
        init_linear_(mod, SEED)
        example = (
            torch.randn(1, cell.seq, cell.width),
            torch.randn(1, cell.seq, cell.width),
        )
        inputs = [
            ct.TensorType(
                name="gate",
                shape=(1, cell.seq, cell.width),
                dtype=np.float16,
            ),
            ct.TensorType(
                name="up",
                shape=(1, cell.seq, cell.width),
                dtype=np.float16,
            ),
        ]
        outputs = [ct.TensorType(name="y", dtype=np.float16)]
        return mod, example, inputs, outputs
    raise ValueError(cell.op)


def convert_cell(cell, *, force: bool) -> dict:
    fp16_path = package_path(cell, "fp16")
    int8_path = package_path(cell, "int8")
    result = {
        "tag": cell.tag,
        "op": cell.op,
        "seq": cell.seq,
        "kv": cell.kv,
        "flops": cell.flops,
        "has_weights": cell.has_weights,
        "fp16": None,
        "int8": None,
        "errors": {},
        "skipped": {},
    }
    if fp16_path.exists() and not force:
        size = sum(p.stat().st_size for p in fp16_path.rglob("*") if p.is_file())
        result["fp16"] = {"path": str(fp16_path), "bytes": size, "reused": True}
        if cell.has_weights:
            if int8_path.exists():
                size8 = sum(
                    p.stat().st_size for p in int8_path.rglob("*") if p.is_file()
                )
                result["int8"] = {
                    "path": str(int8_path),
                    "bytes": size8,
                    "reused": True,
                }
            else:
                result["skipped"]["int8"] = "fp16 reused but int8 missing; rerun --force"
        else:
            result["skipped"]["int8"] = "no weights to quantize"
        print(f"reuse {cell.tag}", flush=True)
        return result

    print(f"trace {cell.tag} ...", flush=True)
    mod, example, inputs, outputs = build_module(cell)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(mod, example, strict=False)
        _ = traced(*example)
    result["trace_s"] = time.time() - t0
    print(f"  traced in {result['trace_s']:.1f}s", flush=True)

    try:
        print(f"  convert fp16 {cell.tag} ...", flush=True)
        t0 = time.time()
        fp16 = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=inputs,
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )
        convert_s = time.time() - t0
        size = _save(fp16, fp16_path)
        result["fp16"] = {
            "path": str(fp16_path),
            "convert_s": convert_s,
            "bytes": size,
        }
        print(f"  converted fp16 in {convert_s:.1f}s", flush=True)
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["fp16"] = f"{type(exc).__name__}: {exc}"
        return result

    if not cell.has_weights:
        result["skipped"]["int8"] = "no weights to quantize"
        return result

    try:
        print(f"  quantize int8 {cell.tag} ...", flush=True)
        t0 = time.time()
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8"
            )
        )
        int8 = linear_quantize_weights(fp16, cfg)
        quant_s = time.time() - t0
        size = _save(int8, int8_path)
        result["int8"] = {
            "path": str(int8_path),
            "quantize_s": quant_s,
            "bytes": size,
        }
        print(f"  quantized int8 in {quant_s:.1f}s", flush=True)
    except Exception as exc:
        traceback.print_exc()
        result["errors"]["int8"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    seqs = {int(x) for x in args.seqs.split(",") if x.strip()}
    cells = build_cells(GEO)
    if ops:
        cells = [c for c in cells if c.op in ops]
    if seqs:
        cells = [c for c in cells if c.seq in seqs]
    meta = {"geo": GEO, "n_cells": len(cells), "results": []}
    for cell in cells:
        print("=" * 60, flush=True)
        r = convert_cell(cell, force=args.force)
        meta["results"].append(r)
        (ARTIFACTS / "convert_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
    print("wrote", ARTIFACTS / "convert_meta.json", flush=True)


if __name__ == "__main__":
    main()
