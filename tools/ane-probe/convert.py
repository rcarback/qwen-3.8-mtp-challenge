#!/usr/bin/env python3
"""Convert the Qwen3.5-0.8B text tower to Core ML fp16 and int8 mlpackages."""

from __future__ import annotations

import argparse
import json
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

from qwen35_torch import (
    SEQ_LEN,
    build_loaded_model,
    rms_norm,
    tokenize_fixed_prompt,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
FP16_NAME = "qwen35_08b_text_fp16.mlpackage"
INT8_NAME = "qwen35_08b_text_int8.mlpackage"
BODY_FP16_NAME = "qwen35_08b_body_fp16.mlpackage"
BODY_INT8_NAME = "qwen35_08b_body_int8.mlpackage"


class FullWindow(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class BodyWindow(torch.nn.Module):
    """Transformer stack only: hidden_states in, hidden_states out.

    Used if the 248k-vocab embedding/lm_head makes a single package miss the ANE.
    """

    def __init__(self, model):
        super().__init__()
        self.layers = model.layers
        self.final_norm = model.final_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = hidden_states
        for layer in self.layers:
            hidden = layer(hidden)
        return rms_norm(hidden, self.final_norm)


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
    print(f"  converted {label} in {time.time() - t0:.1f}s", flush=True)
    return mlmodel


def _quantize_int8(mlmodel: ct.models.MLModel, label: str) -> ct.models.MLModel:
    print(f"linear_quantize_weights int8 {label} ...", flush=True)
    t0 = time.time()
    cfg = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    q = linear_quantize_weights(mlmodel, cfg)
    print(f"  quantized {label} in {time.time() - t0:.1f}s", flush=True)
    return q


def _save(mlmodel: ct.models.MLModel, path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    mlmodel.save(str(path))
    # mlpackage is a directory
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"  saved {path.name} ({size / 1e6:.1f} MB)", flush=True)


def convert_full(model, artifacts: Path) -> tuple[Path, Path] | None:
    wrapper = FullWindow(model).eval()
    example = tokenize_fixed_prompt(SEQ_LEN)
    print("tracing full window (input_ids -> logits) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        _ = traced(example)
    print(f"  traced in {time.time() - t0:.1f}s", flush=True)
    try:
        fp16 = _convert_traced(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=np.int32)
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            label="full-fp16",
        )
    except Exception:
        traceback.print_exc()
        return None
    fp16_path = artifacts / FP16_NAME
    _save(fp16, fp16_path)
    try:
        int8 = _quantize_int8(fp16, "full")
        int8_path = artifacts / INT8_NAME
        _save(int8, int8_path)
    except Exception:
        traceback.print_exc()
        return fp16_path, None
    return fp16_path, artifacts / INT8_NAME


def _convert_one_layer(model, artifacts: Path, kind: str):
    layer = next(l for l in model.layers if type(l).__name__ == kind)

    class One(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.block(hidden_states)

    wrapper = One(layer).eval()
    example = torch.zeros(1, SEQ_LEN, model.hidden, dtype=torch.float32)
    tag = "gdn0" if kind == "GDNBlock" else "attn0"
    print(f"tracing single {kind} ...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
    try:
        fp16 = _convert_traced(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, SEQ_LEN, model.hidden),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden", dtype=np.float16)],
            label=f"{tag}-fp16",
        )
    except Exception:
        traceback.print_exc()
        (artifacts / f"traced_{tag}.graph.txt").write_text(str(traced.graph))
        print("wrote traced graph to", artifacts / f"traced_{tag}.graph.txt")
        return None
    fp16_path = artifacts / f"qwen35_08b_{tag}_fp16.mlpackage"
    _save(fp16, fp16_path)
    try:
        int8 = _quantize_int8(fp16, tag)
        int8_path = artifacts / f"qwen35_08b_{tag}_int8.mlpackage"
        _save(int8, int8_path)
        return fp16_path, int8_path
    except Exception:
        traceback.print_exc()
        return fp16_path, None


def convert_one_gdn(model, artifacts: Path) -> tuple[Path, Path] | None:
    return _convert_one_layer(model, artifacts, "GDNBlock")


def convert_one_attn(model, artifacts: Path) -> tuple[Path, Path] | None:
    return _convert_one_layer(model, artifacts, "AttnBlock")


def convert_body(model, artifacts: Path) -> tuple[Path, Path] | None:
    wrapper = BodyWindow(model).eval()
    example = torch.zeros(1, SEQ_LEN, model.hidden, dtype=torch.float32)
    print("tracing body window (hidden -> hidden) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
        _ = traced(example)
    print(f"  traced in {time.time() - t0:.1f}s", flush=True)
    try:
        fp16 = _convert_traced(
            traced,
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, SEQ_LEN, model.hidden),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden", dtype=np.float16)],
            label="body-fp16",
        )
    except Exception:
        traceback.print_exc()
        return None
    fp16_path = artifacts / BODY_FP16_NAME
    _save(fp16, fp16_path)
    try:
        int8 = _quantize_int8(fp16, "body")
        int8_path = artifacts / BODY_INT8_NAME
        _save(int8, int8_path)
    except Exception:
        traceback.print_exc()
        return fp16_path, None
    return fp16_path, artifacts / BODY_INT8_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "body", "both", "gdn0", "attn0", "all"],
        default="all",
    )
    args = parser.parse_args()
    artifacts = ARTIFACTS
    artifacts.mkdir(parents=True, exist_ok=True)
    print("loading dequantized MLX 4-bit text tower on CPU ...", flush=True)
    t0 = time.time()
    model = build_loaded_model(SEQ_LEN)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    meta = {
        "seq_len": SEQ_LEN,
        "hidden": model.hidden,
        "vocab": model.vocab,
        "n_layers": model.n_layers,
        "torch_params": int(sum(p.numel() for p in model.parameters())),
        "results": {},
    }
    def _store(got):
        if got is None:
            return None
        return {
            "fp16": None if got[0] is None else str(got[0]),
            "int8": None if got[1] is None else str(got[1]),
        }

    if args.mode in ("gdn0", "all"):
        meta["results"]["gdn0"] = _store(convert_one_gdn(model, artifacts))
    if args.mode in ("attn0", "all"):
        meta["results"]["attn0"] = _store(convert_one_attn(model, artifacts))
    if args.mode in ("full", "both", "all"):
        meta["results"]["full"] = _store(convert_full(model, artifacts))
    if args.mode in ("body", "both", "all"):
        meta["results"]["body"] = _store(convert_body(model, artifacts))
    (artifacts / "convert_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", artifacts / "convert_meta.json")


if __name__ == "__main__":
    main()
