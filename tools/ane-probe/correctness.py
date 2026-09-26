#!/usr/bin/env python3
"""Numerical checks: PyTorch reference vs Core ML fp16 vs int8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

from qwen35_torch import (
    SEQ_LEN,
    build_loaded_model,
    tokenize_fixed_prompt,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = a - b
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(np.dot(a.ravel(), b.ravel()) / denom) if denom else None
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cos,
        "argmax_a": int(np.argmax(a.ravel())),
        "argmax_b": int(np.argmax(b.ravel())),
        "argmax_match": int(np.argmax(a.ravel())) == int(np.argmax(b.ravel())),
        "finite_a": bool(np.isfinite(a).all()),
        "finite_b": bool(np.isfinite(b).all()),
    }


def predict_coreml(path: Path, input_ids: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    names = [inp.name for inp in spec.description.input]
    if "input_ids" in names:
        out = model.predict({"input_ids": input_ids.astype(np.int32)})
        key = list(out.keys())[0]
        return np.array(out[key])
    if "hidden_states" in names:
        # body-only package: embed in torch, compare hidden
        return None
    key = names[0]
    out = model.predict({key: input_ids})
    return np.array(list(out.values())[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", default=str(ARTIFACTS / "qwen35_08b_text_fp16.mlpackage"))
    parser.add_argument("--int8", default=str(ARTIFACTS / "qwen35_08b_text_int8.mlpackage"))
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("building CPU PyTorch reference from dequantized MLX 4-bit ...", flush=True)
    model = build_loaded_model(SEQ_LEN)
    ids = tokenize_fixed_prompt(SEQ_LEN)
    with torch.no_grad():
        ref = model(ids).cpu().numpy()
        ref2 = model(ids).cpu().numpy()
    report = {
        "input_ids": ids.tolist(),
        "ref_shape": list(ref.shape),
        "ref_finite": bool(np.isfinite(ref).all()),
        "ref_has_nan": bool(np.isnan(ref).any()),
        "ref_repeat": _stats(ref, ref2),
        "ref_last_argmax": int(np.argmax(ref[0, -1])),
        "ref_last_min": float(ref[0, -1].min()),
        "ref_last_max": float(ref[0, -1].max()),
        "comparisons": {},
    }
    print("ref last_argmax", report["ref_last_argmax"], "repeat", report["ref_repeat"])

    ids_np = ids.numpy()
    for tag, path in (("fp16", Path(args.fp16)), ("int8", Path(args.int8))):
        print("predict", tag, path, flush=True)
        if not path.exists():
            report["comparisons"][tag] = {"error": "missing mlpackage"}
            continue
        pred = predict_coreml(path, ids_np)
        if pred is None:
            report["comparisons"][tag] = {"error": "unsupported input signature"}
            continue
        # broadcast/crop to last-token if needed
        if pred.ndim == 3:
            cmp_a, cmp_b = ref, pred.astype(np.float32)
        elif pred.ndim == 2:
            cmp_a, cmp_b = ref[:, -1, :], pred.astype(np.float32)
        else:
            report["comparisons"][tag] = {"error": f"unexpected rank {pred.ndim}"}
            continue
        if cmp_a.shape != cmp_b.shape:
            # try last token vs last token
            a = ref[0, -1]
            b = pred.reshape(-1, pred.shape[-1])[-1]
            report["comparisons"][tag] = {
                "note": f"shape ref {ref.shape} pred {pred.shape}; compared last token",
                **_stats(a, b),
            }
        else:
            report["comparisons"][tag] = _stats(cmp_a, cmp_b)
        print(tag, report["comparisons"][tag])

    if (
        "fp16" in report["comparisons"]
        and "int8" in report["comparisons"]
        and "max_abs" in report["comparisons"]["fp16"]
        and "max_abs" in report["comparisons"]["int8"]
    ):
        # also fp16 vs int8 if both produced logits of same layout via files
        try:
            p16 = predict_coreml(Path(args.fp16), ids_np)
            p8 = predict_coreml(Path(args.int8), ids_np)
            if p16 is not None and p8 is not None and p16.shape == p8.shape:
                report["comparisons"]["fp16_vs_int8"] = _stats(p16, p8)
        except Exception as exc:
            report["comparisons"]["fp16_vs_int8"] = {"error": str(exc)}

    out = ARTIFACTS / "correctness.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
