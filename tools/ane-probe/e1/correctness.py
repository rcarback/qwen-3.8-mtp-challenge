#!/usr/bin/env python3
"""Verify Core ML attention block against the PyTorch reference (same random weights)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

from attn_block import SEED, build_block

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a.ravel(), b.ravel()) / denom) if denom else None
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cos,
        "finite_a": bool(np.isfinite(a).all()),
        "finite_b": bool(np.isfinite(b).all()),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
    }


def _example(seq_len: int, hidden: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 10_000 + seq_len)
    return torch.randn(1, seq_len, hidden, generator=g)


def predict_coreml(path: Path, hidden: np.ndarray) -> np.ndarray:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = model.predict({"hidden_states": hidden.astype(np.float16)})
    key = "output_hidden" if "output_hidden" in out else list(out.keys())[0]
    return np.array(out[key])


def check_one(seq_len: int, precision: str, seed: int) -> dict:
    path = ARTIFACTS / f"attn27b_s{seq_len}_{precision}.mlpackage"
    report: dict = {"seq_len": seq_len, "precision": precision, "path": str(path)}
    if not path.exists():
        report["error"] = "missing mlpackage"
        return report

    block, geo = build_block(seq_len, seed=seed)
    x = _example(seq_len, geo["hidden_size"], seed)
    with torch.no_grad():
        ref = block(x).cpu().numpy()
        ref2 = block(x).cpu().numpy()
    report["ref_repeat"] = _stats(ref, ref2)
    report["ref_finite"] = bool(np.isfinite(ref).all())

    try:
        pred = predict_coreml(path, x.numpy())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["comparison"] = _stats(ref, pred)
    print(
        f"seq={seq_len} {precision}: cosine={report['comparison']['cosine']:.6f} "
        f"max_abs={report['comparison']['max_abs']:.4f} "
        f"rmse={report['comparison']['rmse']:.4f} "
        f"finite={report['comparison']['finite_a'] and report['comparison']['finite_b']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="16,128")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    seqs = [int(x) for x in args.seqs.split(",") if x.strip()]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    report = {"seed": args.seed, "checks": []}
    for seq in seqs:
        for precision in ("fp16", "int8"):
            report["checks"].append(check_one(seq, precision, args.seed))

    out = ARTIFACTS / "correctness.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
