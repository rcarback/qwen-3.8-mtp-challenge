#!/usr/bin/env python3
"""M3: int8 vs fp16 quality distribution on the E1 attention block, seq=1024.

Compares Core ML int8 against:
  - the Core ML fp16 package (the fp16 reference the ANE path actually runs)
  - the same-weight PyTorch block (E1's one-sample method, now over many draws)

Does not assess end-task quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

from common import E1_ARTIFACTS, SEED, SEQ, ARTIFACTS

sys.path.insert(0, str(E1_ARTIFACTS.parent))
from attn_block import build_block  # noqa: E402


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a.ravel(), b.ravel()) / denom) if denom else None
    return {
        "cosine": cos,
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "finite_a": bool(np.isfinite(a).all()),
        "finite_b": bool(np.isfinite(b).all()),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(vals: list[float]) -> dict:
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "p10": _quantile(s, 0.10),
        "median": _quantile(s, 0.50),
        "p90": _quantile(s, 0.90),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


def _predict(model: ct.models.MLModel, x: np.ndarray) -> np.ndarray:
    out = model.predict({"hidden_states": x.astype(np.float16)})
    key = "output_hidden" if "output_hidden" in out else list(out.keys())[0]
    return np.array(out[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--units",
        default="cpuAndNeuralEngine",
        choices=["cpuOnly", "cpuAndNeuralEngine"],
    )
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    units = (
        ct.ComputeUnit.CPU_AND_NE
        if args.units == "cpuAndNeuralEngine"
        else ct.ComputeUnit.CPU_ONLY
    )
    fp16_path = E1_ARTIFACTS / f"attn27b_s{SEQ}_fp16.mlpackage"
    int8_path = E1_ARTIFACTS / f"attn27b_s{SEQ}_int8.mlpackage"
    if not fp16_path.exists() or not int8_path.exists():
        raise SystemExit(f"missing E1 packages: {fp16_path} {int8_path}")

    print(f"loading E1 attention packages units={args.units} ...", flush=True)
    t0 = time.perf_counter()
    fp16_model = ct.models.MLModel(str(fp16_path), compute_units=units)
    int8_model = ct.models.MLModel(str(int8_path), compute_units=units)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    block, geo = build_block(SEQ, seed=args.seed)
    hidden = geo["hidden_size"]

    rows = []
    print(f"running {args.n} random inputs seq={SEQ} ...", flush=True)
    for i in range(args.n):
        g = torch.Generator().manual_seed(args.seed + 50_000 + i)
        x = torch.randn(1, SEQ, hidden, generator=g)
        with torch.no_grad():
            ref = block(x).cpu().numpy()
        x_np = x.numpy()
        pred_fp16 = _predict(fp16_model, x_np)
        pred_int8 = _predict(int8_model, x_np)
        row = {
            "i": i,
            "seed": args.seed + 50_000 + i,
            "int8_vs_fp16": _stats(pred_fp16, pred_int8),
            "int8_vs_pytorch": _stats(ref, pred_int8),
            "fp16_vs_pytorch": _stats(ref, pred_fp16),
        }
        rows.append(row)
        c = row["int8_vs_fp16"]
        print(
            f"  [{i:02d}] int8-vs-fp16 cosine={c['cosine']:.8f} "
            f"max_abs={c['max_abs']:.5f} rmse={c['rmse']:.5f}",
            flush=True,
        )

    def collect(key: str, metric: str) -> list[float]:
        return [float(r[key][metric]) for r in rows]

    distribution = {}
    for pair in ("int8_vs_fp16", "int8_vs_pytorch", "fp16_vs_pytorch"):
        distribution[pair] = {
            "cosine": _summarize(collect(pair, "cosine")),
            "max_abs": _summarize(collect(pair, "max_abs")),
            "rmse": _summarize(collect(pair, "rmse")),
            "all_finite": all(
                r[pair]["finite_a"] and r[pair]["finite_b"] for r in rows
            ),
        }

    report = {
        "n": args.n,
        "seq": SEQ,
        "units": args.units,
        "seed": args.seed,
        "fp16_path": str(fp16_path),
        "int8_path": str(int8_path),
        "distribution": distribution,
        "rows": rows,
        "note": (
            "End-task quality is not assessed. These metrics are tensor-level "
            "agreement on random Gaussian inputs against the E1 attention block."
        ),
    }
    out = ARTIFACTS / "m3_int8_dist.json"
    out.write_text(json.dumps(report, indent=2))
    print("\nDISTRIBUTION int8 vs fp16 Core ML:", flush=True)
    for metric in ("cosine", "max_abs", "rmse"):
        s = distribution["int8_vs_fp16"][metric]
        print(
            f"  {metric}: min={s['min']:.8g} median={s['median']:.8g} max={s['max']:.8g}",
            flush=True,
        )
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
