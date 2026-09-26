#!/usr/bin/env python3
"""Core ML vs PyTorch numerical check on the same random weights."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import coremltools as ct

from catalog import ARTIFACTS, SEED, build_cells, package_path
from convert import build_module


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom else None
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cos,
        "finite_a": bool(np.isfinite(a).all()),
        "finite_b": bool(np.isfinite(b).all()),
        "n": int(n),
    }


def _feeds_and_ref(cell) -> tuple[dict, list[np.ndarray]]:
    mod, example, _, _ = build_module(cell)
    g = torch.Generator().manual_seed(SEED + 10_000 + cell.seq + cell.kv)
    tensors = []
    for t in example:
        tensors.append(torch.randn(t.shape, generator=g))
    with torch.no_grad():
        out = mod(*tensors)
    if isinstance(out, torch.Tensor):
        refs = [out.cpu().numpy()]
    else:
        refs = [o.cpu().numpy() for o in out]
    names_by_n = {
        1: ["x"],
        2: ["hidden_states", "attn_out"]
        if cell.op == "attn_proj"
        else (["gate", "up"] if cell.op == "elem_swiglu" else ["a", "b"]),
        3: ["q", "k", "v"],
    }
    if cell.op in ("large_gemm_up", "large_gemm_down", "mtp_fc", "mtp_mlp_up"):
        names = ["x"]
    elif cell.op == "attn_proj":
        names = ["hidden_states", "attn_out"]
    elif cell.op == "attn_scores":
        names = ["q", "k", "v"]
    elif cell.op == "elem_swiglu":
        names = ["gate", "up"]
    else:
        names = names_by_n[len(tensors)]
    feeds = {n: t.numpy().astype(np.float16) for n, t in zip(names, tensors)}
    return feeds, refs


def check_one(cell, precision: str) -> dict:
    path = package_path(cell, precision)
    report: dict = {
        "tag": cell.tag,
        "op": cell.op,
        "seq": cell.seq,
        "kv": cell.kv,
        "precision": precision,
        "path": str(path),
    }
    if not path.exists():
        report["error"] = "missing mlpackage"
        return report
    feeds, refs = _feeds_and_ref(cell)
    try:
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
        pred = model.predict(feeds)
        del model
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    if cell.op == "attn_proj":
        keys = ["q", "k", "v", "o"]
    else:
        keys = ["y"] if "y" in pred else list(pred.keys())
    comps = []
    for i, key in enumerate(keys):
        if key not in pred:
            comps.append({"key": key, "error": "missing output"})
            continue
        comps.append({"key": key, **_stats(refs[i], np.array(pred[key]))})
    report["comparisons"] = comps
    cosines = [c["cosine"] for c in comps if c.get("cosine") is not None]
    report["min_cosine"] = min(cosines) if cosines else None
    report["pass"] = bool(cosines) and min(cosines) >= (
        0.99 if precision == "int8" else 0.999
    )
    print(
        f"{cell.tag} {precision}: min_cosine={report['min_cosine']} "
        f"pass={report['pass']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument(
        "--only-small",
        action="store_true",
        help="correctness on seq<=128 (or seq=1/2/8) plus one scores kv=4096",
    )
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    seqs = {int(x) for x in args.seqs.split(",") if x.strip()}
    cells = build_cells()
    if ops:
        cells = [c for c in cells if c.op in ops]
    if seqs:
        cells = [c for c in cells if c.seq in seqs]
    if args.only_small:
        picked = []
        seen_ops = set()
        for c in cells:
            if c.op == "attn_scores" and not (c.seq <= 128 and c.kv == 4096):
                continue
            if c.op != "attn_scores" and c.seq > 128 and c.op not in (
                "mtp_fc",
                "mtp_mlp_up",
            ):
                # one representative per remaining op at seq=128 already covers
                # weights; skip larger seqs (same weights, different S).
                continue
            key = (c.op, c.seq if c.op in ("mtp_fc", "mtp_mlp_up") else 0, c.kv)
            if key in seen_ops:
                continue
            seen_ops.add(key)
            picked.append(c)
        cells = picked

    report = {"checks": []}
    for cell in cells:
        precs = ["fp16"]
        if cell.has_weights:
            precs.append("int8")
        for precision in precs:
            report["checks"].append(check_one(cell, precision))
            (ARTIFACTS / "correctness.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
    n = len(report["checks"])
    n_pass = sum(1 for c in report["checks"] if c.get("pass"))
    n_err = sum(1 for c in report["checks"] if c.get("error"))
    print(f"correctness {n_pass}/{n} pass, {n_err} errors", flush=True)
    print("wrote", ARTIFACTS / "correctness.json", flush=True)


if __name__ == "__main__":
    main()
