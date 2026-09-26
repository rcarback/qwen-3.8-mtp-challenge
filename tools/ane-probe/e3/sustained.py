#!/usr/bin/env python3
"""Sustained (>=60s) loop on the heaviest cells, ANE / GPU / Metal."""

from __future__ import annotations

import argparse
import json
import time

from catalog import ARTIFACTS, UNITS, build_cells, inspect_placement, package_path, tflops
from measure_coreml import BACKENDS, capture_compile, make_feeds
from measure_metal import build_metal_op, _eval
import mlx.core as mx

from catalog import start_macmon, stop_macmon, summarize_power


DEFAULT_TAGS = (
    "large_gemm_up_s2048",
    "large_gemm_down_s2048",
    "attn_scores_s1024_kv16384",
)


def _bins(samples: list[tuple[float, float]], width_s: float = 5.0) -> list[dict]:
    if not samples:
        return []
    t0 = samples[0][0]
    buckets: dict[int, list[float]] = {}
    for t, ms in samples:
        b = int((t - t0) / width_s)
        buckets.setdefault(b, []).append(ms)
    out = []
    for b in sorted(buckets):
        vals = buckets[b]
        out.append(
            {
                "t0_s": b * width_s,
                "t1_s": (b + 1) * width_s,
                "n": len(vals),
                "ms_mean": sum(vals) / len(vals),
                "ms_median": sorted(vals)[len(vals) // 2],
            }
        )
    return out


def decay(bins: list[dict]) -> dict:
    if len(bins) < 2:
        return {"insufficient_bins": True}
    first = bins[0]["ms_mean"]
    last = bins[-1]["ms_mean"]
    mid = bins[len(bins) // 2]["ms_mean"]
    return {
        "first_bin_ms": first,
        "mid_bin_ms": mid,
        "last_bin_ms": last,
        "slowdown_last_over_first": (last / first) if first else None,
        "throughput_decayed": bool(first and last > first * 1.10),
    }


def run_coreml(cell, precision: str, backend: str, duration_s: float) -> dict:
    path = package_path(cell, precision)
    units = BACKENDS[backend]
    compile_s, stderr, model = capture_compile(path, UNITS[units])
    out = {
        "tag": cell.tag,
        "backend": backend,
        "precision": precision,
        "compile_wall_s": compile_s,
        "kind": "MEASURED",
    }
    if model is None:
        out["error"] = "load failed"
        out["compile_stderr"] = stderr[-2000:]
        return out
    try:
        out["placement"] = inspect_placement(path, UNITS[units])
    except Exception as exc:
        out["placement"] = {"error": str(exc)}
    feeds = make_feeds(cell)
    for _ in range(3):
        model.predict(feeds)
    proc, rows, thread = start_macmon()
    time.sleep(1.0)
    samples = []
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        t1 = time.perf_counter()
        model.predict(feeds)
        samples.append((t1, (time.perf_counter() - t1) * 1000.0))
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    stop_macmon(proc, thread)
    bins = _bins(samples)
    out.update(
        {
            "elapsed_s": elapsed,
            "n": len(samples),
            "ms_mean": sum(s[1] for s in samples) / len(samples),
            "ms_first": samples[0][1],
            "ms_last": samples[-1][1],
            "bins_5s": bins,
            "decay": decay(bins),
            "power": summarize_power(list(rows)),
            "tflops_mean": tflops(cell.flops, sum(s[1] for s in samples) / len(samples)),
        }
    )
    del model
    return out


def run_metal(cell, duration_s: float) -> dict:
    op = build_metal_op(cell)
    for _ in range(5):
        _eval(op())
    mx.synchronize()
    proc, rows, thread = start_macmon()
    time.sleep(1.0)
    samples = []
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        mx.synchronize()
        t1 = time.perf_counter()
        _eval(op())
        mx.synchronize()
        samples.append((t1, (time.perf_counter() - t1) * 1000.0))
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    stop_macmon(proc, thread)
    bins = _bins(samples)
    return {
        "tag": cell.tag,
        "backend": "metal",
        "precision": "affine-4-group-64"
        if cell.op not in ("attn_scores", "elem_swiglu")
        else "fp16",
        "kernel_label": op.label,
        "kind": "MEASURED",
        "elapsed_s": elapsed,
        "n": len(samples),
        "ms_mean": sum(s[1] for s in samples) / len(samples),
        "ms_first": samples[0][1],
        "ms_last": samples[-1][1],
        "bins_5s": bins,
        "decay": decay(bins),
        "power": summarize_power(list(rows)),
        "tflops_mean": tflops(cell.flops, sum(s[1] for s in samples) / len(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--backends", default="ane,gpu,metal")
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    wanted = [t.strip() for t in args.tags.split(",") if t.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    cells = {c.tag: c for c in build_cells()}
    results = []
    out_path = ARTIFACTS / "sustained.json"
    for tag in wanted:
        if tag not in cells:
            results.append({"tag": tag, "error": "unknown tag"})
            continue
        cell = cells[tag]
        print("=" * 60, f"SUSTAINED {tag}", flush=True)
        if "ane" in backends:
            print("  ANE ...", flush=True)
            results.append(run_coreml(cell, args.precision, "ane", args.duration_s))
            out_path.write_text(json.dumps({"results": results}, indent=2, default=str))
        if "gpu" in backends:
            print("  GPU ...", flush=True)
            results.append(run_coreml(cell, args.precision, "gpu", args.duration_s))
            out_path.write_text(json.dumps({"results": results}, indent=2, default=str))
        if "metal" in backends:
            print("  Metal ...", flush=True)
            results.append(run_metal(cell, args.duration_s))
            out_path.write_text(json.dumps({"results": results}, indent=2, default=str))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
