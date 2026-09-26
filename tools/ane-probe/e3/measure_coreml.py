#!/usr/bin/env python3
"""Time / place / power Core ML cells on ANE (CPU_AND_NE) and GPU (CPU_AND_GPU)."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import time
import traceback
from pathlib import Path

import numpy as np

import coremltools as ct

from catalog import (
    ARTIFACTS,
    SEED,
    UNITS,
    build_cells,
    engine_watts,
    inspect_placement,
    load_geometry,
    package_path,
    perf_per_watt,
    start_macmon,
    stop_macmon,
    summarize_power,
    tflops,
)

GEO = load_geometry()
BACKENDS = {
    "ane": "cpuAndNeuralEngine",
    "gpu": "cpuAndGPU",
}


def make_feeds(cell) -> dict:
    rng = np.random.default_rng(SEED + cell.seq + cell.kv)
    geo = GEO

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    if cell.op in ("large_gemm_up", "large_gemm_down", "mtp_fc", "mtp_mlp_up"):
        return {"x": rand(1, cell.seq, cell.in_features)}
    if cell.op == "attn_proj":
        return {
            "hidden_states": rand(1, cell.seq, geo["hidden_size"]),
            "attn_out": rand(1, cell.seq, geo["o_in"]),
        }
    if cell.op == "attn_scores":
        return {
            "q": rand(1, geo["num_attention_heads"], cell.seq, geo["head_dim"]),
            "k": rand(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
            "v": rand(1, geo["num_key_value_heads"], cell.kv, geo["head_dim"]),
        }
    if cell.op == "elem_swiglu":
        return {
            "gate": rand(1, cell.seq, cell.width),
            "up": rand(1, cell.seq, cell.width),
        }
    raise ValueError(cell.op)


def capture_compile(mlpackage: Path, compute_units: ct.ComputeUnit):
    buf = io.StringIO()
    model = None
    t0 = time.perf_counter()
    err = None
    try:
        with contextlib.redirect_stderr(buf):
            model = ct.models.MLModel(str(mlpackage), compute_units=compute_units)
            _ = model.get_compiled_model_path()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    elapsed = time.perf_counter() - t0
    text = buf.getvalue()
    if err:
        text = (text + "\n" + err).strip()
    return elapsed, text, model


def time_predicts(model, feeds, *, warmup: int, timed_runs: int, min_duration_s: float):
    for _ in range(warmup):
        model.predict(feeds)
    samples: list[float] = []
    t_loop0 = time.perf_counter()
    while len(samples) < timed_runs or (time.perf_counter() - t_loop0) < min_duration_s:
        t0 = time.perf_counter()
        model.predict(feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if len(samples) >= max(timed_runs * 4, 20) and (
            time.perf_counter() - t_loop0
        ) >= min_duration_s:
            break
    samples_sorted = sorted(samples)
    return {
        "n": len(samples),
        "warmup": warmup,
        "ms_best": samples_sorted[0],
        "ms_median": samples_sorted[len(samples_sorted) // 2],
        "ms_mean": sum(samples) / len(samples),
        "ms_p90": samples_sorted[int(0.9 * (len(samples_sorted) - 1))],
        "ms_worst": samples_sorted[-1],
        "samples_ms": samples,
    }


def power_during(model, feeds, duration_s: float):
    proc, rows, thread = start_macmon()
    time.sleep(1.0)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        model.predict(feeds)
        n += 1
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    stop_macmon(proc, thread)
    return n, elapsed, summarize_power(list(rows))


def measure_one(
    cell,
    precision: str,
    backend: str,
    *,
    warmup: int,
    timed_runs: int,
    power_s: float,
) -> dict:
    units_label = BACKENDS[backend]
    path = package_path(cell, precision)
    out: dict = {
        "tag": cell.tag,
        "op": cell.op,
        "seq": cell.seq,
        "kv": cell.kv,
        "flops": cell.flops,
        "gflop": cell.gflop,
        "precision": precision,
        "backend": backend,
        "units": units_label,
        "mlpackage": str(path),
        "kind": "MEASURED",
        "kernel_label": f"coreml-{precision}-{units_label}",
    }
    if not path.exists():
        out["error"] = "missing mlpackage"
        return out
    print(
        f"\n== {backend} {precision} {cell.tag} ==",
        flush=True,
    )
    compile_s, stderr_text, model = capture_compile(path, UNITS[units_label])
    out["compile_wall_s"] = compile_s
    out["compile_stderr"] = stderr_text[-4000:]
    if model is None:
        out["error"] = "model load failed"
        return out
    feeds = make_feeds(cell)
    try:
        placement = inspect_placement(path, UNITS[units_label])
        out["placement"] = placement
        print(
            f"  placement {placement.get('counts_compute')} "
            f"ane_frac={placement.get('ane_fraction_of_compute_ops')}",
            flush=True,
        )
    except Exception as exc:
        out["placement"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  placement failed:", exc, flush=True)

    local_runs = timed_runs
    local_min = 2.0
    if cell.seq >= 1024 or cell.kv >= 16384:
        local_runs = max(5, timed_runs // 2)
        local_min = 3.0
    try:
        timing = time_predicts(
            model,
            feeds,
            warmup=warmup,
            timed_runs=local_runs,
            min_duration_s=local_min,
        )
        out["timing"] = timing
        ms = timing["ms_median"]
        out["ms"] = ms
        out["tflops"] = tflops(cell.flops, ms)
        print(
            f"  timing best={timing['ms_best']:.2f} med={ms:.2f} "
            f"tflops={out['tflops']}",
            flush=True,
        )
    except Exception as exc:
        out["timing"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  timing failed:", exc, flush=True)

    try:
        n, elapsed, power = power_during(model, feeds, power_s)
        out["power"] = {
            "loops": n,
            "elapsed_s": elapsed,
            "ms_per_predict_in_window": (elapsed / n) * 1000.0 if n else None,
            **power,
        }
        watts = engine_watts(power, backend)
        out["engine_watts"] = watts
        out["tflops_per_watt"] = (
            perf_per_watt(cell.flops, out["ms"], watts)
            if out.get("ms") and watts
            else None
        )
        print(
            f"  power ane={power['ane_power'].get('mean')} "
            f"gpu={power['gpu_power'].get('mean')} "
            f"cpu={power['cpu_power'].get('mean')} "
            f"engine_W={watts} tflops/W={out['tflops_per_watt']}",
            flush=True,
        )
    except Exception as exc:
        out["power"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  power failed:", exc, flush=True)

    del model
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--backends", default="ane,gpu")
    parser.add_argument("--precisions", default="fp16,int8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=12)
    parser.add_argument("--power-s", type=float, default=6.0)
    parser.add_argument("--out", default="measure_coreml.json")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    seqs = {int(x) for x in args.seqs.split(",") if x.strip()}
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    precisions = [x.strip() for x in args.precisions.split(",") if x.strip()]
    cells = build_cells(GEO)
    if ops:
        cells = [c for c in cells if c.op in ops]
    if seqs:
        cells = [c for c in cells if c.seq in seqs]

    out_path = ARTIFACTS / args.out
    results: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = list(prev.get("results") or [])
        except Exception:
            results = []
    done = {
        (r.get("tag"), r.get("precision"), r.get("backend"))
        for r in results
        if not r.get("error")
    }

    # ANE first so GPU heat does not contaminate ANE cells.
    order = [b for b in ("ane", "gpu") if b in backends]
    for backend in order:
        for cell in cells:
            for precision in precisions:
                if precision == "int8" and not cell.has_weights:
                    continue
                key = (cell.tag, precision, backend)
                if key in done:
                    print(f"skip existing {key}", flush=True)
                    continue
                r = measure_one(
                    cell,
                    precision,
                    backend,
                    warmup=args.warmup,
                    timed_runs=args.timed_runs,
                    power_s=args.power_s,
                )
                results.append(r)
                payload = {
                    "machine": "Apple M4 Max",
                    "geo": GEO,
                    "results": results,
                }
                out_path.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
