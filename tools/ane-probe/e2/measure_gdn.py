#!/usr/bin/env python3
"""Time / place / power the gated-delta projection graph at seq=1024."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import traceback
from pathlib import Path

import numpy as np

import coremltools as ct

from common import (
    ARTIFACTS,
    SEQ,
    UNITS,
    inspect_placement,
    start_macmon,
    stop_macmon,
    summarize_power,
)

THRESHOLD_MS = 114.0


def _capture_compile_stderr(
    mlpackage: Path, compute_units: ct.ComputeUnit
) -> tuple[float, str, object | None]:
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


def time_predicts(
    model: ct.models.MLModel,
    feeds: dict,
    *,
    warmup: int,
    timed_runs: int,
    min_duration_s: float,
) -> dict:
    for _ in range(warmup):
        model.predict(feeds)
    samples: list[float] = []
    t_loop0 = time.perf_counter()
    while len(samples) < timed_runs or (time.perf_counter() - t_loop0) < min_duration_s:
        t0 = time.perf_counter()
        model.predict(feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if len(samples) >= max(timed_runs * 4, 40) and (
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


def power_during(
    model: ct.models.MLModel, feeds: dict, duration_s: float
) -> tuple[int, float, dict]:
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
    mlpackage: Path,
    hidden: int,
    precision: str,
    units_label: str,
    *,
    warmup: int,
    timed_runs: int,
    power_s: float,
) -> dict:
    cu = UNITS[units_label]
    print(f"\n== measure GDN proj seq={SEQ} {precision} units={units_label} ==", flush=True)
    out: dict = {
        "seq_len": SEQ,
        "precision": precision,
        "units": units_label,
        "mlpackage": str(mlpackage),
    }
    compile_s, stderr_text, model = _capture_compile_stderr(mlpackage, cu)
    out["compile_wall_s"] = compile_s
    out["compile_stderr"] = stderr_text
    ane_errs = [
        line
        for line in stderr_text.splitlines()
        if "ANE" in line or "ane" in line.lower() or "Neural" in line or "ANEC" in line
    ]
    out["ane_compile_errors"] = ane_errs
    if model is None:
        out["error"] = "model load failed"
        return out
    print(
        f"  compile/load wall {compile_s:.2f}s; ane_err_lines={len(ane_errs)}",
        flush=True,
    )
    feeds = {"hidden_states": np.zeros((1, SEQ, hidden), dtype=np.float16)}
    try:
        placement = inspect_placement(mlpackage, cu)
        out["placement"] = placement
        print(
            f"  placement compute={placement.get('counts_compute')} "
            f"ane_frac={placement.get('ane_fraction_of_compute_ops')}",
            flush=True,
        )
    except Exception as exc:
        out["placement"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  placement failed:", exc, flush=True)

    try:
        timing = time_predicts(
            model,
            feeds,
            warmup=warmup,
            timed_runs=timed_runs,
            min_duration_s=3.0,
        )
        out["timing"] = timing
        print(
            f"  timing best={timing['ms_best']:.2f} ms median={timing['ms_median']:.2f} "
            f"mean={timing['ms_mean']:.2f} n={timing['n']}",
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
        ane = power["ane_power"]
        print(
            f"  power ane mean/max={ane.get('mean')}/{ane.get('max')} "
            f"n_samples={power['n_samples']} loops={n}",
            flush=True,
        )
    except Exception as exc:
        out["power"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  power failed:", exc, flush=True)

    del model
    return out


def decide(results: list[dict]) -> dict:
    candidates = []
    for r in results:
        if r.get("units") != "cpuAndNeuralEngine":
            continue
        timing = r.get("timing") or {}
        placement = r.get("placement") or {}
        if "ms_best" not in timing:
            continue
        candidates.append(
            {
                "precision": r.get("precision"),
                "ms_best": timing["ms_best"],
                "ms_median": timing.get("ms_median"),
                "ane_fraction": placement.get("ane_fraction_of_compute_ops"),
                "ane_power_mean": ((r.get("power") or {}).get("ane_power") or {}).get(
                    "mean"
                ),
                "ane_compile_errors": r.get("ane_compile_errors") or [],
                "counts_compute": (placement or {}).get("counts_compute"),
            }
        )
    decision = {
        "threshold_ms": THRESHOLD_MS,
        "rule": (
            "PASS if ANE is materially under 114 ms with placement at or near 100%."
        ),
        "candidates": candidates,
        "verdict": "FAIL",
        "reason": "",
    }
    if not candidates:
        decision["reason"] = "no cpuAndNeuralEngine timing result"
        return decision
    near = [
        c
        for c in candidates
        if c.get("ane_fraction") is not None and c["ane_fraction"] >= 0.95
    ]
    pool = near if near else candidates
    best = min(pool, key=lambda c: c["ms_best"])
    decision["best"] = best
    placed = best.get("ane_fraction")
    ms = best["ms_best"]
    if placed is None:
        decision["reason"] = "placement missing for best candidate"
        return decision
    if placed < 0.95:
        decision["reason"] = (
            f"ANE placement {placed:.3f} is below ~100% "
            f"(best ms/pred={ms:.2f})"
        )
        return decision
    if ms >= THRESHOLD_MS:
        decision["reason"] = (
            f"ANE ms/pred={ms:.2f} is NOT under {THRESHOLD_MS:.0f} ms "
            f"(placement={placed:.3f})"
        )
        return decision
    if ms >= THRESHOLD_MS * 0.9:
        decision["reason"] = (
            f"ANE ms/pred={ms:.2f} is under {THRESHOLD_MS:.0f} but not materially "
            f"(need < {THRESHOLD_MS * 0.9:.0f}); placement={placed:.3f}"
        )
        return decision
    decision["verdict"] = "PASS"
    decision["reason"] = (
        f"ANE ms/pred={ms:.2f} materially under {THRESHOLD_MS:.0f} ms "
        f"with placement={placed:.3f}"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precisions", default="fp16,int8")
    parser.add_argument("--units", default="cpuOnly,cpuAndNeuralEngine")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--power-s", type=float, default=8.0)
    args = parser.parse_args()
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for precision in precisions:
        path = ARTIFACTS / f"gdnproj27b_s{SEQ}_{precision}.mlpackage"
        if not path.exists():
            results.append(
                {"precision": precision, "error": f"missing {path}"}
            )
            continue
        for units_label in units:
            r = measure_one(
                path,
                args.hidden,
                precision,
                units_label,
                warmup=args.warmup,
                timed_runs=args.timed_runs,
                power_s=args.power_s,
            )
            results.append(r)
            partial = {"results": results, "decision": decide(results)}
            (ARTIFACTS / "measure_gdn.json").write_text(
                json.dumps(partial, indent=2, default=str)
            )

    decision = decide(results)
    report = {
        "threshold_ms": THRESHOLD_MS,
        "results": results,
        "decision": decision,
    }
    out = ARTIFACTS / "measure_gdn.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\nDECISION:", decision["verdict"], "-", decision["reason"], flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
