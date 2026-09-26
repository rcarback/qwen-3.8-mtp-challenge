#!/usr/bin/env python3
"""Shape-sweep timing / placement / power for the 27B-geometry attention block."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
SEQS_DEFAULT = (16, 128, 512, 1024, 2048)
SKIP_OPS = {"const", "constexpr_blockwise_shift_scale", "constexpr_affine_dequantize"}
UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}

# Decision threshold (fixed in advance).
THRESHOLD_SEQ = 1024
THRESHOLD_MS = 160.0


def _device_label(device) -> str:
    if device is None:
        return "None"
    name = type(device).__name__
    blob = (name + " " + str(device)).lower()
    if "neural" in blob or "ane" in blob:
        return "ANE"
    if "gpu" in blob or "metal" in blob:
        return "GPU"
    if "cpu" in blob:
        return "CPU"
    return name


def inspect_placement(mlpackage: Path, compute_units: ct.ComputeUnit) -> dict:
    model = ct.models.MLModel(str(mlpackage), compute_units=compute_units)
    compiled = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=compute_units)
    program = plan.model_structure.program
    if program is None:
        return {"error": "no ML program structure", "compiled": str(compiled)}
    main = program.functions["main"]
    counts: Counter[str] = Counter()
    compute_counts: Counter[str] = Counter()
    ops_by_type: dict[str, Counter] = {}
    n_ops = 0

    def walk(block):
        nonlocal n_ops
        for op in block.operations:
            n_ops += 1
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = _device_label(
                getattr(usage, "preferred_compute_device", None) if usage else None
            )
            counts[preferred] += 1
            name = op.operator_name
            if name not in SKIP_OPS:
                compute_counts[preferred] += 1
                ops_by_type.setdefault(name, Counter())[preferred] += 1
            for nested in op.blocks or []:
                walk(nested)

    walk(main.block)
    compute_n = sum(compute_counts.values())
    ane_frac = compute_counts.get("ANE", 0) / compute_n if compute_n else 0.0
    del model
    return {
        "compiled": str(compiled),
        "n_ops_including_const": n_ops,
        "counts_all": dict(counts),
        "counts_compute": dict(compute_counts),
        "ane_fraction_of_compute_ops": ane_frac,
        "ops_by_type": {k: dict(v) for k, v in sorted(ops_by_type.items())},
    }


def _summarize_power(rows: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r.get(key, 0.0) or 0.0) for r in rows]

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "min": min(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
        }

    return {
        "ane_power": stats(col("ane_power")),
        "gpu_power": stats(col("gpu_power")),
        "cpu_power": stats(col("cpu_power")),
        "all_power": stats(col("all_power")),
        "n_samples": len(rows),
        "ane_power_trace": [float(r.get("ane_power", 0.0) or 0.0) for r in rows],
    }


def _start_macmon() -> tuple[subprocess.Popen, list[dict], threading.Thread]:
    proc = subprocess.Popen(
        ["macmon", "pipe", "-s", "0", "-i", "1000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    rows: list[dict] = []

    def _read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return proc, rows, thread


def _stop_macmon(proc: subprocess.Popen, thread: threading.Thread) -> None:
    proc.send_signal(signal.SIGINT)
    thread.join(timeout=3)
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()


def _make_feeds(model: ct.models.MLModel, seq_len: int, hidden: int) -> dict:
    return {
        "hidden_states": np.zeros((1, seq_len, hidden), dtype=np.float16),
    }


def _capture_compile_stderr(
    mlpackage: Path, compute_units: ct.ComputeUnit
) -> tuple[float, str, object | None]:
    """Load model and return (compile_wall_s, stderr_capture, model_or_None)."""
    import io
    import contextlib

    buf = io.StringIO()
    model = None
    t0 = time.perf_counter()
    err = None
    try:
        with contextlib.redirect_stderr(buf):
            model = ct.models.MLModel(str(mlpackage), compute_units=compute_units)
            # Force compiled path materialization.
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
    # Prefer single-predict timings for stability at large seq; batch short ones.
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
    proc, rows, thread = _start_macmon()
    time.sleep(1.0)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        model.predict(feeds)
        n += 1
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    _stop_macmon(proc, thread)
    return n, elapsed, _summarize_power(list(rows))


def measure_one(
    mlpackage: Path,
    seq_len: int,
    hidden: int,
    precision: str,
    units_label: str,
    *,
    warmup: int,
    timed_runs: int,
    power_s: float,
    skip_placement: bool,
) -> dict:
    cu = UNITS[units_label]
    print(
        f"\n== measure seq={seq_len} {precision} units={units_label} ==",
        flush=True,
    )
    out: dict = {
        "seq_len": seq_len,
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
        if "ANE" in line or "ane" in line.lower() or "Neural" in line
    ]
    out["ane_compile_errors"] = ane_errs
    if model is None:
        out["error"] = "model load failed"
        return out
    print(
        f"  compile/load wall {compile_s:.2f}s; ane_err_lines={len(ane_errs)}",
        flush=True,
    )

    feeds = _make_feeds(model, seq_len, hidden)

    if not skip_placement:
        try:
            # Placement needs a fresh load path via the package; reuse walker.
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

    # Timing (best-of). Scale run count down for long seq.
    local_runs = timed_runs
    local_min_dur = 2.0
    if seq_len >= 1024:
        local_runs = max(5, timed_runs // 2)
        local_min_dur = 3.0
    if seq_len >= 2048:
        local_runs = 5
        local_min_dur = 4.0

    try:
        timing = time_predicts(
            model,
            feeds,
            warmup=warmup,
            timed_runs=local_runs,
            min_duration_s=local_min_dur,
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

    # Power window.
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
    """E1 PASS only if seq=1024 ANE ms/pred materially under 160 with ~100% ANE ops."""
    candidates = []
    for r in results:
        if r.get("seq_len") != THRESHOLD_SEQ:
            continue
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
            }
        )

    decision = {
        "threshold_seq": THRESHOLD_SEQ,
        "threshold_ms": THRESHOLD_MS,
        "rule": (
            "PASS only if at seq=1024, cpuAndNeuralEngine ms/pred is materially "
            "under 160 ms AND ANE op placement is at or near 100%."
        ),
        "candidates": candidates,
        "verdict": "FAIL",
        "reason": "",
    }
    if not candidates:
        decision["reason"] = "no seq=1024 cpuAndNeuralEngine timing result"
        return decision

    # Prefer the best (lowest) ms among precisions that have near-100% ANE placement.
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
            f"ANE placement {placed:.3f} is below ~100% at seq=1024 "
            f"(best ms/pred={ms:.2f})"
        )
        return decision
    if ms >= THRESHOLD_MS:
        decision["reason"] = (
            f"ANE ms/pred={ms:.2f} is NOT materially under {THRESHOLD_MS:.0f} ms "
            f"(placement={placed:.3f})"
        )
        return decision
    # "Materially under" — require at least 10% headroom to avoid noise.
    if ms >= THRESHOLD_MS * 0.9:
        decision["reason"] = (
            f"ANE ms/pred={ms:.2f} is under {THRESHOLD_MS:.0f} but not materially "
            f"(need < {THRESHOLD_MS * 0.9:.0f} for clear win); placement={placed:.3f}"
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
    parser.add_argument("--seqs", default=",".join(str(s) for s in SEQS_DEFAULT))
    parser.add_argument("--precisions", default="fp16,int8")
    parser.add_argument(
        "--units", default="cpuOnly,cpuAndNeuralEngine,all"
    )
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--power-s", type=float, default=8.0)
    parser.add_argument("--skip-placement", action="store_true")
    parser.add_argument(
        "--only-seq",
        type=int,
        default=None,
        help="if set, only measure this seq (still writes decision if 1024 present)",
    )
    args = parser.parse_args()
    seqs = [int(x) for x in args.seqs.split(",") if x.strip()]
    if args.only_seq is not None:
        seqs = [args.only_seq]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for seq in seqs:
        for precision in precisions:
            path = ARTIFACTS / f"attn27b_s{seq}_{precision}.mlpackage"
            if not path.exists():
                results.append(
                    {
                        "seq_len": seq,
                        "precision": precision,
                        "error": f"missing {path}",
                    }
                )
                continue
            for units_label in units:
                r = measure_one(
                    path,
                    seq,
                    args.hidden,
                    precision,
                    units_label,
                    warmup=args.warmup,
                    timed_runs=args.timed_runs,
                    power_s=args.power_s,
                    skip_placement=args.skip_placement,
                )
                results.append(r)
                # Persist incrementally so a long run is not lost.
                partial = {
                    "results": results,
                    "decision": decide(results),
                }
                (ARTIFACTS / "measure.json").write_text(
                    json.dumps(partial, indent=2, default=str)
                )

    decision = decide(results)
    report = {
        "threshold_seq": THRESHOLD_SEQ,
        "threshold_ms": THRESHOLD_MS,
        "results": results,
        "decision": decision,
    }
    out = ARTIFACTS / "measure.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\nDECISION:", decision["verdict"], "-", decision["reason"], flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
