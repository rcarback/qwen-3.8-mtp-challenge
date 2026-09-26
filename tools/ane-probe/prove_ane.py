#!/usr/bin/env python3
"""Prove (or refute) ANE execution for a Core ML mlpackage.

Three independent checks, reported separately:
  1. Power: macmon pipe ane_power / gpu_power / cpu_power during a long loop
  2. Placement: MLComputePlan per-op preferred device
  3. Behavioural A/B: CPU_ONLY vs CPU_AND_NE vs ALL timing and ane_power
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

SKIP_OPS = {"const", "constexpr_blockwise_shift_scale", "constexpr_affine_dequantize"}


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
    print(f"MLComputePlan {mlpackage.name} units={compute_units.name} ...", flush=True)
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
    ane_frac = (
        compute_counts.get("ANE", 0) / compute_n if compute_n else 0.0
    )
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
        "raw": [
            {
                "ane_power": r.get("ane_power"),
                "gpu_power": r.get("gpu_power"),
                "cpu_power": r.get("cpu_power"),
            }
            for r in rows
        ],
    }


def _make_inputs(model: ct.models.MLModel, input_ids: np.ndarray | None) -> dict:
    from coremltools.proto import FeatureTypes_pb2 as ft

    spec = model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        name = inp.name
        td = inp.type.multiArrayType
        shape = [int(s) for s in td.shape]
        dt = td.dataType
        if name == "input_ids" and input_ids is not None:
            arr = input_ids.astype(np.int32)
            if list(arr.shape) != shape:
                raise ValueError(f"input_ids shape {arr.shape} != {shape}")
            inputs[name] = arr
            continue
        if dt == ft.ArrayFeatureType.INT32:
            inputs[name] = np.zeros(shape, dtype=np.int32)
        elif dt == ft.ArrayFeatureType.FLOAT16:
            inputs[name] = np.zeros(shape, dtype=np.float16)
        else:
            inputs[name] = np.zeros(shape, dtype=np.float32)
    return inputs


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


def _sample_during(model, feeds, duration_s: int) -> tuple[int, float, list[dict]]:
    proc, rows, thread = _start_macmon()
    time.sleep(1.2)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        model.predict(feeds)
        n += 1
    elapsed = time.perf_counter() - t0
    time.sleep(0.6)
    _stop_macmon(proc, thread)
    return n, elapsed, list(rows)


def idle_baseline(seconds: int = 5) -> dict:
    print(f"idle macmon baseline {seconds}s ...", flush=True)
    proc, rows, thread = _start_macmon()
    time.sleep(seconds)
    _stop_macmon(proc, thread)
    return _summarize_power(list(rows))


def run_ab(mlpackage: Path, duration_s: int, input_ids: np.ndarray | None) -> dict:
    units = {
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    out: dict = {}
    for label, cu in units.items():
        print(f"A/B load {label} ...", flush=True)
        model = ct.models.MLModel(str(mlpackage), compute_units=cu)
        feeds = _make_inputs(model, input_ids)
        print("  warmup 2 predicts", flush=True)
        for _ in range(2):
            model.predict(feeds)
        print(f"  timed loop ~{duration_s}s with macmon", flush=True)
        n, elapsed, rows = _sample_during(model, feeds, duration_s)
        out[label] = {
            "loops": n,
            "elapsed_s": elapsed,
            "ms_per_predict": (elapsed / n) * 1000.0 if n else None,
            "power": _summarize_power(rows),
        }
        ane = out[label]["power"]["ane_power"]
        gpu = out[label]["power"]["gpu_power"]
        cpu = out[label]["power"]["cpu_power"]
        print(
            f"  {label}: {out[label]['ms_per_predict']:.1f} ms/pred n={n} "
            f"ane max/mean={ane.get('max')}/{ane.get('mean')} "
            f"gpu mean={gpu.get('mean')} cpu mean={cpu.get('mean')}",
            flush=True,
        )
        del model
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument("--skip-placement", action="store_true")
    args = parser.parse_args()
    path = Path(args.mlpackage)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    from qwen35_torch import tokenize_fixed_prompt

    ids = tokenize_fixed_prompt().numpy()
    report = {
        "mlpackage": str(path),
        "exists": path.exists(),
        "input_ids": ids.tolist(),
        "idle": idle_baseline(4),
        "placement": {},
        "ab": None,
    }
    if not args.skip_placement:
        for name, cu in (
            ("cpuAndNeuralEngine", ct.ComputeUnit.CPU_AND_NE),
            ("all", ct.ComputeUnit.ALL),
            ("cpuOnly", ct.ComputeUnit.CPU_ONLY),
        ):
            try:
                report["placement"][name] = inspect_placement(path, cu)
                print("  placement", name, report["placement"][name]["counts_compute"])
            except Exception as exc:
                report["placement"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                print("  placement failed", name, exc)
    if not args.skip_ab:
        try:
            report["ab"] = run_ab(path, args.duration, ids)
        except Exception as exc:
            report["ab"] = {"error": f"{type(exc).__name__}: {exc}"}
            print("A/B failed", exc)
    out_path = ARTIFACTS / f"ane_proof_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
