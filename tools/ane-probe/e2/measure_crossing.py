#!/usr/bin/env python3
"""M1: crossing cost. Python binding timings plus native Swift Core ML timings."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import coremltools as ct

from common import ARTIFACTS, SEQ, UNITS, inspect_placement

SHAPE = (1, SEQ, 5120)
NBYTES = 1 * SEQ * 5120 * 2


def _stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "n": n,
        "best_ms": s[0],
        "median_ms": s[n // 2],
        "mean_ms": sum(s) / n,
        "p90_ms": s[int(0.9 * (n - 1))],
        "worst_ms": s[-1],
        "samples_ms": samples_ms,
    }


def time_loop(n: int, warmup: int, fn) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


def python_path(mlpackage: Path) -> dict:
    print("== python coremltools.predict path ==", flush=True)
    cu = UNITS["cpuAndNeuralEngine"]
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(mlpackage), compute_units=cu)
    load_s = time.perf_counter() - t0
    x = np.random.default_rng(42).standard_normal(SHAPE).astype(np.float16)
    feeds = {"hidden_states": x}

    t0 = time.perf_counter()
    out0 = model.predict(feeds)
    first_ms = (time.perf_counter() - t0) * 1000.0
    out_key = "output_hidden" if "output_hidden" in out0 else list(out0.keys())[0]
    print(f"  load {load_s:.3f}s first_predict {first_ms:.3f} ms", flush=True)

    def construct_numpy():
        return np.array(x, copy=True)

    def predict_reused():
        return model.predict(feeds)

    extract_holder = {"arr": out0[out_key]}

    def extract():
        return np.array(extract_holder["arr"], copy=True)

    def round_trip():
        local = np.array(x, copy=True)
        out = model.predict({"hidden_states": local})
        return np.array(out[out_key], copy=True)

    warmup, timed = 8, 80
    construction = time_loop(timed, warmup, construct_numpy)
    predict_s = time_loop(timed, warmup, predict_reused)
    # Refresh extract source from a real output.
    extract_holder["arr"] = model.predict(feeds)[out_key]
    extract_s = time_loop(timed, warmup, extract)
    round_s = time_loop(timed, warmup, round_trip)

    print(
        f"  numpy copy construct median={construction['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"  predict(reused numpy) best={predict_s['best_ms']:.3f} "
        f"median={predict_s['median_ms']:.3f} ms n={predict_s['n']}",
        flush=True,
    )
    print(
        f"  output np.array copy median={extract_s['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"  python round-trip best={round_s['best_ms']:.3f} "
        f"median={round_s['median_ms']:.3f} ms",
        flush=True,
    )
    del model
    return {
        "binding": "coremltools.MLModel.predict",
        "note": (
            "Python predict always copies numpy into an MLMultiArray internally. "
            "This is NOT the zero-copy native API. See m1_crossing.swift for "
            "initWithDataPointer / IOSurface."
        ),
        "load_s": load_s,
        "first_predict_ms": first_ms,
        "numpy_copy_construct": construction,
        "predict_reused_numpy": predict_s,
        "output_numpy_copy": extract_s,
        "round_trip": round_s,
    }


def compile_and_run_swift(mlpackage: Path) -> dict:
    src = Path(__file__).resolve().parent / "m1_crossing.swift"
    bin_path = ARTIFACTS / "m1_crossing"
    out_json = ARTIFACTS / "m1_crossing_swift.json"
    print("== compile m1_crossing.swift ==", flush=True)
    cmd = [
        "swiftc",
        "-O",
        "-framework",
        "CoreML",
        "-framework",
        "Foundation",
        "-framework",
        "CoreVideo",
        str(src),
        "-o",
        str(bin_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    compile_s = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return {
            "error": "swiftc failed",
            "compile_s": compile_s,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    print(f"  swiftc ok in {compile_s:.1f}s", flush=True)
    print("== run native Core ML crossing ==", flush=True)
    run = subprocess.run(
        [str(bin_path), str(mlpackage), str(out_json)],
        capture_output=True,
        text=True,
    )
    print(run.stdout, flush=True)
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        return {
            "error": "m1_crossing binary failed",
            "returncode": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
        }
    payload = json.loads(out_json.read_text())
    payload["swiftc_s"] = compile_s
    payload["stdout"] = run.stdout
    return payload


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlpackage = ARTIFACTS / "crossing_s1024_fp16.mlpackage"
    if not mlpackage.exists():
        raise SystemExit(f"missing {mlpackage}; run convert_crossing.py first")

    report: dict = {
        "shape": list(SHAPE),
        "nbytes": NBYTES,
        "mlpackage": str(mlpackage),
        "threshold": {
            "pass_copy_ms": 5.0,
            "pass_zerocopy_ms": 2.0,
            "fail_ms": 38.0,
            "rule": (
                "PASS if per-crossing cost is under 5 ms with a copy, or under "
                "2 ms zero-copy. FAIL if above 38 ms."
            ),
        },
    }

    try:
        placement = inspect_placement(mlpackage, UNITS["cpuAndNeuralEngine"])
        report["placement_cpuAndNeuralEngine"] = placement
        print(
            f"placement compute={placement.get('counts_compute')} "
            f"ane_frac={placement.get('ane_fraction_of_compute_ops')}",
            flush=True,
        )
    except Exception as exc:
        report["placement_cpuAndNeuralEngine"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("placement failed:", exc, flush=True)

    report["python"] = python_path(mlpackage)
    report["swift"] = compile_and_run_swift(mlpackage)

    out = ARTIFACTS / "m1_crossing.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
