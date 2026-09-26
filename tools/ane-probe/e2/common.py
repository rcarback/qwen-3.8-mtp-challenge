#!/usr/bin/env python3
"""Shared geometry, placement, and power helpers for E2 measurements."""

from __future__ import annotations

import json
import signal
import subprocess
import threading
from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

CONFIG_PATH = Path(
    "/Users/carback1/Code/llms/qwen-3.8-mtp-challenge/weights/config.json"
)
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
E1_ARTIFACTS = HERE.parent / "e1" / "artifacts"
SKIP_OPS = {"const", "constexpr_blockwise_shift_scale", "constexpr_affine_dequantize"}
UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}
SEQ = 1024
SEED = 42
RMS_EPS = 1e-6


def load_geometry() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    hidden = int(cfg["hidden_size"])
    n_k = int(cfg["linear_num_key_heads"])
    n_v = int(cfg["linear_num_value_heads"])
    head_k = int(cfg["linear_key_head_dim"])
    head_v = int(cfg["linear_value_head_dim"])
    key_dim = n_k * head_k
    value_dim = n_v * head_v
    return {
        "hidden_size": hidden,
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(cfg["head_dim"]),
        "linear_num_key_heads": n_k,
        "linear_num_value_heads": n_v,
        "linear_key_head_dim": head_k,
        "linear_value_head_dim": head_v,
        "linear_conv_kernel_dim": int(cfg["linear_conv_kernel_dim"]),
        "key_dim": key_dim,
        "value_dim": value_dim,
        "conv_dim": key_dim * 2 + value_dim,
        "rms_norm_eps": float(cfg.get("rms_norm_eps", RMS_EPS)),
        "model_type": str(cfg["model_type"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "layer_types": list(cfg["layer_types"]),
        "hidden_act": str(cfg["hidden_act"]),
    }


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
        del model
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


def summarize_power(rows: list[dict]) -> dict:
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


def start_macmon() -> tuple[subprocess.Popen, list[dict], threading.Thread]:
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


def stop_macmon(proc: subprocess.Popen, thread: threading.Thread) -> None:
    proc.send_signal(signal.SIGINT)
    thread.join(timeout=3)
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()
