#!/usr/bin/env python3
"""Time / power the same op classes on Python MLX (Metal).

This is the closest cheap stand-in for the mlx-swift path. It is NOT the
vendored `Vendor/mlx-swift` metallib. GEMM cells use affine 4-bit group-64
`quantized_matmul` (the algorithm `Qwen35Ops.linear` dispatches). Score and
elementwise cells use fp16 `matmul` / `rms_norm` / `silu` because those ops
are not 4-bit in the shipped model.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from catalog import (
    ARTIFACTS,
    SEED,
    build_cells,
    engine_watts,
    load_geometry,
    perf_per_watt,
    start_macmon,
    stop_macmon,
    summarize_power,
    tflops,
)
from convert import build_module

GEO = load_geometry()
GROUP = 64
BITS = 4


def _torch_weight(mod, name: str) -> mx.array:
    p = dict(mod.named_parameters())[name]
    return mx.array(p.detach().cpu().numpy())


def _quantize_linear(weight: mx.array):
    # weight is [out, in], matching nn.Linear and Qwen35LinearWeight.
    packed, scales, biases = mx.quantize(
        weight.astype(mx.float16), group_size=GROUP, bits=BITS, mode="affine"
    )
    return packed, scales, biases


def _qmm(x: mx.array, packed, scales, biases) -> mx.array:
    return mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=GROUP,
        bits=BITS,
    )


class MetalOp:
    def __init__(self, cell, label: str, run, feeds, flops: int):
        self.cell = cell
        self.label = label
        self._run = run
        self.feeds = feeds
        self.flops = flops

    def __call__(self):
        return self._run()


def build_metal_op(cell) -> MetalOp:
    geo = GEO
    rng = np.random.default_rng(SEED + cell.seq + cell.kv)

    def rand(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float16))

    if cell.op in ("large_gemm_up", "large_gemm_down", "mtp_fc", "mtp_mlp_up"):
        mod, _, _, _ = build_module(cell)
        w = _torch_weight(mod, "proj.weight")
        packed, scales, biases = _quantize_linear(w)
        x = rand(cell.seq, cell.in_features)

        def run():
            return _qmm(x, packed, scales, biases)

        return MetalOp(
            cell,
            "python-mlx-0.32.2 quantized_matmul affine-4-group-64 (NOT vendored mlx-swift)",
            run,
            {"x": x},
            cell.flops,
        )

    if cell.op == "attn_proj":
        mod, _, _, _ = build_module(cell)
        wq = _quantize_linear(_torch_weight(mod, "q.weight"))
        wk = _quantize_linear(_torch_weight(mod, "k.weight"))
        wv = _quantize_linear(_torch_weight(mod, "v.weight"))
        wo = _quantize_linear(_torch_weight(mod, "o.weight"))
        x = rand(cell.seq, geo["hidden_size"])
        attn_out = rand(cell.seq, geo["o_in"])

        def run():
            q = _qmm(x, *wq)
            k = _qmm(x, *wk)
            v = _qmm(x, *wv)
            o = _qmm(attn_out, *wo)
            return q, k, v, o

        return MetalOp(
            cell,
            "python-mlx-0.32.2 4x quantized_matmul affine-4-group-64 (NOT vendored mlx-swift)",
            run,
            {"x": x, "attn_out": attn_out},
            cell.flops,
        )

    if cell.op == "attn_scores":
        n_q = geo["num_attention_heads"]
        n_kv = geo["num_key_value_heads"]
        d = geo["head_dim"]
        n_rep = n_q // n_kv
        q = rand(n_q, cell.seq, d)
        k = rand(n_kv, cell.kv, d)
        v = rand(n_kv, cell.kv, d)
        scale = d**-0.5

        def run():
            k_exp = mx.repeat(k, n_rep, axis=0)
            v_exp = mx.repeat(v, n_rep, axis=0)
            scores = (q @ mx.swapaxes(k_exp, -1, -2)) * scale
            return scores @ v_exp

        return MetalOp(
            cell,
            "python-mlx-0.32.2 fp16 matmul QK^T+AV (KV is not 4-bit in the shipped model)",
            run,
            {"q": q, "k": k, "v": v},
            cell.flops,
        )

    if cell.op == "elem_swiglu":
        mod, _, _, _ = build_module(cell)
        w = _torch_weight(mod, "weight").astype(mx.float16)
        gate = rand(cell.seq, cell.width)
        up = rand(cell.seq, cell.width)
        eps = float(geo["rms_norm_eps"])

        def run():
            x = gate.astype(mx.float32)
            var = mx.mean(x * x, axis=-1, keepdims=True)
            x = x * mx.rsqrt(var + eps)
            x = (x * w.astype(mx.float32)).astype(mx.float16)
            return (x * mx.sigmoid(x)) * up

        return MetalOp(
            cell,
            "python-mlx-0.32.2 fp16 RMSNorm+SiLU+mul (elementwise; not quantized)",
            run,
            {"gate": gate, "up": up},
            cell.flops,
        )

    raise ValueError(cell.op)


def _eval(y) -> None:
    if isinstance(y, tuple):
        mx.eval(*y)
    else:
        mx.eval(y)


def time_op(op: MetalOp, *, warmup: int, timed_runs: int, min_duration_s: float):
    for _ in range(warmup):
        _eval(op())
    mx.synchronize()
    samples: list[float] = []
    t_loop0 = time.perf_counter()
    while len(samples) < timed_runs or (time.perf_counter() - t_loop0) < min_duration_s:
        mx.synchronize()
        t0 = time.perf_counter()
        _eval(op())
        mx.synchronize()
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


def power_during(op: MetalOp, duration_s: float):
    proc, rows, thread = start_macmon()
    time.sleep(1.0)
    n = 0
    t0 = time.perf_counter()
    deadline = t0 + duration_s
    while time.perf_counter() < deadline:
        _eval(op())
        n += 1
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    stop_macmon(proc, thread)
    return n, elapsed, summarize_power(list(rows))


def metal_correctness(cell, op: MetalOp) -> dict:
    """Compare 4-bit mlx against the dequantized reference on the same weights."""
    if cell.op not in ("large_gemm_up", "large_gemm_down", "mtp_fc", "mtp_mlp_up"):
        return {"skipped": "graph has multiple outputs or is not a single GEMM"}
    mod, _, _, _ = build_module(cell)
    w = _torch_weight(mod, "proj.weight").astype(mx.float16)
    packed, scales, biases = _quantize_linear(w)
    deq = mx.dequantize(packed, scales, biases, group_size=GROUP, bits=BITS)
    x = op.feeds["x"]
    y_q = _qmm(x, packed, scales, biases)
    y_ref = x @ deq.T
    mx.eval(y_q, y_ref)
    a = np.array(y_q, dtype=np.float32).ravel()
    b = np.array(y_ref, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom else None
    diff = a - b
    return {
        "cosine_vs_dequant": cos,
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pass": bool(cos is not None and cos >= 0.99),
        "note": "4-bit vs dequant of THE SAME packed weights, not vs fp16 PyTorch",
    }


def measure_one(cell, *, warmup: int, timed_runs: int, power_s: float) -> dict:
    print(f"\n== metal {cell.tag} ==", flush=True)
    op = build_metal_op(cell)
    out: dict = {
        "tag": cell.tag,
        "op": cell.op,
        "seq": cell.seq,
        "kv": cell.kv,
        "flops": cell.flops,
        "gflop": cell.gflop,
        "precision": "affine-4-group-64"
        if cell.op
        not in (
            "attn_scores",
            "elem_swiglu",
        )
        else "fp16",
        "backend": "metal",
        "kind": "MEASURED",
        "kernel_label": op.label,
        "metal_caveat": (
            "Python mlx 0.32.2 Metal kernels, same 4-bit affine group-64 "
            "algorithm as Qwen35Ops.quantizedMM. Not the vendored mlx-swift "
            "metallib in this repo."
        ),
    }
    try:
        corr = metal_correctness(cell, op)
        out["correctness"] = corr
        print(f"  correctness {corr}", flush=True)
    except Exception as exc:
        out["correctness"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        timing = time_op(
            op, warmup=warmup, timed_runs=timed_runs, min_duration_s=2.0
        )
        out["timing"] = timing
        out["ms"] = timing["ms_median"]
        out["tflops"] = tflops(cell.flops, out["ms"])
        print(
            f"  timing best={timing['ms_best']:.2f} med={out['ms']:.2f} "
            f"tflops={out['tflops']}",
            flush=True,
        )
    except Exception as exc:
        out["timing"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  timing failed:", exc, flush=True)

    try:
        n, elapsed, power = power_during(op, power_s)
        out["power"] = {
            "loops": n,
            "elapsed_s": elapsed,
            "ms_per_predict_in_window": (elapsed / n) * 1000.0 if n else None,
            **power,
        }
        watts = engine_watts(power, "metal")
        out["engine_watts"] = watts
        out["tflops_per_watt"] = (
            perf_per_watt(cell.flops, out["ms"], watts)
            if out.get("ms") and watts
            else None
        )
        print(
            f"  power gpu={power['gpu_power'].get('mean')} "
            f"ane={power['ane_power'].get('mean')} "
            f"engine_W={watts} tflops/W={out['tflops_per_watt']}",
            flush=True,
        )
    except Exception as exc:
        out["power"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("  power failed:", exc, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--power-s", type=float, default=6.0)
    parser.add_argument("--out", default="measure_metal.json")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ops = {x.strip() for x in args.ops.split(",") if x.strip()}
    seqs = {int(x) for x in args.seqs.split(",") if x.strip()}
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
    done = {r.get("tag") for r in results if not r.get("error")}

    print(
        f"mlx default device={mx.default_device()} "
        f"metal={mx.metal.is_available()}",
        flush=True,
    )
    for cell in cells:
        if cell.tag in done:
            print(f"skip existing {cell.tag}", flush=True)
            continue
        r = measure_one(
            cell,
            warmup=args.warmup,
            timed_runs=args.timed_runs,
            power_s=args.power_s,
        )
        results.append(r)
        payload = {
            "machine": "Apple M4 Max",
            "mlx_version": "0.32.2",
            "geo": GEO,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
