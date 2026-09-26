#!/usr/bin/env python3
"""Load a converted Qwen3.5-0.8B Core ML package and run one static-window forward."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import coremltools as ct

from qwen35_torch import SEQ_LEN, tokenize_fixed_prompt

UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlpackage",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "qwen35_08b_text_fp16.mlpackage"
        ),
    )
    parser.add_argument("--units", choices=list(UNITS), default="cpuAndNeuralEngine")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    ids = tokenize_fixed_prompt(SEQ_LEN).numpy().astype(np.int32)
    print("seq_len", SEQ_LEN)
    print("input_ids", ids.tolist())
    print("loading", args.mlpackage, "units", args.units)
    t0 = time.perf_counter()
    model = ct.models.MLModel(args.mlpackage, compute_units=UNITS[args.units])
    print(f"loaded in {time.perf_counter() - t0:.2f}s")
    feeds = {"input_ids": ids}
    times = []
    out = None
    for i in range(args.repeat):
        t1 = time.perf_counter()
        out = model.predict(feeds)
        times.append(time.perf_counter() - t1)
        print(f"predict[{i}] {times[-1]*1000:.1f} ms")
    key = list(out.keys())[0]
    logits = np.array(out[key])
    print("output", key, logits.shape, logits.dtype)
    print("finite", bool(np.isfinite(logits).all()), "nan", bool(np.isnan(logits).any()))
    last = logits.reshape(-1, logits.shape[-1])[-1]
    argmax = int(np.argmax(last))
    print("last_argmax", argmax, "last_min", float(last.min()), "last_max", float(last.max()))
    if times:
        print("mean_ms", float(np.mean(times) * 1000.0))


if __name__ == "__main__":
    main()
