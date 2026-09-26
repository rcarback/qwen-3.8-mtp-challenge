#!/usr/bin/env python3
"""Convert the trivial crossing graph to a Core ML fp16 mlpackage."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

from common import ARTIFACTS, SEQ, load_geometry
from crossing_model import build_crossing


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    net, geo = build_crossing()
    hidden = geo["hidden_size"]
    example = torch.randn(1, SEQ, hidden)
    print(f"tracing crossing net seq={SEQ} hidden={hidden} ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(net, example, strict=False)
        _ = traced(example)
    print(f"  traced in {time.time() - t0:.1f}s", flush=True)

    meta: dict = {"seq": SEQ, "hidden": hidden, "errors": {}}
    try:
        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="hidden_states",
                    shape=(1, SEQ, hidden),
                    dtype=np.float16,
                )
            ],
            outputs=[ct.TensorType(name="output_hidden", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS18,
        )
        convert_s = time.time() - t0
        print(f"  converted in {convert_s:.1f}s", flush=True)
        path = ARTIFACTS / "crossing_s1024_fp16.mlpackage"
        if path.exists():
            shutil.rmtree(path)
        mlmodel.save(str(path))
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        print(f"  saved {path.name} ({size / 1e6:.1f} MB)", flush=True)
        meta["fp16"] = {
            "path": str(path),
            "convert_s": convert_s,
            "bytes": size,
        }
    except Exception as exc:
        traceback.print_exc()
        meta["errors"]["fp16"] = f"{type(exc).__name__}: {exc}"

    (ARTIFACTS / "convert_crossing_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", ARTIFACTS / "convert_crossing_meta.json", flush=True)


if __name__ == "__main__":
    main()
