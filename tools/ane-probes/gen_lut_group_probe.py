"""Grouped-codebook placement probe at the dense tower's real MLP shapes.

A single conv per package, iOS18 `constexpr_lut_to_dense` weights, one
16-entry codebook per block of ROWS output rows, over a sweep of ROWS. The
2026-09-06 finding this probes: a 64-row codebook stays on the ANE at the
[1280 x 2560] probe shape, and the same declaration at the fused bank's
real shape ([5440 x 5120] gate/up, [5120 x 5440] down) is placed on the
GPU by Core ML with the conv reported as `supported=cpu/gpu`. The sweep
finds the rows-per-codebook boundary at each shape. Sidecars match
`gen_probe_pkgs.py` so `ANEComputePlanProbeTests` reads them unchanged.

usage: gen_lut_group_probe.py OUT
"""
import json
import os
import sys

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as T

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
S = 256
rng = np.random.default_rng(1)

SHAPES = {"gate": (5440, 5120), "down": (5120, 5440), "probe": (1280, 2560)}
ROWS = {"gate": [None, 2720, 1360, 680, 320, 160, 64], "down": [None, 2560, 1280, 640, 320, 64], "probe": [64]}


def lut_uniform(w, per):
    """Per-tensor (per=None) or per-block-of-rows uniform 16-level palette."""
    Oo, Kk = w.shape
    blocks = [(0, Oo)] if per is None else [(i, min(i + per, Oo)) for i in range(0, Oo, per)]
    codes = np.zeros((Oo, Kk), dtype=np.uint8)
    luts = []
    w_eff = np.zeros_like(w)
    for (a, b) in blocks:
        s = np.abs(w[a:b]).max() / 8.0
        lut = np.array([(i - 8) * s for i in range(16)], dtype=np.float16)
        c = np.clip(np.round(w[a:b] / s) + 8, 0, 15).astype(np.uint8)
        codes[a:b] = c
        luts.append(lut)
        w_eff[a:b] = lut.astype(np.float32)[c]
    return codes, np.stack(luts), w_eff


def conv_prog(O, K, codes, luts, groups):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, K, 1, S), dtype=T.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        w = mb.constexpr_lut_to_dense(indices=codes.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=luts.reshape(groups, 1, 1, 1, 16, 1))
        return mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1, name="conv_0")
    return prog


manifest = {}
for shape_name, (O, K) in SHAPES.items():
    w = (rng.standard_normal((O, K)) * 0.02).astype(np.float32)
    x = rng.standard_normal((S, K)).astype(np.float16)
    x32 = x.astype(np.float32)
    for per in ROWS[shape_name]:
        groups = 1 if per is None else O // per
        name = "%s_%dx%d_lut_%s" % (shape_name, O, K, "pertensor" if per is None else "per%drows_g%d" % (per, groups))
        codes, luts, w_eff = lut_uniform(w, per)
        try:
            m = ct.convert(conv_prog(O, K, codes, luts, groups), minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
            pkg = os.path.join(OUT, name + ".mlpackage")
            m.save(pkg)
            side = os.path.join(OUT, name + ".probe")
            os.makedirs(side, exist_ok=True)
            w_eff.astype(np.float16).tofile(os.path.join(side, "w_eff_f16.bin"))
            x.tofile(os.path.join(side, "x_f16.bin"))
            (x32 @ w_eff.T).astype(np.float32).tofile(os.path.join(side, "y_ref_f32.bin"))
            json.dump({"O": O, "K": K, "S": S, "input": "x", "output": m.get_spec().description.output[0].name}, open(os.path.join(side, "meta.json"), "w"))
            manifest[name] = "ok"
            print("built", name)
        except Exception as e:  # noqa: BLE001 - record the converter's refusal beside the packages that built
            manifest[name] = "convert failed: " + str(e)[:300]
            print("FAILED", name, str(e)[:300])
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
