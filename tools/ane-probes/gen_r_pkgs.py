"""Packages for the r re-measurement: real projection shapes of both towers at
several sequence lengths, in fp16 and int4-palette (per-channel scale via a
runtime mul) forms, plus a spatial-layout sweep at one shape. Each package
has a sidecar with the effective fp16 weight, the fp16 input and an fp32
reference; the Swift harness supplies the GPU arm at production quantization.
Layout note: the conv is position-wise, so [1,K,H,W] with H*W=S computes the
same projection as [1,K,1,S]; the sidecar records (H,W) and the Swift side
fills the input surface in [K,H,W] order (row s = h*W + w)."""
import json, os, sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as T

OUT = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(7)

SHAPES = {   # name: (O, K, gpu_bits, gpu_group)  -- the GPU arm's production form
    "moe_expert_gateup": (1280, 2560, 4, 32),
    "moe_expert_down":   (2560, 640, 4, 32),
    "moe_inproj_qkv":    (10240, 2560, 8, 32),
    "dense_mlp_gate_half": (8704, 5120, 4, 64),
    "dense_mlp_down_half": (5120, 8704, 4, 64),
}
SEQS = [16, 128, 256, 512, 1024]

def lut_pcs_fp16(w):
    """Per-channel scale s[o] and a shared fp16 16-level uniform codebook; the
    program applies the scale with a runtime mul after the LUT dequant."""
    s = (np.abs(w).max(1, keepdims=True) / 8.0).astype(np.float16)
    c = np.clip(np.round(w / s.astype(np.float32)) + 8, 0, 15).astype(np.uint8)
    lut = np.array([(i - 8) for i in range(16)], dtype=np.float16)
    w_eff = s.astype(np.float32) * lut.astype(np.float32)[c]
    return c, lut, s, w_eff

def build(name, O, K, H, W, form, w, x):
    S = H * W
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, K, H, W), dtype=T.fp16)], opset_version=ct.target.iOS18)
    def prog(x):
        if form == "fp16":
            wt = mb.const(val=w.astype(np.float16).reshape(O, K, 1, 1))
        else:
            # The per-channel scale goes on the conv OUTPUT. A runtime mul on the
            # constexpr weight makes Core ML rebuild the dense weight every call
            # (0.1 to 3.8 s per conv measured 2026-09-06).
            c, lut, s, _ = lut_pcs_fp16(w)
            wt = mb.constexpr_lut_to_dense(indices=c.astype(T.np_uint4_dtype).reshape(O, K, 1, 1), lut=lut.reshape(1, 1, 1, 1, 16, 1))
            y = mb.conv(x=x, weight=wt, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1)
            return mb.mul(x=y, y=s.reshape(1, O, 1, 1))
        return mb.conv(x=x, weight=wt, strides=[1, 1], pad_type="valid", dilations=[1, 1], groups=1)
    w_eff = w.astype(np.float16).astype(np.float32) if form == "fp16" else lut_pcs_fp16(w)[3]
    m = ct.convert(prog, minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    pkg = os.path.join(OUT, name + ".mlpackage"); m.save(pkg)
    side = os.path.join(OUT, name + ".probe"); os.makedirs(side, exist_ok=True)
    w_eff.astype(np.float16).tofile(os.path.join(side, "w_eff_f16.bin"))
    x.astype(np.float16).tofile(os.path.join(side, "x_f16.bin"))
    (x.astype(np.float32) @ w_eff.T).astype(np.float32).tofile(os.path.join(side, "y_ref_f32.bin"))
    json.dump({"O": O, "K": K, "S": S, "H": H, "W": W, "form": form, "input": "x"}, open(os.path.join(side, "meta.json"), "w"))
    print("built", name, flush=True)

for sname, (O, K, gb, gg) in SHAPES.items():
    if only and only not in sname: continue
    w = (rng.standard_normal((O, K)) * 0.02).astype(np.float32)
    # Save the fp32 weight once per shape so the Swift GPU arm quantizes the SAME tensor.
    w.astype(np.float16).tofile(os.path.join(OUT, sname + ".w_f16.bin"))
    json.dump({"O": O, "K": K, "gpu_bits": gb, "gpu_group": gg}, open(os.path.join(OUT, sname + ".shape.json"), "w"))
    for S in SEQS:
        x = rng.standard_normal((S, K)).astype(np.float16)
        for form in ("fp16", "int4lut"):
            try: build(f"{sname}__S{S}__{form}", O, K, 1, S, form, w, x)
            except Exception as e: print("FAILED", sname, S, form, str(e)[:200], flush=True)
    if sname == "moe_inproj_qkv":
        x = rng.standard_normal((512, K)).astype(np.float16)
        for (H, W) in [(2, 256), (4, 128), (16, 32), (64, 8)]:
            try: build(f"{sname}__S512__fp16__H{H}xW{W}", O, K, H, W, "fp16", w, x)
            except Exception as e: print("FAILED layout", H, W, str(e)[:200], flush=True)
