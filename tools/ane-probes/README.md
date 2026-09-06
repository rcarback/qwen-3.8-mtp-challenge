# ANE probe and bank generators

Python (coremltools 9, Python 3.12) generators for the ANE probes and the
dense tower's Core ML fused bank. They are offline tools for the local M4
fork; nothing here is part of the ranked runtime.

| script | output | consumed by |
| --- | --- | --- |
| `gen_probe_pkgs.py OUT` | one `.mlpackage` per weight form plus a misc-op program, with numpy references | `ANEComputePlanProbeTests` |
| `gen_r_pkgs.py OUT [shape]` | real projection shapes at several sequence lengths, fp16 and int4, plus a spatial-layout sweep | `ANESplitRatioProbeTests` |
| `gen_layer_probe.py OUT` | one whole gated-delta layer at S=1, int4 palettes, timing only | `ANEComputePlanProbeTests` |
| `gen_fused_bank.py WEIGHTS OUT FRACTION BUCKET [LAYERS]` | `S<BUCKET>.mlpackage`, all layers as functions, per-row int4 codebooks | `ANEFusedMLPBank` (`MLX_ANE_BANK_DIR`) |

Run with a Python 3.12 virtual environment holding `coremltools==9.0` and
`numpy`; coremltools on Python 3.14 lacks a working blob writer.
