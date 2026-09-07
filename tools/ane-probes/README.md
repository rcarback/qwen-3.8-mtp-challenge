# ANE probe and bank generators

Python (coremltools 9, Python 3.12) generators for the ANE probes and the
dense tower's Core ML fused bank. They are offline tools for the local M4
fork; nothing here is part of the ranked runtime.

| script | output | consumed by |
| --- | --- | --- |
| `gen_probe_pkgs.py OUT` | one `.mlpackage` per weight form plus a misc-op program, with numpy references | `ANEComputePlanProbeTests` |
| `gen_r_pkgs.py OUT [shape]` | real projection shapes at several sequence lengths, fp16 and int4, plus a spatial-layout sweep | `ANESplitRatioProbeTests` |
| `gen_layer_probe.py OUT` | one whole gated-delta layer at S=1, int4 palettes, timing only | `ANEComputePlanProbeTests` |
| `gen_lut_group_probe.py OUT` | one conv per package at the dense MLP's real shapes, a rows-per-codebook sweep of the grouped int4 palette | `ANEComputePlanProbeTests` |
| `gen_fused_bank.py WEIGHTS OUT FRACTION BUCKET [LAYERS]` | `S<BUCKET>.mlpackage`, all layers as functions, per-row int4 codebooks | `ANEFusedMLPBank` (`MLX_ANE_BANK_DIR`) |
| `serve-sweep.sh OUT.tsv ARM DEPTH [ENV=VAL ...]` | one TSV row per prompt (prefill and decode tok/s, draft counts, request wall time), the raw response and the completion text | the end-to-end arms in `docs/perf/ane-gpu-reevaluation-2026-09-06.md` |
| `agree.py DIR CONTROL ARM...` | greedy-completion agreement of each arm against the control | the "identical completions" column |
| `gates.sh` (source it) | `cool_gate`, `quiet_gate`, `ensure_metallib` for timed arms | every queue that runs a timed arm |
| `cleanup-run.sh [PATH ...]` | removes what a run leaves behind (compiled models, staging directories, the ANE pipeline cache) | after every run |

Run with a Python 3.12 virtual environment holding `coremltools==9.0` and
`numpy`; coremltools on Python 3.14 lacks a working blob writer.
