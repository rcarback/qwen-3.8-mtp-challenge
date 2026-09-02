# ANE hybrid generation collapse, diagnosis and fix (2026-09-02)

The Apple Neural Engine (ANE) and GPU prefill hybrid (`MLX_ANE_DIRECT=1`) collapsed into one repeated
token on every real prompt. Two defects caused it. One was the whole
collapse. The other was a real but smaller numerics error.

1. **Descriptor identity collision.** The loader created every layer's ANE
   program from identical MIL text with an empty weights dictionary. The ANE runtime
   derives a program's identity from that text alone, so all 64 layers shared
   one staging directory and one compiled program. Only the first loaded
   layer computed with its own weights. Fix: stamp a unique
   `programTag` into each program's `buildInfo`.
2. **`silu` approximation.** The ANE lowers the MIL `silu` and `sigmoid` ops
   through a lookup table. Its error on real activations is 10 to 30 times the
   fp16 floor. Fix: spell the activation as `x / (1 + exp(-x))`, the new
   default `ANEActivation.expDiv`, selectable with `MLX_ANE_ACTIVATION`.

After both fixes the hybrid generates prose on all four prompts. All
measurements ran on an Apple M4 Max with 128 GB. The probes used the first
512 tokens of `README.md` as the prompt.

## Why earlier tests missed it

The synthetic single-program tests passed at 1 ULP because they built, used,
and unloaded one program at a time. A program that is alone never collides.
The self-consistent goldens passed because a deterministic garbage stream
matches itself. The GPU-fp16 ablation was healthy because it builds no ANE
program. See `docs/perf/mtp-accept-matrix-2026-09.md` for the collapsed rows.

## Probe on one program with real activations

`ANERealActivationCaptureTests` captures each layer's real multi-layer perceptron (MLP) input
(`MLX_ANE_CAPTURE_DIR`). `ANERealActivationProbeTests` runs the fused ANE
program on that input against an fp32 reference. Max absolute error, 5440
prefix channels, S=512:

| layer | ANE `silu` | ANE `sigmoid`·x | ANE `x/(1+exp(-x))` | ANE tanh form | GPU fp16 | ANE, no activation |
|---|---|---|---|---|---|---|
| 0 | 0.093 | 0.093 | 0.006 | 0.006 | 0.0005 | 0.004 |
| 1 | 0.009 | 0.009 | 0.003 | 0.003 | 0.0003 | 0.003 |
| 15 | 0.015 | 0.015 | 0.003 | 0.003 | 0.0002 | 0.002 |
| 31 | 0.032 | 0.032 | 0.002 | 0.004 | 0.0009 | 0.005 |
| 47 | 0.027 | 0.027 | 0.006 | 0.005 | 0.004 | 0.005 |
| 63 | 0.082 | 0.082 | 0.062 | 0.062 | 0.062 | 0.092 |

Four exact weight-folded rescalings (per-channel SmoothQuant, intermediate
rescale, both, global input scale) left the `silu` error unchanged to three
decimals on every layer. No value comes near fp16 range (max input 50,
max intermediate 209). The error is in the activation op, not in precision.

The production split object (ANE prefix plus 4-bit GPU suffix) on the same
inputs, against the all-GPU 4-bit MLP, is at the bf16 output rounding floor.
Layer 63: 2.0 on a scale of 520. Its concurrent and sequential paths are
bit-identical. This holds with all six probed programs loaded at once, which
is the regression check for defect 1.

## In situ, the model with the offload on

`MLX_ANE_VERIFY=1` computes the all-GPU MLP beside every offloaded call and
logs the error. bf16-sourced prefix, max absolute error against the 4-bit MLP:

| layer | before fix 1 | after fix 1 | output scale |
|---|---|---|---|
| 0 | 0.13 | 0.13 | 10.4 |
| 1 | 3.93 | 0.03 | 2.0 |
| 2 | 7.47 | 0.06 | 11.3 |
| 8 | 46.4 | 0.06 | 4.4 |
| 31 | 107 | 0.24 | 8.3 |
| 63 | 178 | 6.0 | 552 |

Before the fix the first program is right. Every later one is garbage.
After it every layer sits at or under about 1 percent of its output scale.
The larger layer-0 number is the bf16-versus-4-bit representation difference,
not an ANE error (dequantized source: 0.016).

## Goldens, 200 greedy tokens after a 512-token real prompt

`distinct` is the distinct-token ratio. `rep-8g` counts repeated 8-grams.

| prompt | GPU distinct | hybrid before | hybrid after | rep-8g after |
|---|---|---|---|---|
| README | 0.41 | 0.01 | 0.40 | 13 |
| qwen-prefill-research-plan | 0.48 | 0.01 | 0.56 | 0 |
| qwen-mtp-go-live-runbook | 0.56 | 0.01 | 0.20 | 82 |
| private-benchmark-security | 0.55 | 0.05 | 0.54 | 0 |

The runbook hybrid decodes as prose and then enters a greedy enumeration
loop (`serial-2 / serial-3 / ...`), the ordinary greedy failure mode. The
hybrid's greedy trajectory diverges from the GPU's at token 0, 0, 24, and 29
on the four prompts. The fp16 prefix explains that divergence. The prefix is a different
numeric representation, and the earlier GPU-fp16 ablation diverged at token 31 on the
same footing. Token identity with the 4-bit GPU is not the bar. Coherent
generation is.

## Speed

`ANEFusedSplitSpeedTests` at fraction 0.3125, S=512, fastest of 9 runs. The exp
spelling costs nothing measurable against `silu` (grouped 24.9 versus
33.7 ms, interleaved 28.5 versus 31.3 ms, single unpaired runs). The
1.3x prefill claim withdrawn on 2026-09-02 stays withdrawn until a gated
timing run with the fixed build measures it.

## Files

- `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEMILBuilder.swift`:
  `ANEActivation`, `programTag`.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEInMemoryModel.swift`:
  identity note.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/Qwen35ANESplitOffload.swift`:
  `MLX_ANE_ACTIVATION`, `MLX_ANE_VERIFY`.
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/ANEOffload/ANEActivationCapture.swift`:
  `MLX_ANE_CAPTURE_DIR`, `MLX_ANE_CAPTURE_LAYERS`.
- `Tests/MLXFastTests/Model/ANERealActivationCaptureTests.swift`,
  `Tests/MLXFastTests/Model/ANERealActivationProbeTests.swift`.
