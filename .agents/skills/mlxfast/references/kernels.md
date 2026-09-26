# Tune kernels at the model's actual geometry

Capture M, N, K, batch dimensions, strides, dtype, quantization, and target chip.
The production kernel at those shapes is the baseline. A convenient scalar or
per-expert loop can exaggerate a custom kernel's benefit.

## Dispatch before you build

1. Trace the selected kernel family for decode, relevant verify widths, and prefill.
2. Try existing tile, split-K, and layout variants before writing new arithmetic.
3. Include all compile-time specializations in pipeline cache identity. Separate
   treatment and control caches or prove that cache keys distinguish them.
4. Sweep around padding and dispatch boundaries. Choose a small deterministic
   predicate that selects the candidate only on validated shapes.
5. Compare tensor results, then required model fidelity and full-forward timing.
   Tile and split-K changes can change accumulation order and rounding.

Include tails, noncontiguous layouts, supported dtypes, and quantization metadata
in correctness tests. Reject unsupported inputs explicitly and test fallback.
Use direct kernel benchmarks to find candidates. They do not replace model timing.

## This fork's build path

The vendored MLX Swift build uses JIT source strings for many kernel families.
For those families, edit the AOT source and generated C++ runtime twin together.
The host dispatcher is a separate file, not the generated kernel source.

| Role | Repository path |
| --- | --- |
| Quantized matmul host dispatcher | `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp` |
| AOT quantized kernels | `Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h` and `quantized_nax.h` |
| JIT kernel strings | `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` and `quantized_nax.cpp` |

Inspect the current build for each family. AOT-only changes require rebuilding
`mlx.metallib` with `tools/build-mlx-metallib.sh`. Direct Swift builds/tests in this
repository require `--force-resolved-versions`. The ranked worker uses
`.build-worker/release/mlxfast-runtime-worker`, so an unrelated `.build/release`
binary does not validate the submitted runtime. Follow the current track's wrapper.

The `_nax` branch targets newer hardware and can bypass the plain row-tile path.
Testing with `_nax` disabled diagnoses the plain path only. It does not show
a win in the production M5 dispatch. Missing hardware blocks that performance
claim, not source review or correctly labeled local experimentation.

## Evidence and stopping rule

The local M4 experiments illustrate why predicates matter:

- `d4d3520`: bm=16 reported 1.30x and 1.38x at M=12 and 16 in its recorded setup.
- `34bd4fbc`: bm=64 averaged 1.08x at M=1024 but only 0.78x at M=96.
- `8f586dbf`: bm=128 lost to bm=64 at tested large shapes, and bm=256 failed compilation.
- `2285fab0`: custom decode GEMV variants failed to beat stock `qmv_fast`.
- `997ea5bf`: mismatched fused-path width bounds caused token divergences.

Read the experiment and current dispatcher before inheriting its numbers. Retain
shape-gated wins with tested fallback. Remove variants with no relevant winning
shape. A correctness failure blocks promotion even when the isolated kernel wins.
Use [measurement](measurement.md) for paired timings and records.
