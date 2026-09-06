# Chip support: what each Apple Silicon generation is measured to do

Apple publishes no per-op, per-chip ANE table. What exists is independent
measurement on specific parts plus a few documented capability statements. So
this table records the evidence per generation and marks the rest unknown.
The MIL op set the ANE compiler accepts is, as far as the sources show, the
same across generations; what changes by generation is bandwidth, dispatch
cost, the program budget, and whether int8 activation compute exists.

## Capability by generation

| capability | M1 (A14-class ANE) | M2 / M3 Max | M4 Max (this box) | A17 Pro / M4 class, documented |
| --- | --- | --- | --- | --- |
| fp16 conv / matmul | yes (paper) | yes | **confirmed** | yes |
| ANE DRAM read | ~85 GB/s roofline (paper) | ~78 GB/s fit, M3 Max (field guide) | **107 to 110 GB/s measured** | |
| ANE fp16 compute | 12 TFLOP/s (paper) | | | |
| per-dispatch floor | ~190 us, ~98 percent software/firmware; `ANE_ProgramSendRequest` ~163 us (paper) | ~119 us fixed plus bytes/78 GB/s (field guide); ~95 us XPC+IOKit (Orion) | **0.30 ms per program including surface staging** | |
| int8 weight storage, dequantized to fp16 on-chip | yes (paper: "folds to dense fp16") | | **confirmed, 0.8 percent error** | yes |
| int4 palette (LUT) weights | yes, ~2.37x fp16 bandwidth and speed (paper, M1) | | **confirmed**, in-memory compile, 15.9 percent error at a per-tensor uniform palette | yes |
| int4 blockwise (affine group) weights | accepted (paper) | | **rejected**, the in-memory compiler refuses the iOS18 op with a byte-identical blob | yes (Core ML int4) |
| structured sparsity (>= 50 percent zeros) | 1.55 to 1.64x at 0.43x bytes (paper) | | not probed | |
| int8 activation compute (int8 x int8) | no evidence | no evidence | not measured here | **advertised**: "increased throughput for int8-int8 compute on Neural Engine" on A17 Pro and M4 |
| fp32 compute | no, fp16-native (paper) | no | **rejected by compiler** | no |
| fp4 / MXFP4 / NVFP4 | no | no | no (GPU-only: Metal 4.1 / MLX) | no |
| loaded programs per process | | | **126** (127th fails 0x50004) | |
| compilations per process | ~119 before silent failure (Orion; part unstated, M-series) | | | |
| cross-process ANE concurrency | | ~5 (field guide, M3 Max) | | |
| multifunction `.mlpackage` dispatch | | | **confirmed** | requires macOS 15 / iOS 18 |
| direct `_ANERequest` `procedureIndex` dispatch | (ane-infer, part unstated) | | **confirmed for procedure 0** | |

## Reading the table

- **Bandwidth is the axis that moves.** M1 about 85, M3 Max about 78 (a fit,
  not a peak), M4 Max 107 to 110 measured. All are the narrowest path on
  their chip; the GPU on the same M4 Max reads 330 to 440 isolated against a
  546 peak. No generation in the evidence makes the ANE the right engine for
  a bandwidth-bound step.
- **The dispatch floor is software, and it is slowly falling**: about 190 us
  on M1, about 95 to 120 us on M3 Max class, and our 0.30 ms on M4 Max
  includes fresh IOSurface staging per call that persistent surfaces remove
  (the field guide's 119 us is the floor to aim at).
- **int8 activation compute is the one documented capability that could change
  the decode verdict**, and it is unmeasured here. A17 Pro / M4 advertise
  int8-int8 on the engine. If it delivers, the ANE's effective throughput on
  int8 activations rises without a bandwidth change. Measure it before
  assuming.
- **The program budget and the multifunction path are the only M4-only
  facts here** because they were measured only on this box. Expect the count
  to be process-software state on every generation (Orion and the field guide
  agree), but re-measure the number.

## Where a claim comes from

Paper: Bryngelson, arXiv 2606.22283 (M1). Field guide: skyfallsin (M3 Max).
Orion: arXiv 2603.06728. Documented int8-int8: the Core ML optimization
overview, a documented API statement rather than an engine measurement. "This
box": the probes listed in `sources.md`, M4 Max, macOS 26.5.2. Blank cells are
absence of evidence, not absence of capability.
