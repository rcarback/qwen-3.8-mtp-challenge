# Sources and what each established

The skill trusts independent reverse-engineering and this repository's own
hardware probes. Apple documents Core ML, not the engine; where an Apple page
is cited it is for a documented API surface, not for a claim about ANE
behavior.

## Independent analyses

| source | what it established |
| --- | --- |
| Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance*, arXiv 2606.22283 (302 pages, M1 measurements) | The datapath reconstructs a compressed weight to fp16 at the multiplier input; int8, int4 and palettized weights are dequantized on the way in. int4 palette streams natively at about 2.37x the bandwidth and runs about 2.37x faster than fp16. Structured sparsity with at least half zeros runs 1.55 to 1.64x faster at 0.43x the bytes. int8 affine and blockwise forms fold to dense fp16 in conversion. DRAM roofline about 85 GB/s on M1, 12 fp16 TFLOP/s. Per-dispatch budget about 190 us, of which about 98 percent is software and firmware; `ANE_ProgramSendRequest` about 163 us. "GPU leads on bandwidth-bound autoregressive decode." |
| Bryngelson, *ane-guide* (readthedocs / GitHub), the kernel-driver chapter | The `H11ANEIn` IOKit selector tables (control client and direct-path client), `DeviceOpen` sizes, `ProgramCreate` and `SendRequest` layouts. Used by `tools/ane-load/load_probe.m`. |
| Orion, *Characterizing and Programming Apple's Neural Engine for LLM Training and Inference*, arXiv 2603.06728 | The ANE compiler keeps internal state that limits a process to about 119 compilations, after which compilations silently fail; it is compiler-side, not hardware. Workarounds: `exec()` restart (about 50 ms) or unload and reload existing programs with new weights. XPC plus IOKit dispatch about 0.095 ms. |
| skyfallsin, *Apple Neural Engine field guide* (GitHub) | Decode latency fits about 119 us plus bytes over 78 GB/s on M3 Max. Cross-process concurrency about 5. The `0x50004` failure is runtime program-load behavior, not engine saturation. Host-side IOSurface work is not the dominant cost. |
| maderix, *Inside the M4 Apple Neural Engine, Part 1* (Substack) | M4-specific direct access. The MIL text form `program(1.3) [buildInfo = ...] { func main<ios18>(...) {...} -> (out); }`, compiled to the E5 binary with model type `kANEFModelMIL` and plist `model.mil`. Evaluation requests carry `procedureIndex`. |
| thebasedcapital, *ane-infer* (GitHub) | Multi-procedure MIL programs dispatched by `procedureIndex`; intermediates stay in IOSurfaces; a fused feed-forward block (eight ops) as one dispatch reached 3.6 TFLOPS; the `_ANEIOSurfaceOutputSets` fix. |
| jundot, *oMLX* (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`) | The MIL text builders this repository ported: `fp16_linear_mil`, `fp16_swiglu_down_mil` (whole MLP as one program), `*_bank_mil`. The `make_blob` chunk-descriptor weight format. The compile recipe: stage `model.mil` plus `weights/` under `NSTemporaryDirectory()/<identifier>`, do not override `modelURL`, `compileWithQoS` then `loadWithQoS`. Issue #2781: a packed dual-ANE bank can fail load with `0x20004` and fall back to per-layer. |
| AnandTech, *Apple M1 Max performance review*, page 5 | The CPU P-cluster reads DRAM at about 224 GB/s, 243 with the E-cores, against the 409 GB/s the chip can deliver: the CPU never reaches DRAM peak, the fabric caps it. |
| Draw Things engineering blog, *Making the Apple Neural Engine work in a custom stack* | Practical ANE integration outside Core ML. |

## This repository's hardware probes (M4 Max, macOS 26.5.2)

| probe | result |
| --- | --- |
| `Tests/MLXFastTests/Model/ANEProgramCountLimitTests.swift` | 126 loaded programs per process, identical at 8 KB and 26 MB each; the 127th fails `0x50004`. |
| `ANEProcedureBankProbeTests.swift` | Loader counts programs, not functions: 512 functions resident in 64 programs. Only `main` dispatches through the bare in-memory path; `procedureIndex >= 1` inference-errors under every naming scheme. |
| `ANEMultiFunctionProbeTests.swift` | The multifunction `.mlpackage` path dispatches every declared function by name. The in-memory `MLModelAsset(specification:)` blob rejects a multifunction description. Zero-copy: output via `outputBackings` reads at 0.005 ms; predict 0.632 ms vs quantized GPU 0.684 ms (0.92x); the 0.22 ms input stage is a standalone GPU launch floor, not data movement. `_ANEModel`/`_ANEClient` load Espresso `model.espresso.net` and cannot consume an ML Program `model.mil`. |
| `ANEQuantizedWeightProbeTests.swift` | int8 affine weight storage runs on the ANE at 0.8 percent relative error. fp16 at 0.03 max-abs. int4 blockwise (iOS18 op) compiles from coremltools but the in-memory ANE compiler returns `InvalidMILProgram` with byte-identical blob format; open. |
| `MemoryHeadroomTests.swift` | GPU alone about 189 GB/s; ANE alone 107 to 110; CPU 121 to 129 (four threads); GPU+ANE about 217; GPU+CPU about 163; all three about 149. Two engines never beat the GPU alone in that harness. |
| `Qwen4ExpANEBench` (M=1..8 sweep) | ANE per-program floor 0.30 ms including staging; every decode shape 5 to 10x slower than in-graph GPU; `lm_head` 10.9 ms vs 1.65 ms. |
| `ANEFusedSplitSpeedTests` (decode shapes) | S=1 split 7.5 to 8.3 ms vs GPU 0.56 to 1.1 ms; S=128 parity 0.94 to 1.05x. `MLX_ANE_MIN_SEQ=128` is the balance point. |
