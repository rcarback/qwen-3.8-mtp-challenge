# Decode optimisation methods for the Qwen3.8-Flash-Next fork

Catalogue of decode-time methods from the literature and from other Apple
Silicon engines, read against the fork's measured decode step. Compiled
2026-09-05 from the sources at the end. Estimates are exploratory unless a row
cites a measurement from `qwen38-flash-2026-09.md`.

## The step behind every estimate

Depth-0 serial decode on this M4 Max, after the dense-q8 fix:

| term | ms | measurement |
| --- | --- | --- |
| weight bytes, dense q8 plus experts q4 | about 21 | 5.9 GB per token at the measured 300 to 370 GB/s |
| launch cost | about 33 | 0.69 ms per layer. 5 to 12 µs per dependent launch. 60 to 100 ops per layer |
| whole step | 57.1 | steady state over 200 warm steps |

Launch cost is now the larger term. Each table below groups the rows by the
term they attack: launch, bytes, or per-token amortisation.

BaseRT gives the external reference point. On an M4 Pro its native Metal
engine beats MLX by 1.01 to 1.35x on dense Q4 models and by 1.01x on
`Qwen3-30B-A3B`, and loses to MLX on `Gemma-4-26B-A4B` at 0.90x. Dispatch
engineering alone, done well, is worth 0 to 35 percent over MLX on a dense
model and close to nothing on a MoE.

## Launch cost

| # | method | source | what | applies here | evidence in the fork | value | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | Whole-layer or whole-step graph capture | CUDA graphs, arXiv 2605.30571 | Trace the step once, replay it without per-op CPU work | `MLX.compile` plays the same role | 2605.30571 measured 1.26x on H100 where the launch band is 21 percent of the step. Here the band is 58 percent | up to 25 ms | blocked by the layer-2 host n-gram gather |
| L2 | Make the n-gram gather a graph op | this fork | Wrap the memory-mapped table as an MLXArray without copy and gather with `take`, or split the compiled region at layer 2 | needs a no-copy MLXArray over 102 GB of mmap, or two compiled regions | none yet | unlocks L1 | not built |
| L3 | Megakernel, persistent kernel | Mirage MPK arXiv 2512.22219, Hazy Research, Ada-MK 2605.11581 | One launch per step with an on-GPU task interpreter | CUDA tooling only. On Metal this means a hand-written kernel | the fork's single-launch MoE prototype was 48x slower at prefill geometry | large in principle | too costly to build |
| L4 | Indirect command buffers, zero-allocation decode loop | Metal ICB docs, BaseRT 2607.00501 | Encode the step's dispatches once at init and replay. No allocations on the hot path | MLX does not expose ICBs. Needs a custom Metal path | BaseRT: 1.01x over MLX on the MoE it tested | small on MoE | not built |
| L5 | Async graph pipelining | `mx.async_eval`, mlx-lm generate | Build step N+1's graph while step N runs on the GPU | the MTP session already uses `asyncEval` on the draft chain | the code records per-step asyncEval as neutral | 0 | tried |
| L6 | Fuse elementwise chains inside a layer | BaseRT, MLX compile docs | Norms, gates, residual mixes into one kernel each | done for the read gate and shared-expert glue | 2.1 percent, counterbalanced | a few ms more if the rest of the chain follows | partly done |
| L7 | Fused router tail | BaseRT routing fusion | Router GEMM, top-k, softmax in one launch | written | +1.0 percent slower at one row. Both sides sit on the 0.19 ms floor | 0 | refused |
| L8 | Count and remove host round trips | general | Every GPU-to-CPU readback per step is a stall | the layer-2 token readback and the sampling readback. Others uncounted | not measured | unknown | measure first |
| L9 | Fused gate+up gather GEMM | SonicMoE 2512.14080 | One launch instead of two per layer | built | -0.6 percent at prefill, within drift. Not measured at one row | 48 launches per token | re-measure at decode |

## Bytes

| # | method | source | what | applies here | evidence in the fork | value | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Quantize the dense weights | upstream `quant_predicate`, mlx-lm PR 1788 | Everything except the router | done at q8 | 64 to 57.1 ms, perplexity unchanged | 11.5 percent | default |
| B2 | Mixed q8/q4 policy | this fork | q8 on attention and `lm_head`, q4 on the shared expert, and q4 on the dense MLP | transform takes per-layer entries already | q4 everywhere costs 16 percent perplexity | part of the 15 ms between q8 and q4 | needs its own perplexity arm |
| B3 | Faster quantized GEMV kernel | arXiv 2605.30571 | Kernel quality decides bandwidth, not bit width. nf4 gave 1.05x and ExLlamaV2 3.59x on the same weights | `qmv` in the vendored MLX kernels | q4 g32 streams 301 GB/s, q8 369, bf16 441, against a peak of about 546 | q4 dense time falls 30 percent if q4 reaches bf16's fraction of peak | not started |
| B4 | Measure the expert gather at one row | this fork | 10 experts × 48 layers, 1.51 GB per token. Achieved GB/s at M=1 unknown | `gather_qmm` | none at decode | unknown | measure first |
| B5 | `lm_head` bytes | this fork | 248,320 × 2560 is 1.27 GB bf16 per token, 0.64 GB at q8 | included in B1 | none separate | included | done at q8 |
| B6 | Expert group-64 | standard | Halves scale bytes | rejected | 9 percent of routing decisions flip, and the g64 kernel is slower than g32 | 0 | dead |
| B7 | KV cache int4 | arXiv 2605.05699 | Quantized KV outruns fp16 on Apple Silicon at long context | 12 full-attention layers with 2 KV heads. KV is small at 1k context | none | near 0 at this context | not worth it |

## Per-token amortisation

| # | method | source | what | applies here | evidence in the fork | value | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | Deeper MTP chain with a better head | FastMTP arXiv 2509.18362 | Self-distilled MTP head. Acceptance holds past position 1. Speed peaks at K=3 at 2.03x | this fork may bring its own head | rows 2 to 8 cost 2.13x to 2.85x one row, 20 ms per token at depth 8 if accepted | the largest untried lever | not started |
| S2 | Confidence-adaptive draft depth | TALON arXiv 2601.07353 | Draft more when the head is confident, less when not | `draftPolicy` is a per-round function | accept rate falls from 0.89 to 0.63 as prompts lengthen | recovers the long-prompt loss | not started |
| S3 | Draft trees | EAGLE-2/3, mlx-lm discussion 890 | Verify more than one candidate branch per step | needs per-row position ids. MLX KV caches use one offset | mlx-lm on M3 Ultra measured each verify row at about 1 ms and 1.05x total. The fork's rows past the second are cheap | unclear | not started |
| S4 | Prompt-lookup or n-gram drafting | apoorvumang/prompt-lookup-decoding, PLD+ 2412.01447 | Draft from the prompt's own n-grams. 2 to 4x on input-grounded tasks | legal on this fork, which is not the ranked model | none | high on retrieval and code, near 0 on open prose | not started |
| S5 | Self-speculative layer skipping | LayerSkip 2404.16710, SWIFT 2410.06916, ConfLayers 2604.14612 | Draft with a subset of the model's own layers | the MTP head is a cheaper drafter that already exists | none | below S1 | skip |
| S6 | Lookahead decoding | arXiv 2402.02057 | Jacobi n-gram candidates verified in the same forward | more rows per step. Rows past the second are cheap here | none | unclear | after S1 |

## Mixture of experts rows

| # | method | source | what | applies here | evidence in the fork | value | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | Fused MoE decode op at M=1 | MobileMoE arXiv 2605.27358 | Selection, dispatch, gate/up, activation, down, and scatter in one op | the prefill-geometry prototype failed on tiling. At one row the work is 10 GEMVs, a different kernel | 48x slower at prefill. Unmeasured at M=1 | removes 3 launches per layer | re-scope to decode |
| M2 | Shared expert as expert 11 | this fork | Run the shared expert inside the gather instead of a separate branch | straightforward | none | 2 launches per layer | not built |
| M3 | Batch-aware routing | OEA arXiv 2511.02237 | Reuse experts across sequences in a batch | batch size 1 | not applicable | 0 | dead |
| M4 | Expert prefetch and offload | MoE-SpeQ arXiv 2511.14102 | Stream experts from host memory | every expert is resident | not applicable | 0 | dead |

## Reading

Three rows carry most of the remaining value, and they compound:

1. **L2 then L1.** Making the n-gram gather a graph op unlocks whole-layer
   compile, which attacks the 33 ms launch term directly. Nothing else in the
   launch table comes close.
2. **B3.** The q4 kernel streams at 55 percent of peak where bf16 reaches 80.
   That gap is kernel work inside the vendored MLX sources.
3. **S1 and S2.** Rows past the second are cheap, and the fork may bring its
   own head. FastMTP's self-distilled head is the published recipe.

## Sources

- Memory-Bound but Not Bandwidth-Limited, batch-1 decode on four GPUs,
  <https://arxiv.org/abs/2605.30571>
- BaseRT, LLM inference on Apple Silicon via native Metal,
  <https://arxiv.org/pdf/2607.00501>
- Mirage Persistent Kernel, <https://arxiv.org/html/2512.22219v1>
- Ada-MK adaptive megakernel search, <https://arxiv.org/html/2605.11581v1>
- Apple, encoding indirect command buffers,
  <https://developer.apple.com/documentation/Metal/encoding-indirect-command-buffers-on-the-cpu>
- MLX compile documentation,
  <https://ml-explore.github.io/mlx/build/html/usage/compile.html>
- Writing fast MLX (async_eval), <https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50>
- EAGLE-3 on Apple Silicon, mlx-lm discussion,
  <https://github.com/ml-explore/mlx-lm/discussions/890>
- FastMTP, <https://arxiv.org/pdf/2509.18362>
- TALON confidence-aware token trees, <https://arxiv.org/pdf/2601.07353>
- Prompt lookup decoding, <https://github.com/apoorvumang/prompt-lookup-decoding>
- PLD+, <https://arxiv.org/html/2412.01447v1>
- LayerSkip, <https://arxiv.org/pdf/2404.16710>
- SWIFT, <https://arxiv.org/pdf/2410.06916>
- ConfLayers, <https://arxiv.org/pdf/2604.14612>
- Lookahead decoding, <https://arxiv.org/pdf/2402.02057>
- MobileMoE, <https://arxiv.org/html/2605.27358v1>
- Opportunistic Expert Activation, <https://arxiv.org/pdf/2511.02237>
- MoE-SpeQ, <https://arxiv.org/html/2511.14102v1>
- int4 KV cache on Apple Silicon, <https://arxiv.org/pdf/2605.05699>
- MetalRT decode engine numbers,
  <https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon>
