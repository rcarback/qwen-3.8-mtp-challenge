# Prefill research plan: where the time actually goes

Measured 2026-08-25 on the local M4 Max (40 GPU cores, macOS 26.5.2). Every
number below comes from a microbenchmark in
`Tests/MLXFastTests/Model/PrefillMatmulCostTests.swift` or
`Tests/MLXFastTests/Model/GatedDeltaScanCostTests.swift`, run with
`MLXFAST_RUN_MLX_RUNTIME_TESTS=1`.

## The answer first

Prefill costs about 13.4 ms per token at a context depth of 8192. The budget
splits like this:

| Component | Layers | ms/token | Share |
| --- | --- | --- | --- |
| Dense projections (GEMM) | 64 | ~10.9 | 81% |
| Full attention | 16 | 1.65 | 12% |
| Gated-delta recurrence | 48 | 0.54 | 4% |
| Gated-delta depthwise conv1d | 48 | ~0.3 | 2% |

The projection row is by subtraction from the whole-model 13.4 ms; every other
row is measured directly.

Two of the three strategies in the original question do not apply, and the
third applies for a different reason than expected. The real finding is that
both large rows run far below what this hardware does on the same operation at
a different shape.

## What the measurements rule out

**The gated-delta scan is not the bottleneck.** The Metal kernel in
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift` runs a fully
sequential `for (int t = 0; t < T; ++t)` loop, which looks like the classic
serial-in-T term. It is not: 196,608 threads hide the dependency chain. All 48
layers together cost 0.535 ms per token at T = 8192, and the cost per token
falls as T grows (1.564 at T = 128, 0.535 at T = 8192).

This matters because the chunkwise parallel form of the gated delta rule -- the
WY representation and UT transform from Yang et al., as implemented in
`flash-linear-attention` and FlashQLA -- is the standard fix for exactly this
shape of problem, and it would be a large piece of work for at most 4% of the
budget. Do not start there.

**Chunk sizing is exhausted.** `Qwen36MTPBlockSession.prefillChunkRange`
already records the sweep: 13.38, 13.14 and 13.41 ms per token at chunk sizes
256, 512 and 1024. End to end, moving 4096 to 1024 shifted a 20k prefill by
1.8%. There is no headroom left in the knob as the code stands. See the second
opportunity below for why that is a symptom rather than a dead end.

**Quantization is not the tax.** At the model's real projection shapes, 4-bit
affine group-64 matmul and bf16 matmul run at the same speed. The measured
ratios are 0.55x to 2.85x with no consistent direction, which is noise around
parity. Dequantization overhead is a real concern in the MLX literature for
decode; it is not what prefill is paying for.

**The depthwise conv1d is not the bottleneck either.** MLX issues #2180 and
#2369 report the depthwise Metal path at roughly 9x PyTorch MPS, and this model
runs it at groups = 10240 in 48 layers, so it was worth pricing. It costs 0.05
to 0.32 ms per token across all 48 layers.

## Opportunity 1: full attention has no fused prefill kernel

`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp:625`
gates the fused prefill kernel on head dimension:

```cpp
const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
    (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);
```

Qwen 3.8 has `head_dim = 256`. The fused kernel is therefore never dispatched
during prefill, and attention falls back to the unfused path that materializes
the full score matrix. The decode path is unaffected, because
`sdpa_vector_supported_head_dim` two lines above does include 256.

That single line explains three separate observations:

- Measured prefill attention throughput is about 1 TFLOPS, against 14.7 TFLOPS
  for a square bf16 GEMM on the same machine in the same process.
- The `prefillChunkProductBudget` comment describes an `S x S` allocation per
  head reaching 307 GB at S = 80k. A fused kernel does not allocate that.
- Decode is healthy at 13.7 tokens per second while prefill is not.

**Work item.** Add head dimension 256 to the steel attention kernel and its
dispatch gate. Both forms must change together: the AOT source under
`backend/metal/kernels/steel/attn/` and the JIT twin
`mlx-generated/steel_attention*.cpp`, because the vendored package builds in
JIT mode. Confirm the register budget at D = 256 first; that is the reason the
list stops at 128, and it may force a smaller block tile.

**Editable-surface note.**
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp`
is not in `benchmark.json`'s `editablePaths`: all 89 entries under
`backend/metal` sit beneath `kernels/`, and this file is one level up from
that. The dispatch-gate change described above therefore applies to the local
server only and would not be packaged by `yukon submit`.

**Expected gain.** Attention is 12% of the budget at depth 8192 and grows with
depth. Closing most of a 14x gap on that row is worth roughly 1.5 ms per token
at 8192 and more at 30k, where the user's real sessions sit.

**Landed.** `sdpa_full_supported_head_dim` now includes 256. The edited file,
`scaled_dot_product_attention.cpp`, is a dispatch host with no separate
AOT/JIT pair of its own to keep in sync -- but that is true of this ONE
dispatch file, not of the steel attention kernel family it dispatches into:
`backend/metal/kernels/steel/attn/` and its JIT twin
`mlx-generated/steel_attention*.cpp` do form an AOT/JIT pair, same as every
other kernel family in this repository. The register budget did not force a
smaller tile for bf16/fp16 -- the
full BQ=32/BK=16/WM=4 tile fits at 29,184 bytes against the 32 KiB limit,
matching the estimate above. fp32 does not fit at that tile (Q_smem alone is
33,280 bytes), so the dispatch additionally drops to BQ=16/BK=8/WM=2 when
`q.itemsize() >= 4`; bf16/fp16 keep the full tile. The x16-layers ms/token
column, same `attentionCost` microbenchmark, before and after:

| T | Before | After |
| --- | --- | --- |
| 1024 | 0.520 | 0.04-0.11 |
| 4096 | 0.722 | 0.10 |
| 8192 | 1.649 | 0.21-0.27 |

A live inference server was running concurrently with these measurements
(the same one described in Opportunity 3), which is why the "after" column is
a range rather than a point: T=4096 held steady around 0.10 across repeated
runs, while T=1024 and T=8192 moved with server load. Even at the noisy end
this is a 4-6x drop in the attention row's contribution to the prefill
budget.

## Opportunity 2: projection GEMM runs at a quarter of the machine

The same machine, the same process, the same contention:

| Shape | M | TFLOPS (bf16) |
| --- | --- | --- |
| Square reference | 4096 | 14.69 |
| `mlp.gate/up` (5120 to 17408) | 4096 | 6.07 |
| `mlp.gate/up` | 1024 | 3.99 |
| `mlp.gate/up` | 256 | 3.75 |

Because this is a ratio taken under identical conditions, GPU contention and
the machine's current memory pressure cannot explain it. The model's GEMMs are
tall and thin: M between 256 and 1024 against K = 5120 and N = 17408. The tile
grid is lopsided and each threadgroup reloads a long K.

Note the trend. Throughput rises with M -- 1.31, 2.97 and 5.35 TFLOPS at
M = 256, 1024 and 4096 for the 4-bit `mlp.gate/up`. **Larger prefill chunks make
the GEMM four times more efficient.** The reason the chunk-size sweep found
nothing is that the unfused attention path (opportunity 1) makes large chunks
more expensive at exactly the same rate, and the two cancel.

**These two opportunities are therefore one opportunity.** Fix the attention
kernel first, then re-run the chunk sweep. The knob that reads as dead today
should come back to life, because the term that was cancelling it will be gone.

**Work item.** After the attention fix, re-sweep `prefillChunkRange` upward
against `prefillChunkProductBudget`. Then, if a gap to the square reference
remains, look at the steel GEMM tile selection for tall-thin shapes in
`backend/metal/matmul.cpp`. The `_nax` variants are not relevant on this
machine: `is_nax_available()` requires GPU generation 17 or higher, and an M4
Max is generation 16. They do apply on the ranked M5 box.

**Sweep result, 2026-08-26: attempted, abandoned, `prefillChunkRange` stays
at `256 ... 1024`.** The re-sweep the work item calls for was run twice
against the fused attention path, at caps 1024 / 2048 / 4096 / 8192, measuring
cold prefill of a ~12000-token prompt through `serve`. Both attempts were
discarded for measurement contamination, not for what they showed.

The host screensaver is the cause. Its idle timer (`idleTime` 300) restarts it
every five idle minutes, it takes 73 to 80 percent of a CPU, and GPU power
spikes from a 0.3 W idle floor to 9.2 W while it runs. Repeat samples of one
configuration spread 30 percent (cap 2048 read 145.84 s, 157.92 s and
190.21 s) against a 2.8 percent spread on a quiet machine. Two host facts are
worth carrying forward: `caffeinate -d` does not suppress the screensaver,
because display sleep and the screensaver idle timer are separate clocks, and
a `pkill` watchdog does not either, because `loginwindow` respawns it within
seconds. A clean re-run must stop the screensaver at its source first.

Every partial table agreed on direction: cap 1024 fastest, larger caps
monotonically slower. That matches the microbenchmark recorded in the
`prefillChunkRange` doc comment, so the bound stands and the work item is
closed. The prediction that the fused attention fix would revive this knob is
neither confirmed nor refuted; it remains open for a run on a quiet host.

One artifact of the discarded data is still unexplained and worth a look if
anyone resumes this: caps 4096 and 8192 both reported prefill 160.72 s, from
two distinct runs 4.5 minutes apart. `prefillChunkSize` derives
`prefillChunkProductBudget / cached` clamped to `[256, cap]` with `cached` as
the running position, so the two caps take different chunk schedules and
should not agree to 10 ms. The two candidates are a stale worker build
(`swift build --product ...` has been observed reporting success in this
repository without recompiling an edited file) or something structural that
collapses both caps onto one schedule.

## Opportunity 3: the prefix cache is working, and disk persistence is next

The restart at 20:10 activated chunked checkpointing, and the serve log shows
it doing its job:

```text
reusing 29396 cached tokens, prefilling 232      prefill 1.20s
reusing 33774 cached tokens, prefilling 2830
```

Turns that previously paid 250 to 450 seconds now pay 1 to 4 seconds. This is
the largest single improvement available and it is already landed. Remaining
work in this area, in order of value:

1. **Disk persistence**, LANDED 2026-08-25. Checkpoints are written to
   `DARKBLOOM_PREFILL_CACHE_DIR` as safetensors, guarded by a fingerprint over
   the weights identity, the running worker binary, the chunk size and the KV
   basis. Measured on a 5979-token prompt, restarting the server between runs:
   prefill 70.70 s cold against 0.36 s from the checkpoint, byte-identical
   output.

   Two things are worth carrying forward from how this was verified. The first
   working version rejected every cold restore, because the adopt path compared
   the checkpoint's layer count against the session's own cache array, which is
   empty until the first `begin` builds it. The unit tests passed because they
   restored into a session that had already prefilled. Only a real restart
   reaches the state the feature exists for.

   The second is that the failure was silent. The disk path deletes an
   unrestorable checkpoint and falls back to a full prefill, and it reports why
   on worker stderr, which `serve` does not forward. A permanently broken cache
   and a merely cold one produce the same log. `DARKBLOOM_PREFILL_CACHE_DEBUG_LOG`
   now names a file that receives the match, read and restore trail.
2. **Stride fill-in inside long turns.** Turn boundaries place no interior
   checkpoint, so a single 10,000-token tool result leaves a 10,000-token gap.
   Take the union of turn boundaries and stride boundaries.
3. **Non-prefix reuse**, from the CacheBlend and SparseX line of work. This
   handles repeated content that appears in a different position across
   requests, which prefix caching cannot match. Substantially more complex, and
   worth deferring until the two items above are done.

## What the literature offers that does not apply here

**Sparse attention and token pruning** (MInference, FlexPrefill, SlimInfer,
FTP) target the quadratic attention term. Only 16 of 64 layers in this model
are full attention, and that row is 12% of the budget. These methods also trade
accuracy. The unfused-kernel fix above addresses the same row without that
trade.

**Layer skipping and early exit** (LayerSkip, AdaInfer, ShortGPT, SwiftKV)
would cut the projection row, which is the row that matters. AdaInfer reports
17.8% of layers pruned for under 1% quality loss; SwiftKV requires fine-tuning.
There is a model-specific hazard worth stating plainly: skipping one of the 48
gated-delta layers does not merely drop one token's activations, it corrupts a
**persistent recurrent state** that every later token in the sequence reads.
The blast radius of a skipped GDN layer is the whole rest of the sequence,
which is categorically worse than skipping an attention layer. If this line is
ever pursued, restrict it to the 16 attention layers and the MLPs.

**Chunked parallel gated delta rule** is well-developed (chunk size 64, limited
by SRAM; the triangular solve is 22% to 31% of chunked cost, and the Neumann
series approximation removes it) and would be the right answer on a model where
the recurrence dominated. Here it addresses 4%.

## Recommended order

1. Land disk persistence for the checkpoint cache. Largest practical effect on
   the user's actual sessions, and independent of everything else.
2. Add head dimension 256 to the fused steel attention kernel.
3. Re-sweep the prefill chunk size, which should now be a live knob.
4. Measure again before touching GEMM tiling or anything in the literature
   section.

## Open question

The absolute throughput numbers were taken while the server was serving live
traffic (GPU at about 35%) and while the machine was swapping: 108 GB of 128 GB
resident and 9.0 GB of 10.7 GB swap in use. The ratios are sound, because both
legs of every comparison ran in the same conditions. The absolute TFLOPS
figures should be re-taken on a quiet machine before they are quoted anywhere
that matters.

The memory pressure itself deserves a look. `QwenSessionCacheBudget` clamps to
a quarter of physical memory, which is 32 GiB here, and at a depth of 33k a
single checkpoint carries roughly 2.2 GB of full-attention KV plus 144 MiB of
recurrent state. A handful of turn boundaries fills the budget.
