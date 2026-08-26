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

**Every number in the table above is an artifact. Do not quote it.**
Re-measured on 2026-08-26 the same shapes read three to nine times higher, and
the ordering the table rests on inverts. Two independent defects produced it,
and both are recorded here because the second is easy to repeat.

The first was already flagged in "Open question" below: those readings were
taken while the server was serving live traffic and the machine was swapping
9.0 GB. The second is new. Position inside the measuring process dominates
every number this suite produces. The square bf16 reference reads 14.69, 14.60,
14.66, 14.65 and 14.74 TFLOPS when its block runs first in a fresh process, and
6.35 to 6.62 when any other measurement block ran before it in the same
process. The effect survives `Memory.clearCache()`. It does not survive a
process boundary: a fresh process reads 14.60 immediately after a heavy GEMM
run in a separate process, which rules out GPU clock and thermal state. One
isolated aspect-sweep point read 7.34 and then 11.63 TFLOPS on consecutive
runs.

That defect defeats the argument the table was resting on. The claim was that
the ratio is trustworthy because both legs ran under identical conditions. They
did not: the square reference ran first in the process and the projections ran
later, so the control sat at a privileged position on the very gradient that
dominates the measurement.

**GEMM gap attribution, 2026-08-26. There is no tall-thin gap.** Measured one
shape per process with `tools/gemm-point-sweep.sh`, `tools/host-quiet-gate.sh`
passed before every point, no server running. Every point below uses the same
call shape, so the projection and the square reference are directly
comparable.

| Point | mode | M | N | K | ms | TFLOPS |
| --- | --- | --- | --- | --- | --- | --- |
| Square reference | bf16 | 4096 | 4096 | 4096 | 10.298 | 13.35 |
| Square reference | 4-bit g64 | 4096 | 4096 | 4096 | 11.304 | 12.16 |
| `mlp.gate/up` | bf16 | 256 | 17408 | 5120 | 3.472 | 13.14 |
| `mlp.gate/up` | 4-bit g64 | 256 | 17408 | 5120 | 3.773 | 12.10 |

The production projection at its real prefill shape runs at 12.10 TFLOPS
against a square 4-bit reference of 12.16, which is 99.5 percent of it. The
bf16 pair agrees: 13.14 against 13.35, within 2 percent. Quantization costs
about 9 percent and shape costs nothing measurable. The premise of this
section, that the projection GEMM runs at a quarter of the machine, does not
survive isolated measurement.

**The tiling target named in the old work item is not on the production path.**
`Qwen35Ops.linear` (`Sources/MLXFastModel/Qwen35Ops.swift:33`) sends a weight
with scales to `quantizedMM` and only a dense weight to `matmul`. Every
production projection is 4-bit affine group-64, so all of them enter
`quantized.cpp` and none reach `GEMM_TPARAM_MACRO` in
`backend/metal/matmul.cpp`. The `max(M, N)` masking in that macro is real, but
it governs dense GEMMs, and dense attention now goes through the fused SDPA
kernel. The probe that was planned against it was not run, because it would
have measured a path production does not execute.

For the record, the quantized path reaches `qmm` with fixed tiles.
`QuantizedMatmul::eval_gpu` (`quantized.cpp:1418`) sees M = 256 above the
vector limit, so it is a matrix-matrix product; the weights are transposed and
the batch is 1, so it calls `qmm_splitk` (`:776`). There `split_k` computes as
`max(1, 512 / (544 * 8))` = 1, which falls straight through to `qmm` (`:682`)
with `bm = 32, bn = 32, wm = 2, wn = 2`. Those tiles do not depend on M. The
`qmm_nax` variant (`:473`) does carry 64 x 64 x 64 tiles, but
`is_nax_available()` requires GPU generation 17 and this M4 Max is generation
16, so it is reachable only on the ranked M5 box.

**Follow-on work: closed.** No gap survives isolated measurement, so the
tall-thin GEMM tiling work has no target on this machine. The one open thread
is `qmm_nax` on the ranked M5, which this machine cannot measure at all.

**Method note.** Any future measurement in `PrefillMatmulCostTests` must run
one shape per process. `tools/gemm-point-sweep.sh` does this and gates on a
quiet host before each point. A table produced by a loop inside one test is not
evidence, whatever it shows.

**These two opportunities are therefore one opportunity.** Fix the attention
kernel first, then re-run the chunk sweep. The knob that reads as dead today
should come back to life, because the term that was cancelling it will be gone.

**Work item.** After the attention fix, re-sweep `prefillChunkRange` upward
against `prefillChunkProductBudget`. Both halves of this work item are
now closed by measurement. See "GEMM gap attribution" below: there is no gap
to the square reference, and `backend/metal/matmul.cpp` is not on the
production path. The `_nax` variants are not relevant on this
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
closed. The prediction that the fused attention fix would revive this knob was
neither confirmed nor refuted by these discarded runs. The quiet-host re-run
recorded below refutes it.

One artifact of the discarded data is still unexplained and worth a look if
anyone resumes this: caps 4096 and 8192 both reported prefill 160.72 s, from
two distinct runs 4.5 minutes apart. `prefillChunkSize` derives
`prefillChunkProductBudget / cached` clamped to `[256, cap]` with `cached` as
the running position, so the two caps take different chunk schedules and
should not agree to 10 ms. The two candidates are a stale worker build
(`swift build --product ...` has been observed reporting success in this
repository without recompiling an edited file) or something structural that
collapses both caps onto one schedule.

**Re-sweep result, 2026-08-26 (quiet host): the prediction is refuted and the
bound stands.** The run above was repeated on a host held quiet by
`tools/host-quiet-gate.sh`, which refuses to start a measurement while the
screensaver idle timer is armed, while `legacyScreenSaver` is running, or
while GPU power peaks above 2.0 W over an eight-second window. The gate read
`gpu peak 0.08W 45.5C` before this measurement started.

The instrument changed as well as the host. The chunk cap became a runtime
value (`DARKBLOOM_PREFILL_CHUNK_CAP`, floor 512, ceiling 16384), so one binary
serves every sample and the stale-build candidate above is removed by
construction rather than argued away. Cost is measured in process, per
appended chunk, against `callWithHidden` — the same backbone forward the
server runs.

Appended-chunk cost against the fused attention path, seconds for one chunk of
`chunk` tokens appended at depth `cached`:

| chunk | cached 0 | cached 2048 | cached 8192 | cached 16384 |
|---|---|---|---|---|
| 256 | 9.160 | 9.647 | 10.284 | 11.015 |
| 512 | 9.293 | 9.687 | 10.216 | 11.279 |
| 1024 | 9.583 | 9.998 | 10.688 | 12.025 |
| 2048 | 9.889 | 10.363 | 12.204 | 12.306 |
| 4096 | 10.666 | 11.036 | 11.677 | 12.975 |

Values are milliseconds per token, so a column is directly comparable down its
length. The widest chunk costs more per token than the narrowest at every
depth: 16 percent more from chunk 256 to chunk 4096 at depth 0, 18 percent at
depth 16384. The rise is not strictly monotonic. At depth 8192 chunk 512 reads
0.7 percent under chunk 256, and chunk 4096 reads 4 percent under chunk 2048,
which is the one point in the grid that breaks the order. That row is a
candidate for a repeat sample if anyone revisits this, but it does not change
the reading: every depth puts its most expensive point at chunk 2048 or wider,
and no depth makes a wide chunk cheaper than the narrow ones. The prediction
that the
fused attention kernel would make wide chunks pay is refuted, and the
mechanism is visible in the shape. Wider chunks do improve projection GEMM
efficiency, which was the basis of the prediction, but attention within a
chunk is quadratic in the chunk width, and fusing the kernel lowered the
constant on that term without removing the term. The quadratic growth outruns
the GEMM saving across the whole measured range.

Whole-prompt prediction for an 11682-token prompt, summing each cap's real
chunk schedule over the surface: cap 1024 is 120.81 s, cap 2048 is 129.59 s
(1.073x), cap 4096 is 130.84 s (1.083x). `prefillChunkRange` stays at
`256 ... 1024` and the work item is closed by measurement rather than by
abandonment.

One reading from this run must not be quoted: an early version of the
prediction table reported cap 8192 at 84.76 s, or 0.702x, which would have
read as a 30 percent win. It is an artifact of the instrument. The surface is
measured out to chunk 4096 and clamps outside the measured box instead of
extrapolating, which is the correct choice for a lookup. Summing clamped
lookups is not: a schedule of 8192-wide chunks charges 8192 tokens the cost of
4096 and halves the total by construction. The bias runs toward wide chunks,
which is the direction the hypothesis wanted, so the prediction now reports
`unmeasured` for any cap whose schedule leaves the grid
(`ChunkCostGrid.predictionLeavesGrid`). Deciding whether a cap above 4096 pays
needs those points measured, not interpolated, and the table above gives no
reason to expect it would.

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
