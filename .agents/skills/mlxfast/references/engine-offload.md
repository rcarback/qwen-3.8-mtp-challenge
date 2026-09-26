# Share dense work between the ANE and GPU

## What it is

Dense engine offload splits a dense, non-MoE subgraph across two engines.
One part executes on the ANE, and the rest stays on the GPU, instead of
running the whole path on one engine. It targets MLPs, attention
projections, shared experts, and other sufficiently large static subgraphs.
The mechanism repacks or requantizes weights for a second engine. It needs
the same numerical and fidelity checks any execution or representation
change needs.

The sibling [ane-gpu-offload skill](../../ane-gpu-offload/SKILL.md) covers
MIL, IOSurface, and runtime mechanics. Its mechanism references retain historical claims about
weight compatibility, fp16-only execution, decode, and program banks. Use the
current entrypoint and this guide for decisions, and verify the tested program.

## When it applies

- The candidate subgraph is dense and static, such as an MLP, an attention
  projection, or a shared expert. It does not cover the MoE router or a
  routed-expert gather.
- The objective is request latency or a stated energy or memory budget.
  Using both engines is not itself a benefit.
- A MoE model can still contain a useful dense share. Measure that share to
  bound the opportunity instead of rejecting the whole architecture.
- Promotion requires a measured gain after dispatch costs, a validated weight
  conversion where needed, and the deployment's required fidelity checks.
  Missing evidence leaves an experiment unproven rather than making the
  technique inapplicable to every shape.

Separate three questions before drawing a conclusion: can the graph compile,
where does it execute, and what numbers does it compute. Preferred placement
and a compiled program count do not prove the treatment executed inside the
model.

## How to do it

1. Inventory the weight forms the target engine actually accepts. Local
   September 2026 probes found:

   | Weight form | Observed placement and limitation |
   | --- | --- |
   | fp16 | ANE eligible, but engine rounding can differ from the GPU |
   | int8 per-channel affine | ANE eligible, a different representation from a blockwise int8 form |
   | int4 palette or LUT, including grouped LUTs | ANE eligible in tested programs, with measurable quantization error |
   | Blockwise int4 group 32 or 64, and int8 group 32 | Tested convolution variants placed on the GPU |

   These are program- and compiler-specific results, not proof of universal
   support or impossibility. Apple documents W8A8 compute on newer ANEs,
   including M4. A failed custom activation probe does not refute that path.
   Treat W8A8 as a separate calibrated experiment with conversion,
   placement, and quality checks.
   [Apple compression overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).

2. Account for bandwidth before choosing a split fraction. Weight
   compression saves bandwidth only when weights stay compressed until use.
   Some backends expand weights before execution. More LUT groups can also
   raise runtime cost.
   [Apple palettization performance](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html).

   For a fraction `f` of MLP intermediate channels assigned to the ANE,
   assume each engine reads its partition once. This estimate excludes metadata,
   padding, and activation traffic:

   ```text
   split_weight_payload_traffic / GPU_q4_weight_payload_traffic = (1 - f) + f * b_ANE / 4
   ```

   At `f = 0.3125` against a deployed q4 group-64 GPU baseline, the ratios
   are 1.9375 for fp16, 1.3125 for int8, and 1.0 for int4. Int4 cuts traffic
   against the fp16 ANE arm. It adds no nominal byte reduction against the
   q4 GPU-only baseline in this model. If the GPU still computes the full
   projection, count those duplicated bytes too.

3. Separate traffic from residency. Full GPU weights can stay resident for
   decode and fallback while an ANE program adds another representation.
   Count scales, LUTs, alignment, duplicated inputs, partial outputs,
   padding, and the GPU addition. Zero-copy sharing removes a copy, not the
   producer write or the consumer read.

4. Repacking unchanged effective weights for another engine still needs the
   deployment's numerical and fidelity checks. Requantizing target weights
   or activations needs permission and a declared quality budget too.

## Options to try

| Option | Expected effect where measured | What makes it lose |
| --- | --- | --- |
| Split a fused dense MLP by intermediate channels | The ANE computes the gate, up, activation, and down prefix. The GPU computes the suffix | Charge full-output writes and the join, not the split compute alone |
| Split gate and up, keep down on the GPU | Join activation slices before down | The extra boundary can exceed the saved down cost |
| int4 per-tensor palette, fraction 0.3125 | Prefill gained 7.1 percent in a back-to-back sweep and 14.8 percent with a 45-second cool gap between prompts, same fraction under two thermal protocols | Costs perplexity in proportion to its share, priced under Stop conditions |
| Split independent token rows | Begin with token-local dense layers | Charge duplicated weight reads and preserve attention/recurrent ordering outside the split |
| Independent dense projections, overlapped | Overlap only branches with ready inputs, synchronized before their first shared consumer | Contention on shared memory beats isolated timing |
| Larger fused dense subgraph | Amortizes dispatch cost at small row counts | Needs proven operator support and numerical correctness first |
| W8A8 or another supported compressed program | A separate calibrated experiment | Measure placement, traffic, and full-forward latency apart from a weight-only change |

Start with one layer, one static fraction, and one production weight form.
Vary fraction and precision independently before testing their interaction.
Try static sequence buckets or smaller tiles before an adaptive scheduler or
a program bank.

## Test plan

Apply [measurement](measurement.md) for the arm protocol. Measure standalone
GPU, standalone ANE, sequential split, and concurrent split:

```text
split time = preparation + max(concurrent_GPU_leg, concurrent_ANE_leg) + join
```

Measure concurrent legs under contention, because isolated times do not
predict shared-memory interference. Sequentially dependent work has no such
overlap to measure.

Cover the minimum eligible row count, bucket boundaries, representative
prefill, and short-row fallback. Test odd channel sizes, split fraction
zero, and distinct weights across layers. Charge compilation where requests
pay it. Check live program count, identity, buffer lifetime, and fallback
after failures. Test output identity on exact staging operations, and test
required model fidelity on the combined path.

Use an unloaded baseline first. Add repeatable competing GPU load only when
relevant to the deployment. Record both control drift and achieved
competing load before claiming a load-dependent optimum.

## Measured examples

Local M4 Max, 128 GiB, macOS 26.5.2 observations recorded in
`docs/perf/ane-gpu-reevaluation-2026-09-06.md`:

| Experiment | Result |
| --- | --- |
| Dense fused prefix, hidden 5120, inner 5440, S=1024 | fp16 33.89 ms, int8 18.92 ms, int4 14.96 ms. Int4 is 2.27x fp16 on this isolated leg, not a measured request speedup |
| Dense int8 split, fraction 0.5, unloaded | 145.5 tokens per second against controls 124.8 and 128.2. Perplexity 5.571 against 5.566 on six prompts is limited quality evidence |
| Dense int8 split, fraction 0.625, under a 40 percent duty synthetic GPU load | 26.7 percent above the first control, paired range 1.07 to 1.53. The source calls this a candidate from a noisier loaded-box sweep, not a settled default |
| Small single-row ANE programs | Lost 5 to 10x with about 0.30 ms dispatch overhead. This discourages those programs, not every future fused decode design |
| Split program banks | Lost throughput even after placement returned to the ANE. Package size and dispatch costs were not isolated, and the losing bank runtime was removed |
| MoE micro-batch restructuring | Lost about 10 percent before ANE work. Dense and shared-expert offload did not recover that cost |

## Stop conditions

Retire a design when no relevant shape clears both the correctness and the
request-cost gates. Retain a measured static win with a tested fallback.
Reopen a rejected design when a cost changes, such as fusion, padding,
representation, or compiler support. Do not build a bank or an automatic
load policy solely to keep an experiment alive.

Quality has its own budget. The int4 per-tensor palette already costs 3.7
percent of perplexity at fraction 0.3125, and 6.4 percent at fraction 0.5,
on the split lane. It costs 10.0 percent when applied to the MoE shared
expert. A design that needs a larger int4 share to win on speed needs a
matching quality receipt before it clears this gate.
