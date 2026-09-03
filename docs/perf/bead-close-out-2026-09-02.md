# Bead close-out, M4 Max, 2026-09-02

This record closes the five open beads of the local performance fork
(`local/perf-2026-08`): `a3f` wide-verify internals, `367` n-gram
prompt-lookup drafting, `506` MTP head chain cost, `2i7` prefill compute,
and `mbn` the ship gate. It also closes the host-synchronization plan, whose
Tasks 2 and 3 are recorded in `docs/perf/decode-round-trace-m4max-2026-09.md`.
Every measurement below ran on 2026-09-02 on an Apple M4 Max with 128 GB,
and numbers taken from earlier records are cited to their source.
The binaries were built at 16:43 from `70eba8a` (the ANE fix `37538ee` and
the wall lane merge) plus the host-QoS code later committed as `83c41b0`;
the worker was rebuilt for the two ANE fixes below. Later commits are
records only.
The thermal gate's GPU sensor read an implausible 1.6C on most runs after
17:00, as it did on the Task 1 run. Where that matters the text says so.

The bead notes carry the same facts in `bd show` form. This file is the
version that survives a `bd` reset.

## What changed in the tree today

| commit | change |
|---|---|
| `37538ee` | ANE hybrid: unique program identity per layer; SiLU spelled through `exp` |
| `70eba8a` | merge of the wall lane: three width walls named, caps runtime-settable, `provenExactDepthCeiling` |
| `83c41b0` | `MLX_MTP_HOST_QOS` (host-sync Task 2), measured, no win |
| `a02ddc3` | host-sync Task 3 arm sweep, no winner |
| `ae2616b` | ANE: remove a program's staging directory once it is loaded (the temp folder had filled the disk) |
| `dd8b852` | head cost-ratio re-fit recorded in the constant's comment; stale readout claim corrected |
| `392411b` | ANE: program identity from the weight-blob hash instead of a random tag |

No default moved. Every lever measured today stays off, and the one that
pays (lookup drafting on copy-heavy input) stays an explicit switch.

## a3f: Wide-verify internals

Landed: the A1 width probe, the BM tile fix (`d4d3520`), the depth unlock
(`bcc8c67`, `65f0d75`, `f5a3449`), and the wall lane. The wall lane found
three width walls, not one. The sdpa fused vector path ends at width 5.
The QMV replica's per-row exactness and the fused gated-delta in-projection
both end at width 9, and the in-projection bound is now tied to the replica's.
It proved widths 6 to 16 bit-exact per row with the two settable bounds
raised (`MLX_QWEN_QMV_MAX_WIDTH=16`, `MLX_QWEN_MTP_MAX_DRAFT_DEPTH=15`) and
the depth forced per run, on one prompt and one machine. It left the shipped
ceiling at width 9 (depth 8) with both schedule caps unchanged at 5 and 7.

Measured today:

- Contract suites with the two width variables unset: `QwenVerifyDepthCapTests`
  7 of 7, `QwenDepthAndWidthBoundTests` and `WideVerifyLadderBandTests` 17 of
  17. One pre-existing failure is unrelated:
  `Qwen35ArtifactContractTests.qwen36ConfigContractDigestMatchesTheReferenceManifest`
  fails identically on `d4d3520`.
- Depth-15 serve arm with the two width bounds and the schedule cap raised
  (`MLX_QWEN_MTP_SEGMENTED_VERIFY_DEPTH_CAP=15`, `--mtp-depth 15`) on the
  public, zebra and raccoon prompts: identical rounds, accept rate and text to
  the depth-8 arm. The offered depth reads 15, the effective depth stays
  between 2.8 and 3.9. The cost model never takes a deep round on this
  material, so the unlock is inert. The repeat-heavy prompt was not run on
  this arm.

The width-9 input-group item is scoped, not done. At the shipped head
schedule a round has at most 8 rows, so width 9 runs only in the untimed
session warm. The lookup ladder does dispatch it: today's lookup arm ran 15
rounds at 9 rows and 36 at 32 rows, and widths 16 and 32 sit outside both
proven bounds at shipped defaults. The honest ceiling on the change is about
19.5 ms on a width-9 round (the eval-only 8 to 9 step), not the 19% the
earlier note carried. It stays open only if lookup drafting ships on.

Residuals: `sdpaWidthWallDepthCap` is declared and settable but never read
as a gate. `measuredRawDepthPrice` is an 8-element array whose precondition
aborts under the `.pbfit` price arm with a raised ceiling (inert at the
shipped arm). The quantized-KV path has no width guard. `Qwen35FusedMLP`
fuses gate and up projections through width 16 while its comment claims
width 9. The exactness proof is one prompt on one machine.

## 367: N-gram prompt-lookup drafting

Landed and merged, default off (`DARKBLOOM_QWEN_LOOKUP_DRAFT=1` enables).
The worker banner marks it serve-only: the ranked and gate verbs reject a
round wider than 8. The final review had downgraded the corpus gate to
marginal and required that two rung-8 threshold settings be measured. Both
were. The tests and the serve arms load the pinned bf16 head.

Real-weight tests (release bundle):

- `lookupEquivalence`: the emitted streams agree over 192 tokens. On the
  repeat-heavy seed, lookup off took 26 rounds and lookup on took 6, with
  186 of 186 proposed tokens accepted.
- `lookupWidthCost` at fill depth 2048 and 20480 (width: total ms, tape MiB,
  repair ms; the repair path read `replay` at every width):

| width | 2048 | 20480 | tape MiB | repair ms |
|---|---|---|---|---|
| 4 | 57.6 | 65.4 | 154 | 2.2 / 2.4 |
| 9 | 179.3 | 220.6 | 164 | 2.7 / 2.7 |
| 16 | 197.3 | 290.2 | 177 | 3.1 / 2.8 |
| 32 | 299.0 | 396.0 | 208 | 8.9 / 7.6 |

The instrument asserts the plan's three tripwires. The first (width 32 above
1.5 times width 16) fired by 1% at fill 2048 (299.0 against 295.9) and
cleared by 9% at fill 20480. The other two did not fire. The test fails
at the default fill depth on that one condition and passes at the
serve-realistic depth. Ruling (driver): the shipped ladder `[3, 8, 15, 31]`
is kept. The operator accepted the full ladder at plan time, the miss is at
the boundary and reverses at the depth this fork targets, lookup ships off,
and the plan's action (drop rung 31) needs another worker rebuild. This is
flagged for the operator.

Serve A/B, one serve process per arm in one session, `--mtp-depth 8`,
decode tokens per second:

| prompt | off | on | on, thresholds 3,6,10,16 |
|---|---|---|---|
| public fixture (copy task, 192 out) | 21.7 | 84.1 | 65.3 |
| repeat-heavy Swift file (1024 out) | 24.8 | 65.2 | 50.5 |
| zebra (17 out) | 13.4 | 20.3 | 22.7 |
| raccoon (61 out) | 26.6 | 25.3 | 24.7 |

The emitted text is byte-identical to lookup-off on every prompt in both
arms. Lookup rounds made up 75 of 101 rounds in the default-threshold arm,
with an effective depth of 22.3 on the public prompt. On the two
jcode-style prompts accept fell (zebra 0.400 to 0.292, raccoon 0.742 to
0.682) while decode moved in opposite directions. Raccoon lost 5%. The
17-token zebra reply gained 51%. Lookup is a 2.6x to 3.9x decode win on
copy-heavy input. On prose it is within noise to slightly negative.
The raised threshold loses on three of four prompts and wins on the
17-token zebra reply, which discharges the marginal-gate obligation in favour
of the shipped thresholds. Default stays off. Turn it on per workload.

A side result narrows an open ledger item. Worker stderr never reached the
serve log under `sandbox-exec`. With the parent run under
`MLXFAST_NO_SANDBOX=1`, forwarded lines appear (80 and 81 in the lookup
arms, 494 in the ANE arm), while four arms still forwarded nothing and no
sandboxed control ran today. The sandbox is the leading cause, not a proven
one.

## 506: MTP head chain cost

Landed and merged: the head priming cap and the proposal-only head
quantization, both off. The serve A/B the earlier note called pending had
already run on 2026-08-29 (`head-levers-result.md`, build `d4d3520`, depth
forced to 2): 8-bit quantization did not flip one proposal in 63 rounds,
4-bit cost accept, and every cap from 64 to 8192 left accept unchanged at
22.8k tokens with the capped arms at most 4% slower. Its recommendation,
ship neither, stands.

Measured today:

- Head step bench (random weights of the pinned geometry, 2048 history rows):
  bf16 2.95 ms per step. At 8 bits the step is 2.33 ms and the head eval
  reads 1.57 ms over 558 MB. At 4 bits the step is 2.07 ms and the head
  eval 1.21 ms over 398 MB. The saving is under 1 ms per draft step against
  a width-1 round of about 80 ms at 512 tokens.
- Serve A/B: 8-bit reproduces the baseline rounds, drafts and accept on all
  three prompts with identical text. 4-bit moves one draft on public and
  costs 2.6 accept points and two rounds on raccoon. The plan's pass rule
  (identity plus accept within 2 points) holds for 8-bit and fails for
  4-bit, with the serve text standing in for the harness identity line.
- Priming cap 128 on a 34,715-token prompt, two turns: identical rounds,
  accept and text. Decode read 14.4 and 14.6 tok/s against 14.9 and 14.8.
  The cold prefills were 294.7 and 289.1 s, the warm resumes 0.15 and
  0.16 s. The cap is a pure memory bound.
- The mandatory re-fit of `headStepCostRatio` (head-chain plan Task 6):
  forced depths 1 to 4 on the README prompt (8,690-token seed, 512 decoded
  tokens) fit `wall = P + rounds * (V + d * H)` with P = 54.2 s,
  V = 113.4 ms and H = 23.0 ms, so h = 0.203 with a one-sigma band of
  0.037. All five emitted streams were byte-identical and the live schedule
  landed within 1% of the best forced arm. The component side was not
  re-run today. The constant stays 0.18, inside the band, and its comment
  now carries the derivation and the corrected readout claim (`dd8b852`).
  The residual that the first quantized forward mutates state without
  synchronisation is carried forward.

The full debug-build `swift test` run started for this bead did not complete.
The `CPUColumnSplitQuantizedMMTests.realDownShapeSplitMatchesHalves` case ran
more than 60 minutes CPU-bound in the debug build and the run was stopped at
20:26. The release-bundle runs of the head-chain and lookup suites stand as
the test evidence for this bead.

## 2i7: Prefill compute

The prefill question is answered in `docs/compute-engine-map.md`, which
supersedes the older ledger reading. The forward is about 85% GEMM at the
chained rate of 7.36 TFLOPS. The lead item is the 1.83x dilution between a
solitary GEMM and a chained one, caused by dependency serialization and
weight-cache residency, not by shape switching. Both residency hypotheses
are dead, and the BM tile ladder peaks at 64, which shipped.

Lever 1 (N-split scheduling) is closed by measurement, because there is no
cache knee to schedule around. Lever 2 (CPU column assist) was implemented
and then measured dead as implemented. The CPU matrix unit reaches
2.48 TFLOPS, but the 4-bit to fp32 dequant blocks it. Neither lever ships.
The column-split code stays behind an unreachable knob pending a removal
decision.

The second engine that replaced them is the ANE channel-split hybrid. Its
generation collapse was fixed this morning. Measuring it this afternoon found
two more defects, both fixed:

- Each program's staging directory under the temp folder was never removed
  while a worker held its 64 programs. 886 directories (87 GB) filled the
  volume, later builds failed with ENOSPC, and the hybrid fell back to the
  GPU silently. The sweep's hybrid legs mismatched their own goldens for that
  reason. `ae2616b` removes the staging files once a program is loaded.
- A random per-program tag made every process's programs new to the ANE
  daemon's root-owned cache, about 6 GB per hybrid process. `392411b` derives
  the tag from the weight-blob hash. On the golden path the second run then
  hit the cache (1 MB of growth against 10.2 GB for the first). In serve the
  same prompt still grew the cache by 10 GB and paid a 54 s seed prefill, and
  that difference is unresolved.

Two more limits were observed. A process cannot hold about 128 programs:
loads fail with `0x50004` after the second shape. And serve compiles a fresh
set of 64 programs for every distinct chunk length (512, 527, 605 and 639
were seen); the public prompt alone built 64 programs at 512 and 34 at 527
inside one 54 s seed prefill.

Measured on the fixed build, the six layers of this morning's in-situ table
reproduce exactly. The bf16 README golden reproduces the morning golden
across processes. The full-layer picture is wider than the six-layer table:
of 128 verify rows over the two passes, 117 exceed 1% of their output scale
and 16 exceed 5%. The worst is 14% on layer 40 (0.16 of a 1.19 output
scale). The large relative errors sit on the layers with the smallest
outputs. In serve the three chat prompts emit byte-identical text
to all-GPU with decode within noise (public 21.0 against 21.7 tok/s at 74
against 72 rounds). Seed prefill read 38 to 43 s per prompt against 4.1 to
4.8 s all-GPU, each prefill spanning one or two shape compiles.

Not measured on the fixed build: the hybrid rows of the accept matrix and
the gated prefill pair. The wrapper sweep aborted on the full disk, the
direct `mtp-timed` matrix cannot consume raw goldens, and with 17 GB free
after the serve cache miss no further hybrid process is safe on this machine.
The all-GPU README rows on today's tree reproduce the 2026-09-01 rows exactly
(accept 0.540 at depth 8 and 0.619 at depth 2), which is evidence the GPU
path is unchanged since that matrix.

Finding handed forward: the hybrid is not serve-viable as built. It needs a
fixed-shape prefill chunking policy so programs compile once per shape,
per-shape eviction under the resident-program limit, the daemon cache hit
confirmed for the serve worker, and the operator's root-owned ANE caches
cleared with sudo before any further hybrid run. The 70 GB of Aug 29 Core ML
bundles under `~/Library/Caches/org.python.python` are the operator's to
clear as well.

## mbn: Ship

One rebuild of both products at the final tree, then two worker rebuilds for
the ANE fixes. The rebuild invalidated the prefill checkpoint cache once.
Serve A/B on the three frozen chat prompts, `--mtp-depth 8`:

| prompt | decode tok/s | accept | rounds | seed prefill s |
|---|---|---|---|---|
| public | 21.7 | 0.556 | 72 | 4.09 |
| zebra | 13.4 | 0.400 | 9 | 4.82 |
| raccoon | 26.6 | 0.742 | 16 | 4.65 |

The warm resume on a 34,715-token prompt read 0.16 s. The 2026-08-28 record
read 21.3, 15.5 and 26.7 tok/s at the same accept rates and round counts:
public 2% higher, raccoon within 1%, zebra 13.5% lower on a 17-token reply. Two arms with
identical proposals to base (quant 8 and depth 15) read 22.3 and 21.4 on
public, 16.2 and 15.0 on zebra, and 27.9 and 27.0 on raccoon, so the spread
on these short replies is about 10%. Every arm emitted text
byte-identical to this baseline. The QoS arm was skipped because Task 2
produced no win. Lookup rounds are 0% of rounds at defaults and 74% with
lookup on. The turn usage exposes one aggregate accept rate, not a per-source
split. Defaults after this ship are unchanged: everything off, with lookup
the one lever worth turning on per workload.

## Host-synchronization plan

Task 2 (host QoS) applied in every worker and did not move the round: 80.8 ms
against 81.9 ms for the same-session arm A and 79.4 ms for Task 1, with no
repeated configuration to size the spread. Task 3 (five command-buffer and
ladder arms plus arm A) produced no winner: every arm sits between 81.1 and
82.8 ms, arm A included. The ladder arms move time between the build and eval
buckets without moving their sum, so the host is not on the critical path at
width 1. No env preset was written. Task 4 did not fire. The plan closes. The
next decode lever is the GPU work inside the width-1 step.

