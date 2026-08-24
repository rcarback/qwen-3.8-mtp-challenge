# DFlash2 head port plan

Replace the pinned MTP head with a Swift port of `z-lab/Qwen3.8-27B-DFlash2`,
declared through `mtp-head.manifest.json`. The goal is a higher accept rate,
which is the quantity the published score depends on.

Status: DONE, and the answer is no. The head is ported, reference-exact, wired
end to end, and measured. It ties the organizer head and loses to the head that
actually ships by 15.6%, so it is not worth swapping in. The wiring stays in the
tree as unused capability. Read "Against the head that actually ships" before
reopening this.

## Measured result

The upstream Python MLX path, run against our transformed 27B tree on two
prose prompts, 160 decode tokens each, after a discarded warm-up.

| Configuration | Decode tok/s | Against serial |
|---|---|---|
| Serial, no drafter | 11.25 | 1.00 |
| DFlash2 4-bit, block 2 | 13.65 | 1.21 |
| DFlash2 4-bit, block 4 | 15.88 | 1.41 |
| DFlash2 4-bit, block 8 | 12.00 | 1.07 |
| DFlash2 BF16, block 4 | 13.77 | 1.22 |

Accepted drafts per round at block 8 hold near 2.7 at every precision from
BF16 down to 4-bit. Head precision does not change the accept rate.

Head precision is therefore pure cost on this machine. At block 4 the decode
rate falls from 15.88 to 13.77 tok/s as the head grows from 1.01 GiB to 3.58
GiB, and the accept rate does not move. Use affine 4-bit group-64.

Block 4 beats block 8. Accept saturates near 2.8 while the verify cost grows
with block width. Re-measure block size after the port: our Swift target
forward is faster than the Python one, which makes the verify cheaper relative
to the drafter and may move the optimum back toward 8.

Sample size is two prompts. Trust the ranking between configurations, which
holds across four independent precision runs. Do not trust the absolute
figures.

## Parity against the reference

`tools/dflash2/dump_dflash2_fixture.py` runs the reference drafter on
deterministic inputs and saves every input, every per-layer activation and the
proposal. `Tests/MLXFastTests/Model/Qwen38DFlash2ParityTests.swift` runs the
Swift port on the same inputs. One round of 8 rows over 5 context rows:

| Head precision | Worst layer delta | Final hidden | Drafted path |
|---|---|---|---|
| bfloat16 | 1.0 ULP | 1.5 ULP | identical |
| affine 8-bit group-64 | 1.0 ULP | 2.1 ULP | identical |
| affine 4-bit group-64 | 8.1 ULP | 39.6 ULP | diverges at row 1 |

ULP means bfloat16 units in the last place at the reference's own magnitude. The
residual stream runs near 1e6 before the final norm and near 20 after it, so one
ULP is 4096 in one place and 0.125 in the other.

The port is correct. The 4-bit row is a property of the vendored affine
quantized matmul, not of the port: the divergence scales with quantization
coarseness, which a structural error would not do, and all three runs exercise
the same modules on the same inputs.

The 4-bit row does carry a planning consequence. Emitted tokens are safe, since
the target verifies every row, but the 1.41 figure was measured on the Python
path at 4-bit and the Swift path proposes different tokens there. Measure both
precisions on the Swift path before choosing the artifact. The 8-bit head is
1.90 GiB against a 2 GiB cap, so it fits, but it costs 0.89 GiB more of weight
traffic per round.

The port also gained one behaviour the first draft missed. The reference session
calls `propose` with `logits_start=1`: block position 0 is the anchor the target
already committed, so drafting it wastes a row and shifts the proposal by one.

## Accept rate on real prose

Measured on this box through `mtp-verify`, one 106-token open-ended prose
prompt, 192 reference rows, greedy. Every run matched the serial trajectory
exactly. `mean` is committed tokens per round, which is the quantity a round's
cost is paid against.

| Head | Offer 2 | Offer 4 | Offer 8 |
|---|---|---|---|
| Pinned native | 1.98 | 2.83 | 3.20 |
| DFlash2 8-bit | 1.90 | 2.66 | 3.00 |
| DFlash2 4-bit | 1.87 | 2.85 | not run |

Per-draft accept rate at offer 4: pinned 61.6%, DFlash2 8-bit 65.6%, DFlash2
4-bit 59.9%. The 8-bit head proposes fewer drafts for the same committed
tokens, which is a real difference, but it is not more committed tokens.

DFlash2 does NOT beat the pinned head on accept rate here. Two readings, and
they are not exclusive.

- The Python 1.41 figure compares DFlash2 against PLAIN SERIAL decode, not
  against this repository's native head. That head runs a persistent
  committed-history KV cache, which the session's own notes measure at 0.903
  accept with history against 0.262 without. Beating serial is not the bar.
- 4-bit is measurably worse than 8-bit, exactly as the parity result predicted:
  the Swift 4-bit path proposes different tokens from the reference. Ship 8-bit
  if this head ships at all.

WHAT THIS DOES NOT SETTLE. Accept rate is half the round's economics. The
autoregressive head costs `V + d*H` and the block head costs `V + H`, flat in
depth, so at equal accept the block head still wins on TIME at depth. That is
the whole thesis and accept rate cannot test it. Settle it with a timed run.

One consequence to handle first: `draftPolicy` is a cost model built for the
autoregressive head. It prices depth `d` at `d` head forwards, which a block
head does not pay, so it systematically under-drafts one. Give the block path a
depth-flat policy before reading any timing.

## Timing

Forced-depth sweep, 128 decode tokens on the same prose prompt, median round
wall (`p50_block_request_seconds_after_first`) normalised by each head's own
depth-0 round. Every configuration matched the serial trajectory exactly.

| d | Native R/R0 | Block R/R0 | Native s/token | Block s/token |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.1129 | 0.1215 |
| 1 | 1.261 | 1.249 | 0.0824 | 0.0892 |
| 2 | 1.622 | 1.290 | 0.0799 | 0.0671 |
| 3 | 1.522 | 1.416 | 0.0613 | 0.0598 |
| 4 | 1.724 | 1.564 | 0.0622 | 0.0586 |
| 6 | 2.313 | 1.957 | 0.0684 | 0.0628 |
| 8 | 2.884 | 2.489 | 0.0759 | 0.0717 |

That single sweep put the block head ahead at every depth from 2 up. It did
NOT replicate. Three runs at each head's own optimum:

| Run | Native d=3 | Block d=4 |
|---|---|---|
| 1 | 0.0636 | 0.0614 |
| 2 | 0.0565 | 0.0613 |
| 3 | 0.0623 | 0.0663 |
| median | 0.0623 | 0.0614 |
| mean | 0.0608 | 0.0630 |

Medians favour the block head by 1.5%, means favour the native head by 3.5%,
and the ranges overlap completely. The two heads are indistinguishable here.

READ THESE AS HOT-START NUMBERS. This host idles at 46.9C against the cool
gate's 40C target, so the gate cannot arm and every reading above is ungated.
Within-sweep ratios survive that; a 1.5% difference does not.

## Against the head that actually ships

The comparison above used the ORGANIZER head. The declared head in
`mtp-head.manifest.json` is a different artifact -- `q2-q4-rerank`, 428 MB
against the organizer head's 810 MB -- and it is the one a submission is
measured on. Digest and byte count verified against the manifest before the
run.

Accept rate first. The declared head proposes IDENTICALLY to the organizer
head at offers 2 and 4 (109 accepted / 55 rejected over 83 rounds, then
122 / 76 over 70), and differs only at offer 8 (124 / 83 over 68 against
126 / 85 over 66). That is the rerank index working as designed: a coarse
shortlist with an exact reranker reproduces the full-precision argmax, so
identical proposals are the goal rather than a coincidence. The offer-8
divergence is what confirms the two artifacts really are different.

Its advantage is therefore entirely round cost, and it is large. Median round
wall at low depth, where this host's readings are still monotonic:

| d | Declared | Organizer | Block |
|---|---|---|---|
| 0 | 0.0980 | 0.1036 | 0.1112 |
| 1 | 0.1061 | 0.1306 | 0.1389 |
| 2 | 0.1184 | 0.1680 | 0.1434 |

The first draft step costs 0.008 s against the organizer head's 0.027 s. The
depth-0 round is cheapest too, because 428 MB resident costs less than 810 MB
or 1.9 GiB.

Head to head at the live policy, four interleaved runs, 128 decode tokens:

| Run | Declared | Block 8-bit |
|---|---|---|
| 1 | 0.0543 | 0.0631 |
| 2 | 0.0571 | 0.0595 |
| 3 | 0.0597 | 0.0670 |
| 4 | 0.0530 | 0.0657 |
| median | 0.0557 | 0.0644 |

Four paired wins out of four, a 15.6% median gap, and the declared head wins
while drafting LESS (3.94 committed tokens per round against 4.76). This is
far larger than the noise that swamped every earlier comparison.

WHY IT WINS, and why the block drafter cannot answer it. DFlash2 reduces the
NUMBER of head forwards. The declared head instead makes each one about three
times cheaper, by not streaming the 248320-row vocabulary matrix on every
draft, and gives up nothing on proposals to do it. Fewer expensive head steps
lose to the same number of cheap ones.

## Decision

Keep the declared head. Do not declare the block drafter.

Fidelity ranks the three candidates cleanly, and it is the only axis that
separates them once timing is a wash:

1. The organizer-pinned native head IS the reference. No substitution, no
   port, no re-quantization, nothing to verify. The declared head matches its
   proposals exactly at the depths a round actually uses.
2. DFlash2 8-bit is a reference-exact port: 1.0 ULP per layer and an identical
   drafted path. Faithful, but a substituted architecture behind a 1.9 GiB
   artifact and a digest declaration.
3. DFlash2 4-bit is the only configuration with MEASURED infidelity, and it is
   the one this plan intended to ship. Rule it out.

Head fidelity is not a correctness argument. Every configuration emitted tokens
identical to the serial trajectory, because the head only proposes and the
target decides. Fidelity buys accept-rate faithfulness and nothing else, which
is exactly the channel the 4-bit head's divergence showed up in.

WHAT WOULD REOPEN THIS. Nothing on the timing side. The 1.5% between the block
head and the ORGANIZER head needs the ranked box to settle, but that question
stopped mattering once the declared head beat the block head by 15.6% with four
paired wins out of four. A block drafter would have to close that gap, not the
1.5% one.

## Why this target

The drafter is built for the exact pinned backbone. Its `config.json` declares
`num_target_layers: 64`, `vocab_size: 248320`, `hidden_size: 5120`, and
`intermediate_size: 17408`. Every one of those matches the ranked target. Its
`block_size` is 8, which equals the trusted maximum draft depth.

An unmodified tree scores about 0.994. A tree that never drafts scores 1.0.
The shipped speculative machinery therefore costs more than it earns today, so
accept rate is the binding constraint. Kernel work does not move it.

## What the contract allows

`benchmark.json` lists 89 editable paths. Four matter here.

| Path | Role in this plan |
|------|-------------------|
| `mtp-head.manifest.json` | Declares the head artifact by source, digest and byte count |
| `mtp-head/` | Holds in-branch head weights, exempt from the source byte budget |
| `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift` | The vendored MTP head module |
| `Sources/MLXFastModel/` | The block session, the target glue, and new head code |

`docs/qwen-mtp-editable-surface.md` states the rule that permits the
substitution: the head only proposes, and the target decides every emitted
token. The trusted parent re-checks the whole stream after the clock stops.

## Artifact size: resolved

The published drafter is 3.58 GiB of BF16 across 81 tensors, which exceeds the
2 GiB manifest cap. Every rung of quantization fits, because 1.924 G of its
1.925 G parameters are quantizable.

| Precision | Size |
|---|---|
| BF16 | 3.58 GiB |
| 8-bit group-64 | 1.90 GiB |
| 6-bit group-64 | 1.46 GiB |
| 4-bit group-64 | 1.01 GiB |

Ship affine 4-bit group-64 at 1.01 GiB. The measurements above show it is the
fastest option and loses no accept rate, so the cap does not constrain the
design. The previous pinned head shipped in the same format
(`mlx-community/Qwen3.6-27B-MTP-4bit`).

The head weights are inside the editable surface on this track, so quantizing
our own head is permitted. The DFlash-track rule that forbids re-quantizing the
drafter governs a different track and does not apply here.

## Architecture gap

The two heads differ in three ways that decide the work.

| | Pinned MTP head | DFlash2 |
|---|---|---|
| Depth | 1 layer | 5 sliding-attention layers |
| Conditioning | Final hidden state | Hidden states from target layers 5, 19, 33, 47, 61 |
| Proposal | One token per forward, autoregressive | Whole block in one forward, then a selector traces a path |

The third row is the expensive one. `Qwen36MTPBlockSession` drives the head one
step at a time through `mtpForwardWithHidden`. DFlash2 produces the whole block
in a single forward, so the round loop needs a second shape.

The second row is the other cost. `callWithHidden` returns only the final
hidden state. The target forward must also publish five intermediate layer
outputs, concatenated into the 25600-wide input that `fc` consumes.

## Reference implementation

Upstream ships `dflash/model_mlx.py`, a complete MLX implementation in 907
lines. It defines every module the port needs and removes most of the design
risk. Port from it directly rather than from the paper.

Four modules carry the DFlash2 delta over DFlash v1:

- `GroupedDynamicCausalConv` — a two-tap causal convolution whose kernel is
  produced per position by a linear projection, added to a static base kernel.
  Group size 16, kernel size 2.
- `DFlash2DecoderLayer` — wraps both the attention block and the MLP block in
  one of those convolutions.
- `CandidateSelector` — takes the top 16 logits per position, scores each
  candidate with a rank-256 bilinear edge term between a predecessor codebook,
  a projected hidden state, and a successor codebook, then walks the block
  greedily.
- `DFlash2DraftModel` — binds the target embedding table and `lm_head`, builds
  the layer stack, and exposes `propose`.

## Plan

### Done

1. Measured the accept rate and decode speed against the transformed 27B tree
   across four head precisions and four block sizes. Result above: 1.41 over
   serial at 4-bit, block 4.
2. Ported the four modules to Swift as `Sources/MLXFastModel/Qwen38DFlash2Head.swift`
   and checked the port against the reference. Result below.
3. Published the five target hidden states from the Qwen 3.8 forward. The
   capture is off unless a caller asks for it: `callWithHiddenNormedAndLayers`
   with an empty `layerIDs` is the ordinary forward and adds no operation.
4. Added a block-parallel round shape to `Qwen36MTPBlockSession`
   (`installBlockDrafter`). The autoregressive shape stays, because the pinned
   head still uses it.

### Now

5. Wired the drafter end to end. A head tree whose config declares
   `DFlash2DraftModel` is loaded beside the backbone instead of merged into it,
   and the session installs it. Verified against the 27B tree: exact tokens at
   offers 2, 4 and 8.
6. Measured accept rate against the pinned head on prose. Result above.

7. Replaced the block path's draft price. The shipped cost model charged a
   block head for `d` head forwards it never performs. The fitted replacement
   puts the drafter forward on the first step and leaves extra drafts cheap.
8. Timed both heads. Result above: a wash.

### Now

Nothing. The question this plan asked is answered.

### If this is reopened

9. Re-time on the ranked box behind the thermal gate. That is the only
   measurement that can separate the two heads.
10. Refit `blockDraftForwardCostRatio` and `blockDraftRowCostRatio` there. The
    shipped values are this host's fit scaled into the native constant's unit,
    which is a transfer argument, not a measurement.

## What the round shape turned out to be

The block-parallel round is structurally cheaper than the autoregressive one,
beyond the flat cost model. The drafter caches only the injected target
context. The proposal block's own keys and values are concatenated for one
attention call and then dropped, so every row that reaches the drafter cache is
already committed. A rejected draft leaves nothing behind, and the round needs
no draft-cache rollback at all. The pinned head needs one every round.

Two consequences for the measurement to come:

- The reference block size counts the anchor row. `draftPolicy` returning `d`
  proposes `d` tokens from a `d + 1` row block, so the Python optimum of block
  4 is `d = 3` here.
- `Qwen36MTPLimits.maxDepth` is 8, which makes a 9-row block reachable. The
  drafter declares `block_size` 8 and trained at 16. Nothing in the port fixes
  the width, but 9 rows is untested.

| Risk | Effect if it holds |
|------|--------------------|
| Publishing five hidden states slows the target forward | The numerator gets slower, which the accept rate must repay. This is the main remaining risk. The capture costs five tensor adds per forward on the boundary-fused path, plus one concatenation |
| Block-parallel drafting changes round accounting | The trusted parent reads effective depth from its own journal, so the ledger must still close |
| The 1.41 figure comes from two prompts on the Python path | The Swift result may differ. Re-measure on the eight-prompt shape after the port |

Two risks from the first draft of this plan are now closed. Accept rate does
survive 4-bit, and the artifact fits the cap with room to spare.
