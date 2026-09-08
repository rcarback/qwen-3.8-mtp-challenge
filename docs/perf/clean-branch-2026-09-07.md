# A clean branch off `origin/main` for the ranked track (2026-09-07)

Status: in progress. The tables at the end fill in as the arms finish.

## What `origin/main` is

The `Layr-Labs/qwen-3.8-mtp-challenge` main branch is not an unmodified
tree. The last organizer commit is `0b071ed9` (2026-08-17, "Raise published
decode-speedup ceiling 3.0 -> 5.0.") Above it sit 51 "Validate submission" and
"Accept submission" commits, dated 2026-08-17 to 2026-08-23. Each of those commits
carries a submission's editable diff, and consecutive commits differ by small
increments, so the tip is one lineage of submissions rather than a set of
unrelated overlays. The tip, `0863b06a` (the 2026-08-23 validation of submission ec24d591), differs from the organizer commit in eight files: the custom
quantized matmul and the xsums fill fusion in `Qwen35.swift`, the cost-model
schedule in `Qwen36MTPBlockSession.swift`, the quantized kernel twins, and an
`mtp-head.manifest.json` that declares the `amal-david/qwen38-mtp-head-q2-q4-rerank-v1`
head. All eight are inside `editablePaths`.

`local/perf-2026-08` merged that tip on 2026-08-26, so its merge-base with
`origin/main` is the tip itself and every one of its 398 commits sits above
it. The clean branch is based on `0863b06a`, and "before" means the
tip as the leaderboard scored it.

## What the ranked measurement rewards

From `fixtures/qwen3_8_27b_mtp_track.json` and the ranked workflow:

- Per prompt, `raw = mean(serial depth-0 seconds/token) / mean(candidate
  seconds/token)` over a 512-token decode window. The 512-token seed prefill
  is charged inside the window on both legs. The published score is the
  median over the eight hidden prose prompts. Floor 0.90, ceiling 5.0.
- The serial denominator runs from a pinned prebuilt baseline tree
  (`MLXFAST_QWEN_MTP_BASELINE_WS`), never from the candidate workspace. A
  general forward speedup in the candidate moves the score. The
  local script measures the serial control from the candidate's own tree, so a
  local score understates a kernel win and a ranked-style estimate needs the
  base tree's serial leg over the candidate's MTP leg.
- Token fidelity is absolute: `mtp-verify` at the candidate depth must report
  every emitted token equal to the serial trajectory. A change that moves the
  target's numerics at a near-tie fails the run. The MTP head only proposes,
  so head changes cannot move an emitted token.
- Only `editablePaths` ship. New files under `Vendor/mlx-swift-lm` (the
  `ANEOffload` directory, `Qwen35KVRotation.swift`, `FusedQuantizedSDPA.swift`,
  the `Qwen4Exp` model) and every change under `Sources/MLXFastCore`,
  `MLXFastHarness`, `MLXFastCLI`, `MLXFastTrustedHarness`, the host-side MLX
  dispatch files and `MLXLMCommon/Load.swift` stay behind.

## What the branch carries, by surface

Of the branch's 202 source commits, 88 touch the ranked dense path. The
review classified every one of them against three tests: inside
`editablePaths`, compiles against the unmodified trusted tree, and leaves the
target's numerics byte-exact. Three changes pass all three and carry a
plausible gain at the ranked window. The branch `mlx-fast-submission` holds
exactly those three, in three commits, touching three shipped files. Two
test-only commits follow them.

| step | change | file | default | target numerics |
| --- | --- | --- | --- | --- |
| 1 | Gated-delta prework packed at widths 1 and 2, compiled g and beta and the fused gated post-norm at width 1, one memoized epsilon scalar | `Qwen35.swift` | always on | byte-equal, six receipt tests with negative controls |
| 2 | The three session eval barriers evaluate the cache roots (`innerState()`) instead of a per-round trimmed slice of the full-attention keys | `Qwen36MTPBlockSession.swift` | always on | no computed value changes |
| 3 | A derived quantized copy of the MTP head serves proposals; `MLX_QWEN_MTP_HEAD_QUANT=0` disables it, `8` selects 8 bits | `Qwen35.swift`, `Qwen35MTP.swift` | on at 4 bits since `ce85157b`, 8 bits before | head only, the target decides every emitted token |

The branch after the port, oldest first:

| commit | subject | shipped files |
| --- | --- | --- |
| `3cf2b861` | perf(qwen35): packed gated-delta prework at widths 1 and 2 | `Qwen35.swift` |
| `aad40971` | perf(mtp): evaluate cache roots instead of trimmed slices | `Qwen36MTPBlockSession.swift` |
| `308b786c` | perf(mtp): quantize the MTP head for proposals at 8 bits by default | `Qwen35.swift`, `Qwen35MTP.swift` |
| `ce85157b` | perf(mtp): quantize the proposal head at 4 bits by default | `Qwen35MTP.swift` |
| `f240258f` | test: drop a stale suite reference from the forward-stream receipts | none |
| `b151eb08` | test: fix the Comment conversion error blocking the test target | none |

A whole-diff review against the base tip approved the port with eight
findings and no blocking defect. Two findings changed the branch: a comment
in `Qwen35MTP.swift` gave the wrong reason why the derived head twin stays out
of the parameter walk, and a test comment cited a suite that does not exist.
One finding is the ship condition for step 3, which round 2 of the
measurement decides. The last finding was a base-tree problem: the test
target at `0863b06a` does not compile under Swift 6.3.2, so no receipt could
run until the fifth commit fixed one string concatenation in
`QwenMTPVerbTests.swift`. With that fix the seven receipt tests in
`QwenForwardStreamTests` and `QwenEvalBarrierTests` pass. `Package.resolved`
is untouched, and the three changed test files compile without warnings.

Two facts from the review reshape what "our work" means for this track:

- The base tip already carries the adaptive cost-model draft schedule. The
  session's `init` overwrites the `draftPolicy` property with
  `costModelDepth` on both trees, and the branch's only edits to that schedule
  serve the MoE tower. The local speedups of about 1.36 recorded in
  `mtp-accept-matrix-2026-09.md` are a property of the base, not of the branch.
- The branch changes no editable kernel source. Its kernel work sits in
  trusted host-side dispatch files (`quantized.cpp`, the attention dispatch,
  `Stream.swift`), which cannot ship.

## What stays behind, and the single reason for each

- Prompt-lookup and n-gram drafting: input-derived drafting, excluded by the
  track rules, and it depends on a trusted file.
- The pinned head loader does not accept the DFlash2 block drafter's
  architecture, and the branch measured it as a wash against the native head.
- Session cache store, prefill checkpoints, divergence memo, prefix
  extension: caches keyed on request input.
- Exact speculative sampling: the track decodes greedy only.
- KV-cache quantization and rotation, and the fused quantized decode
  attention: they change target numerics when active, and their files are new
  files outside the allowlists.
- The ANE split-MLP offload: prefill only, fp16 numerics, non-shippable files.
- The Qwen3.8-Flash-Next MoE tower: a different tower, dead on the dense path.
- Chunked prefill: single-chunk at a 512-token seed.
- Verify-width and depth knobs: each reproduces base behaviour with no
  environment variable set, and the trusted cap of 8 makes the wider regime
  unreachable.
- Trusted host-side kernel dispatch: outside `editablePaths`.
- The head seed-priming cap is off by default and ran 2 to 4 percent slower
  when on.
- Instrumentation and benches: no ranked-path effect.

Two exclusions are worth re-examining later, and neither belongs on this
branch today:

- The quantized-matmul row-tile selection (`d4d35202`, `34bd4fbc`) measured
  1.30 to 1.38 times at M of 12 to 16 rows. It lives in the trusted
  `quantized.cpp` dispatch, so it cannot ship as written. The ranked window
  never presents those shapes: decode and verify run at 1 to 9 rows under the
  depth cap of 8, and the seed prefill runs at 512. Whether the in-surface
  custom QMV can carry the same idea at 1 to 9 rows is an open question that
  needs its own measurement first.
- The verify-width family (`bcc8c676`, `1bc3aab2`, `997ea5bf`, `18cf25c4`,
  the `AttentionUtils.swift` half of `65f0d753`) is numerics-preserving and
  one commit is a correctness fix, but every member reproduces base behaviour
  with no environment variable set. A minimal branch leaves them out.

The ANE work from `ane-gpu-reevaluation-2026-09-06.md` does not transfer to
this branch. The split-MLP lane arms only at 128 tokens and above, so it
touches the seed prefill alone, about four seconds of a window near
forty-five seconds, and its fp16 compute moves the MLP numerics, which the
fidelity gate forbids. Its files also sit outside `editablePaths`.

## Measurement design

- Two worktrees on this box: `qwen-base.noindex` at `0863b06a` and
  `qwen-clean.noindex` on `mlx-fast-submission`, each built and transformed
  by its own scripts, both excluded from Time Machine and Spotlight.
- Goldens: the base tree generates 512-step goldens for seven prose prompts
  (cooking, geology, music, the README, the prefill research plan, the go-live
  runbook, the private benchmark security note) with `generate-golden`. The
  public long-copy fixture is the eighth prompt. The local script's drift
  tripwire teacher-forces every golden token, so a base-generated golden is
  the fidelity gate for the clean tree.
- Arms: `./benchmark-qwen-mtp.sh --local-iterate` with
  `MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512` and an offered depth of 8 (the
  ranked offer, and each tree's `draftPolicy` decides what it drafts). Arm order
  alternates per prompt. The GPU cool gate runs before every timed leg and the
  quiet gate before every arm.
- Head: both arms use the pinned MTP head. The declared q2-q4 rerank head is
  staged on the ranked box as a bare `model.safetensors`, and the local script
  requires a `config.json` beside the head, so the local pair cannot use it.
  The manifest is unchanged on the clean branch, so the declared head applies
  to both the before and the after state on the ranked box.
- Two ratios per prompt: the local ratio (each tree's own serial leg) and the
  ranked-style estimate (the base tree's serial leg over the clean tree's MTP
  leg).

## Results (measured 2026-09-07, 18:31 to 22:47)

Four rounds ran on the two worktrees, one model-holding process at a time.
Every arm decoded 512 tokens at the ranked offer of depth 8. Every arm that
produced a payload matched all tokens and passed the drift tripwire. The
receipts, the goldens, the prompts, and the scripts are under
`docs/perf/clean-branch-runs/`.

| round | arms | order |
| --- | --- | --- |
| 1 | base, clean | alternating per prompt |
| 2 | base repeat, clean with the 8-bit head off (`MLX_QWEN_MTP_HEAD_QUANT=0`) | alternating per prompt |
| 3 | clean repeat | single arm |
| 4 | clean with the head at 4 bits (`MLX_QWEN_MTP_HEAD_QUANT=4`) | single arm |
| 5 | clean rebuilt at `ce85157b`, no head variable, the 4-bit default | geology and cooking |

### Prompt pool

Seven of the nine generated goldens are valid. Two of the original seven and
one replacement failed the seed check at step 0 on both trees, with the same
token pair on each tree. The step-0 probe on the golden path shows why:

| prompt | golden token | timed-verb token | top-2 margin |
| --- | --- | --- | --- |
| prefill-plan | 7793 | 2695 | 0.125 |
| security | 2531 | 27370 | 0.000, three tokens tied |
| sailing | 34669 | 6463 | 0.125 |
| cooking (control) | 314 | 314 | 4.125 |

At logit magnitudes of 16 to 20, bf16 resolves to steps of 0.125, so the
three failing prompts are within one representable step of a tie. The timed
verb projects one hidden row through the head with the stock width-1
quantized matrix-vector kernel. Every other path takes the logits of a
512-row forward through the matrix-matrix kernel. Decode never compares
across the two kernels, because the candidate rows and the reference walk
share the width-1 path. Only the seed does. This is a base-tree property. The
clean diff leaves the seed path untouched. The dyeing passage replaced the two
excluded prompts, and sailing was screened out before any long arm ran.

### Seconds per token, MTP leg

| prompt | base r1 | base r2 | clean r1 | clean r3 | head off r2 | 4-bit r4 |
| --- | --- | --- | --- | --- | --- | --- |
| cooking | 0.0813 | 0.0813 | 0.0802 | 0.0802 | 0.0808 | 0.0806 |
| dyeing | 0.0692 (warm) | 0.0643 | void (gate) | 0.0633 | 0.0652 | 0.0638 |
| geology | 0.0451 | 0.0450 | 0.0435 | 0.0439 | 0.0455 | 0.0419 |
| music | 0.0567 | 0.0567 | 0.0553 | 0.0553 | 0.0569 | 0.0542 |
| readme | 0.0443 | 0.0451 | 0.0423 | 0.0437 | 0.0442 | 0.0425 |
| runbook | 0.0417 | 0.0417 | 0.0408 | 0.0407 | 0.0416 | 0.0407 |
| public | 0.0354 | 0.0357 | 0.0346 | 0.0346 | 0.0362 | 0.0340 |

The round 1 base arm on dyeing ran with its serial leg at 0.0944 against the
0.090 every other arm shows, so its MTP figure is the warm outlier and the
round 2 base arm is the base reading for that prompt. The serial legs of the
two trees agree within noise on every other prompt.

### Medians of the per-prompt ratio

| arm | median | prompts |
| --- | --- | --- |
| base, round 1 | 2.001 | 7 |
| base, round 2 | 2.003 | 7 |
| clean, round 1 | 2.100 | 6 |
| clean, round 3 | 2.051 | 7 |
| clean, head off | 1.967 | 7 |
| clean, 4-bit head | 2.106 | 7 |

The ranked-style estimate for clean in round 1, the base serial leg over the
clean MTP leg, has a median of 2.104. Base reproduces between rounds to 0.1
percent. Clean varies by about 2 percent between rounds, and readme and
geology carry that spread.

### What each change is worth

The head-off arm is level with base on every prompt. The two other ported
changes, the packed gated-delta prework and the cache-root eval barriers, are
within noise of base at this window. The derived head twin carries the gain.

| prompt | acceptance base / 8-bit / 4-bit | draft length base / 8-bit / 4-bit |
| --- | --- | --- |
| cooking | 0.395 / 0.394 / 0.367 | 0.64 / 0.63 / 0.65 |
| dyeing | 0.441 / 0.440 / 0.434 | 2.29 / 2.31 / 2.20 |
| geology | 0.753 / 0.753 / 0.767 | 3.60 / 3.60 / 3.69 |
| music | 0.563 / 0.559 / 0.560 | 2.60 / 2.62 / 2.71 |
| readme | 0.743 / 0.763 / 0.730 | 3.95 / 3.93 / 3.98 |
| runbook | 0.798 / 0.794 / 0.751 | 4.23 / 4.12 / 4.50 |
| public | 0.886 / 0.886 / 0.884 | 6.38 / 6.38 / 6.29 |

The 4-bit head is as fast as the 8-bit head on cooking, dyeing, readme, and
runbook, and faster on geology, music, and public. Acceptance drops by up to
five points at 4 bits, and the cheaper head step lets the adaptive schedule
draft deeper, so the cost per emitted token holds or improves. All 4-bit arms
matched tokens. The default moved to 4 bits in `ce85157b` on
`mlx-fast-submission` after this campaign, which also halves the derived
head's footprint. Round 5, on the rebuilt tree with no head variable set,
measured geology at 0.0410 and cooking at 0.0806 seconds per token with the
same acceptance and draft length as round 4 and all tokens matched.

### Measurement notes

- The GPU temperature sensor reports 1.6C whenever the GPU is fully
  power-gated, and the cool gate accepts that reading. Every arm before 21:20
  cleared its gates at once for that reason. With the display active and
  Spotlight indexing, the sensor reported the real die temperature and two
  arms voided at 42C against the 40C target. Disabling Spotlight indexing let
  the following arms cool below 40C and pass.
- The seed prologue is charged inside the local window. The ranked parent
  starts its clock after the seed-prefill response, so local ratios sit a
  little below what the box would publish.
- The Photos analysis daemon and two leaked gterm test readers were frozen
  with SIGSTOP for the campaign. The measure script resumes the analysis
  daemons on exit.

## The leader's configuration (2026-09-08)

The board entry at 273.6 percent is a ratio of 3.74 on the M5 box: 97.8 decode
tokens per second against the fixture's serial calibration of 0.0380 seconds
per token, 26.3 tokens per second. Its commit, the xsums fill fusion, is in
`0863b06a`, so the base tree of this campaign is the leader's code. The entry
was measured with the head that `mtp-head.manifest.json` declares, and the
ranked candidate leg runs that head while the baseline leg runs the pinned
head.

The declared head is `amal-david/qwen38-mtp-head-q2-q4-rerank-v1` at revision
`ae6282749a52e052496dd5300b4aa441df7301e8`, one `model.safetensors` of
427,742,600 bytes. The manifest's digest is not the file's digest. The runner
hashes the staged tree: for each file in sorted order, the line
`<sha256>  <name>`, and the SHA-256 of those lines. For this one-file tree the
file hashes to `d038fd41...` and the tree to `559b24eb...`, which matches the
manifest. The head's weights are already quantized. It has a 4-bit group-64
projection and MLP, a 2-bit compact draft vocabulary of 98,336 rows with a
4-bit rerank, and bf16 precision islands for the attention projections. Its
metadata names the format `qwen38-mtp-incumbent-q4-g64-plus-bf16-qkv-islands-v1`.

Two consequences follow for this branch:

- The derived head twin builds a quantized copy only when the head's
  projection is not already quantized. With the declared head it builds
  nothing, and the local gain of this branch, which the head-off arms
  attribute to the twin, does not apply to the ranked candidate leg. On the
  first compared prompt, base and clean with the declared head decode within
  1 percent of each other with identical acceptance.
- The local script cannot load a bare head file. A copy of the pinned head's
  `config.json` beside the declared file satisfies its check, and the worker
  loads the file the way the runner's staged tree is loaded. The local
  measurements from this point use the declared head unless a row says
  otherwise.

The decision on 2026-09-08 is to measure and ship with the declared head. The
4-bit default in `ce85157b` stays because it is inert with a quantized head
and helps any run on the pinned head.
