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
| 3 | A derived 8-bit copy of the MTP head serves proposals; `MLX_QWEN_MTP_HEAD_QUANT=0` disables it, `4` selects 4 bits | `Qwen35.swift`, `Qwen35MTP.swift` | on at 8 bits | head only, the target decides every emitted token |

The branch after the port, oldest first:

| commit | subject | shipped files |
| --- | --- | --- |
| `3cf2b861` | perf(qwen35): packed gated-delta prework at widths 1 and 2 | `Qwen35.swift` |
| `aad40971` | perf(mtp): evaluate cache roots instead of trimmed slices | `Qwen36MTPBlockSession.swift` |
| `308b786c` | perf(mtp): quantize the MTP head for proposals at 8 bits by default | `Qwen35.swift`, `Qwen35MTP.swift` |
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
