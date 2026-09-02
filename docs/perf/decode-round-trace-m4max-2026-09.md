# Decode round trace, M4 Max, 2026-09-02

## Run conditions

- Date: 2026-09-02 (run started ~01:01 local / 05:01 UTC).
- Machine: local M4 Max worktree (`/Users/carback1/Code/llms/qwen-host-sync`, branch
  `local/host-sync-2026-09`, HEAD `db862a3`).
- Command: the brief's Step 10 block verbatim, run from this worktree's root:

  ```bash
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="$PWD/.plans/trace/round-trace-2026-09-02.log"
  export MLXFAST_NO_SANDBOX=1
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=64
  rm -f "$MLX_QWEN_MTP_TRACE_PATH"
  macmon pipe -i 500 > "$MLX_QWEN_MTP_TRACE_PATH.macmon.jsonl" &
  MACMON_PID=$!
  ./benchmark-qwen-mtp.sh --local-iterate
  kill "$MACMON_PID"
  tools/mtp-round-trace-summary.sh "$MLX_QWEN_MTP_TRACE_PATH"
  grep '^mtp-cache:' "$MLX_QWEN_MTP_TRACE_PATH"
  jq -r 'select(.gpu_usage[1] > 0.5) | .gpu_usage[0]' "$MLX_QWEN_MTP_TRACE_PATH.macmon.jsonl" \
    | sort -n | awk '{ a[NR] = $1 } END { print "gpu_mhz_busy_lower_median", a[int((NR + 1) / 2)], "samples", NR }'
  ```

  Ran on the first attempt after prerequisites were re-checked (no `mlxfast` process,
  GPU usage ~0.6-0.7%); no gate abort, no retry needed.
- GPU temperature at gate release: the three thermal gates behaved differently.
  - Before reference-row generation: waited 30 s, released at a plausible 40.0C.
  - Before the true serial control leg (depth=0, the leg this report's bucket table
    is built from): the sensor read 1.6C, at or below `benchmark.sh`'s own 5C
    plausibility floor. `benchmark.sh` printed its own warning
    (`the GPU temperature reads 1.6C ... the thermal gate may be ineffective and
    gated timings may effectively be ungated`) and released the gate after 10 s
    rather than aborting -- this is `benchmark.sh`'s documented behavior for an
    implausible reading, not a bug in this run.
  - Before the native-MTP (depth=8) leg: same implausible 1.6C reading, gate
    released immediately (0 s wait).
  - **Caveat:** the depth-0 leg analyzed below ran behind a thermal gate that
    warned it could not confirm the GPU was actually cool. The measured numbers
    are recorded as-is; treat the absolute magnitudes with that caveat in mind
    if a rerun later shows a materially different `parent round (measured)`.
- `mtp-cache:` line (printed identically for both sessions in this run):
  ```
  mtp-cache: layers=64 first4=MambaCache,MambaCache,MambaCache,KVCacheSimple kv_policy=bf16 ladder=default compiled_decode=true
  ```
  This matches the `MambaCache`/`KVCacheSimple` stack and default ladder the
  42.6 ms in-process reference (`QwenPhaseBreakdownTests.swift:69-80`) and the
  2,294 us verify-graph reference (`Qwen36MTPBlockSession.swift:1871-1874`) were
  both measured under, so the comparisons below are against a matching
  configuration.
- GPU MHz busy lower median (samples with `gpu_usage[1] > 0.5`, i.e. under load,
  across the whole run including both legs): `858 MHz`, 83 samples.
- Trace files: recorded live at `.plans/trace/round-trace-2026-09-02.log` (364
  lines: 74 `mtp-parent:`, 74 `mtp-worker:`, 64 `mtp-trace0:`, 2 `mtp-cache:`)
  and `.plans/trace/round-trace-2026-09-02.log.macmon.jsonl`; both are
  committed verbatim as the raw record this doc's numbers trace to, at
  `docs/perf/raw-round-trace-2026-09-02.log` (51,684 bytes) and
  `docs/perf/raw-round-trace-2026-09-02.macmon.jsonl` (148,127 bytes).
  `bash tools/mtp-round-trace-summary.sh docs/perf/raw-round-trace-2026-09-02.log`
  reproduces the summary table below exactly.
- `all_tokens_matched` read `true` in both legs' reports (serial control:
  `rounds=64 accepted_draft_rate=0.0000 all_tokens_matched=true
  reference_checked_rows=64/64`; native-MTP: `rounds=10 accepted_draft_rate=1.0000
  all_tokens_matched=true reference_checked_rows=64/64`).
- Local estimated decode speedup for this run (informational only, not a ranked
  score): `1.5936` (serial 0.131461 s/token vs MTP 0.082493 s/token).

## Summary table (depth-0 rounds, `tools/mtp-round-trace-summary.sh`)

```
rounds: parent=64 worker=64 session=64

bucket (lower median over depth-0 rounds)           us   share
parent encode + pipe write                          63    0.1%
transport + scheduling (derived)                    72    0.1%
worker JSON decode                                  67    0.1%
worker outside the session (derived)                57    0.1%
session graph build                              20153   25.4%
session eval wall                                59081   74.4%
session readout                                    172    0.2%
worker JSON encode                                 104    0.1%
worker pipe write                                   16    0.0%
parent JSON decode                                 183    0.2%
parent round (measured)                          79424  100.0%
parent gap between rounds (outside the round)        52    0.1%

parent closure: round - (encode_write + wait + decode)        14    0.0%
session build+eval+readout                       79406  100.0%
forward excess over in-process 42.6 ms           36806   46.3%
session host thread cpu (us, from cpu_ns)        21297   26.8%
```

All 64 depth-0 rounds are measured on every side (`rounds: parent=64 worker=64
session=64`); the script exited 0, so no derived bucket was negative.

## Exit criterion

- Parent closure within 10% of the parent round: 14 us of 79424 us (PASS).
- No derived bucket negative (the script exits 1 otherwise): PASS (script
  exit code 0; `transport=72`, `worker_outside=57`, `closure=14`, all
  non-negative).
- Every bucket measured on at least 60 of the 64 depth-0 rounds: PASS (64/64
  measured on the parent, worker, and session sides).
- Session build + eval wall + readout vs in-process forward + lm_head
  (42.6 ms, timed around build and eval, `QwenPhaseBreakdownTests.swift:69-80`):
  79406 us, forward excess 36806 us.
- Session graph build: 20153 us (in-process reference for the 64-layer verify
  graph: 2,294 us on an M4 Pro, `Qwen36MTPBlockSession.swift:1871-1874`).
- Outside-session total (transport + worker outside + worker
  decode/encode/write + parent encode/decode + gap): 72 + 57 + 67 + 104 + 16 +
  63 + 183 + 52 = 614 us.

Note: `forward_excess` above is measured against `reference_us=42600`, a
debug-build best-of-5 figure (every raw log under `docs/perf/raw-phase-*.txt:6`
says "Building for debugging..."), while the worker session in this run is a
release build. Some of the 36806 us excess may be build-mode overhead rather
than real host-thread cost; read the triggers below with that in mind.

## Task selection

- Task 2 (host thread QoS) runs if forward excess is at least 5,000 us or
  graph build is at least 5,000 us. Both conditions hold (36806 us and
  20153 us respectively) -- **Task 2 runs.**
- Task 3 (command-buffer and ladder arms) runs if forward excess is still at
  least 5,000 us after Task 2. Not yet evaluated -- Task 2 has not run yet.
  **Pending Task 2's outcome.**
- Task 4 (serial prefetch across the protocol gap) runs if the outside-session
  total is at least 4,000 us. 614 us is well under that threshold --
  **Task 4 does not run.**
- If the parent round (measured) is under 50,000 us, the 81.3 ms figure came
  from a condition this run did not reproduce, and the plan says to stop and
  record the differing condition instead of proceeding. Here the parent round
  is 79424 us (close to the 81.3 ms figure this plan is chasing), so this run
  reproduces the gap and the stop route is **not** taken. Corrected stop route
  for reference: if the gap had not reproduced, the plan would route to the
  headroom plan's Task 2 (the sub-layer mixer/MLP seam) before choosing
  between scope sections A and B.

The session eval wall (59081 us, 74.4% of the round) and the session graph
build (20153 us, 25.4%) together account for essentially the entire round
(99.8%); every parent/worker/transport bucket combined is under 1,000 us.
Task 2 is the next step.

## Reading

The protocol is exonerated: every parent, worker, and transport bucket
combined is about 600 us, under 1% of the round. The excess is inside the
session. The in-process width-1 reference splits the step build-heavy: at
context depth ~10k the M=1 row of the verify-width sweep reads build 36.9 ms /
eval 5.7 ms / total 42.6 ms (`docs/perf/raw-phase-sumTable.txt:90`; the same
split holds at shallower context, `docs/perf/raw-phase-sumTable.txt:58`, build
36.0 ms / eval 9.9 ms). That split is the ladder overlapping the GPU with the
host's graph build, so most of the wall time reads as "build" even though the
GPU is doing real work underneath it. In this worker trace the same step
splits eval-heavy instead: build 20.2 ms / eval 59.1 ms. The overlap the
in-process ladder relies on is not happening in the worker -- the GPU work
that used to hide under the build window is showing up as wall-clock eval
time instead. Candidates for Task 2 and Task 3: the worker host thread's
scheduling class (does the thread doing the build actually get scheduled
promptly enough to keep submitting into the ladder), and the ladder
configuration inside the worker (is the same ladder that produces the
in-process overlap actually engaged there).
