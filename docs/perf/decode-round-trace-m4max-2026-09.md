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
debug-build best-of-5 figure. Every raw log under `docs/perf/raw-phase-*.txt:6`
says "Building for debugging...". The worker session in this run is a
release build. Some of the 36806 us excess may be build-mode overhead rather
than real host-thread cost; read the triggers below with that in mind.

## Task selection

- Task 2 (host thread QoS) runs if forward excess is at least 5,000 us or
  graph build is at least 5,000 us. Both conditions hold (36806 us and
  20153 us respectively) -- **Task 2 runs.**
- Task 3 (command-buffer and ladder arms) runs if forward excess is still at
  least 5,000 us after Task 2. After Task 2 the forward excess is 36,350 us
  (see the Task 2 section) -- **Task 3 runs.** Its result is in the Task 3
  section.
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
session.

The in-process width-1 reference splits the step build-heavy. At context
depth ~10k the M=1 row of the verify-width sweep reads build 36.9 ms / eval
5.7 ms / total 42.6 ms (`docs/perf/raw-phase-sumTable.txt:90`). The same
split holds at shallower context: build 36.0 ms / eval 9.9 ms
(`docs/perf/raw-phase-sumTable.txt:58`).

That split is the ladder overlapping the GPU with the host's graph build.
Most of the wall time reads as "build," even though the GPU is doing real
work underneath it.

In this worker trace the same step splits eval-heavy instead: build 20.2 ms /
eval 59.1 ms. The overlap the in-process ladder relies on is not happening in
the worker. The GPU work that used to hide under the build window now shows
up as wall-clock eval time instead.

Candidates for Task 2 and Task 3: the worker host thread's scheduling class,
and the ladder configuration inside the worker. For scheduling class, check
whether the thread doing the build actually gets scheduled promptly enough
to keep submitting into the ladder. For the ladder, check whether the same
ladder that produces the in-process overlap is actually engaged in the
worker.

## Task 2: `MLX_MTP_HOST_QOS=interactive` (2026-09-02, gated run)

Same command as the Step 10 block above, plus `MLX_MTP_HOST_QOS=interactive`
and the trace path `.plans/trace/round-trace-2026-09-02-qos.log`. Run from the
main tree (`local/perf-2026-08`, wall lane merged at `70eba8a`, host QoS code
uncommitted at run time) at about 16:53 local, machine idle.

Run conditions:

- Thermal gate: all three gates read plausible temperatures and released at
  39.7C, 39.8C and 39.8C after 70 s, 70 s and 60 s. The 1.6C sensor caveat on
  the Task 1 run does not apply here. The two runs agree on the parent round
  within 2%, so the caveat on the Task 1 absolute round is discharged.
- `mtp-worker: host_qos=interactive status=0` appears three times, once per
  worker process (reference worker, serial-control worker, native-MTP
  worker). The QoS class was applied in every worker.
- `mtp-cache:` line identical to the Task 1 run for both sessions.
- GPU MHz busy lower median: 851 MHz, 79 samples (Task 1: 858 MHz).
- `all_tokens_matched` read `true` in both legs (serial control: `rounds=64
  accepted_draft_rate=0.0000 all_tokens_matched=true reference_checked_rows=64/64
  seconds_per_token=0.135635`; native-MTP: `rounds=10 accepted_draft_rate=1.0000
  all_tokens_matched=true reference_checked_rows=64/64 seconds_per_token=0.080477`).
  Local estimated decode speedup 1.6854 (Task 1: 1.5936).
- Raw record: `docs/perf/raw-round-trace-2026-09-02-qos.log` (52,201 bytes,
  215 parent/worker/trace0 lines) and
  `docs/perf/raw-round-trace-2026-09-02-qos.macmon.jsonl`.
  `bash tools/mtp-round-trace-summary.sh docs/perf/raw-round-trace-2026-09-02-qos.log`
  reproduces the Task 2 column below.

Lower medians over the 64 depth-0 rounds, in microseconds. The delta is
Task 2 minus Task 1. Negative is faster.

| bucket | Task 1 (default QoS) | Task 2 (`interactive`) | delta |
|---|---|---|---|
| parent encode + pipe write | 63 | 31 | -32 |
| transport + scheduling (derived) | 72 | 55 | -17 |
| worker JSON decode | 67 | 35 | -32 |
| worker outside the session (derived) | 57 | 69 | +12 |
| session graph build | 20153 | 10565 | -9588 |
| session eval wall | 59081 | 68275 | +9194 |
| session readout | 172 | 110 | -62 |
| worker JSON encode | 104 | 63 | -41 |
| worker pipe write | 16 | 10 | -6 |
| parent JSON decode | 183 | 88 | -95 |
| parent round (measured) | 79424 | 80780 | +1356 |
| parent gap between rounds | 52 | 26 | -26 |
| parent closure | 14 | 92 | +78 |
| session build + eval + readout | 79406 | 78950 | -456 |
| forward excess over in-process 42.6 ms | 36806 | 36350 | -456 |
| session host thread cpu | 21297 | 11285 | -10012 |

Parent seconds per token, both legs, from the wrapper's reports: serial
control 0.135635 (Task 1: 0.131461), native-MTP 0.080477 (Task 1: 0.082493).
The wrapper deletes its per-leg reports on exit, so the parent round bucket
above (the lower median of the round request) stands in for
`p50RoundRequestSeconds`.

### Decision

Task 2 does not win. The parent round moved by +1,356 us (+1.7%), inside the
run-to-run spread, and no bucket outside the session moved by more than
100 us. `MLX_MTP_HOST_QOS` stays unset by default and stays unset for the
Task 3 arms.

The split inside the session did move. The host thread's CPU time and the
graph-build bucket both halved (21.3 ms to 11.3 ms of CPU), while the
eval-wall bucket grew by the same amount. The round total did not change.
The Task 3 arm A run below, same session and default QoS, reads build
6.2 ms and eval wall 75.5 ms. So the build/eval split is run
dependent, not QoS dependent. Only the sum (build + eval + readout) is
stable across runs, at 79 to 83 ms. The round is bounded by the GPU work
the session submits, not by host scheduling.

Forward excess after Task 2 is 36,350 us, above the 5,000 us trigger, so
Task 3 (command-buffer and ladder arms) runs.

## Task 3: Command-buffer and ladder arms (2026-09-02, gated runs)

Five arms, each one wrapper run with the Task 1 Step 10 environment, default
QoS (Task 2 did not win), and its own trace path. Arm A is the same-session,
same-build baseline with no extra variables, run after the five arms. Every
run reads `all_tokens_matched=true` on both legs.

Thermal gate: arm B released at 39.6C, 39.9C and 39.2C. Arms C, D, E, F and A
hit the implausible 1.6C sensor reading on every gate (the Task 1 caveat) and
released after 0 to 20 s. Treat differences under 2% between those runs as
noise.

Lower medians over the 64 depth-0 rounds, in microseconds.

| arm | variables | build | eval wall | build+eval+readout | parent round | host cpu | serial s/token | MTP s/token |
|---|---|---|---|---|---|---|---|---|
| Task 1 | (worktree, 2026-09-02 01:04) | 20153 | 59081 | 79406 | 79424 | 21297 | 0.131461 | 0.082493 |
| Task 2 | `MLX_MTP_HOST_QOS=interactive` | 10565 | 68275 | 78950 | 80780 | 11285 | 0.135635 | 0.080477 |
| A | none (same-session baseline) | 6214 | 75466 | 81716 | 81864 | 6665 | 0.135173 | 0.081435 |
| B | `MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=50` | 13940 | 66355 | 80490 | 81074 | 14835 | 0.134412 | 0.085425 |
| C | `MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=500` | 7732 | 71945 | 79749 | 81899 | 8242 | 0.134442 | 0.080992 |
| D | `MLX_QWEN_MTP_LADDER=off` | 3185 | 79506 | 82733 | 82777 | 6876 | 0.135126 | 0.081337 |
| E | `MLX_QWEN_MTP_LADDER=front` | 3231 | 78987 | 82259 | 82309 | 6671 | 0.138804 | 0.083288 |
| F | `MLX_QWEN_MTP_LADDER=dense` | 6477 | 75422 | 81936 | 82116 | 0.135665 | 0.081627 |

The `mtp-cache:` line confirms each ladder arm engaged (`ladder=off`,
`ladder=front`, `ladder=dense`); the command-buffer arms keep `ladder=default`.
Arm D's derived closure bucket reads negative 2 us. The summary script exits 1 on any
negative derived bucket. Here the value is the rounding of independently
taken lower medians, two microseconds on an 82,777 us round, not a missing
stamp. Its table is complete and is used as printed.

Raw records: `docs/perf/raw-round-trace-2026-09-02-arm-{A,B,C,D,E,F}.log`.
`bash tools/mtp-round-trace-summary.sh <file>` reproduces each row.

### Decision

No arm cleared 1,000 us. Against arm A (81,864 us), the closest arm is B at
790 us below it, under the margin and inside the sensor-caveat band. Against Task 1
(79,424 us), every arm is slower. `tools/mtp-host-env.sh` is not created.
`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER` and `MLX_QWEN_MTP_LADDER`
stay unset by default.

The ladder arms move time between the build and eval buckets (arm D:
3.2 ms build, 79.5 ms eval) without moving the sum. This is the same shape
as Task 2. The host is not on the critical path at width 1. The 36 to
40 ms "forward excess" over the in-process 42.6 ms reference is GPU
execution time the in-process reference hid under its ladder overlap, plus
whatever debug-versus-release build difference that reference carries. The
host-synchronization plan closes here. Task 4 did not fire, and neither
host lever moved the round. The next lever is the GPU work inside the
width-1 step itself.
