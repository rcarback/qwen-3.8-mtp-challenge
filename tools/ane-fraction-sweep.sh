#!/usr/bin/env bash
# Task E: sweep ANEFusedSplitMLP concurrent-vs-all-GPU speedup over ANE
# fractions, ONE fraction per process (position inside a process moves GPU
# timings by more than the effect). Local-fork-only; never part of a ranked
# submission. Run only on a confirmed-idle machine.
set -euo pipefail

FRACTIONS=("${@:-0.0 0.0625 0.125 0.1875 0.25 0.375 0.5}")
# Allow a single space-separated arg or multiple args.
if [[ $# -eq 1 ]]; then read -r -a FRACTIONS <<< "$1"; fi

SEQ_LEN="${MLXFAST_SEQ_LEN:-512}"
ITERS="${MLXFAST_TIMING_ITERS:-9}"

echo "ANE fraction sweep: S=${SEQ_LEN} iters=${ITERS} fractions=${FRACTIONS[*]}"
for f in "${FRACTIONS[@]}"; do
  echo "=== fraction ${f} ==="
  MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
  MLXFAST_ANE_FRACTION="${f}" \
  MLXFAST_SEQ_LEN="${SEQ_LEN}" \
  MLXFAST_TIMING_ITERS="${ITERS}" \
    swift test --force-resolved-versions \
      --filter 'ANEFusedSplitSpeedTests' 2>&1 | grep -E 'ANE-SPEED|error:|Fatal|fail' || true
done
