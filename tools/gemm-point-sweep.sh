#!/bin/bash
# Measures GEMM points one per process, because position in the process
# dominates the numbers.
#
# The square bf16 reference reads about 14.7 TFLOPS when its block runs first
# in a fresh process and about 6.5 when any other block ran before it. The
# effect survives Memory.clearCache() and does not survive a process boundary,
# so isolation is the only reliable control. Each point therefore pays a full
# process start, which is why this script exists rather than a loop inside one
# test.
#
# Usage: tools/gemm-point-sweep.sh <points-file> <out.tsv>
# Each line of the points file: <label> <mode> <M> <N> <K>
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POINTS=${1:?points file}
OUT=${2:?output tsv}

printf "label\tmode\tM\tN\tK\tms\ttflops\n" >"$OUT"

while read -r label mode M N K; do
  case "$label" in '' | '#'*) continue ;; esac
  # The GPU needs to settle after the previous point before the gate will
  # pass, so retry rather than abort. Each gate call itself samples for eight
  # seconds, which is the wait.
  quiet=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ./tools/host-quiet-gate.sh >/dev/null 2>&1; then
      quiet=1
      break
    fi
  done
  [ "$quiet" = "1" ] || {
    echo "host never went quiet before $label, aborting" >&2
    exit 1
  }
  line=$(MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    MLXFAST_GEMM_M="$M" MLXFAST_GEMM_N="$N" MLXFAST_GEMM_K="$K" \
    MLXFAST_GEMM_MODE="$mode" \
    swift test --force-resolved-versions --filter singleGemmPoint </dev/null 2>&1 |
    grep '^GEMMPOINT' | head -1)
  if [ -z "$line" ]; then
    echo "no measurement for $label" >&2
    exit 1
  fi
  rest=${line#GEMMPOINT	}
  printf "%s\t%s\n" "$label" "$rest" >>"$OUT"
  echo "$label $rest"
done <"$POINTS"

echo "done"
