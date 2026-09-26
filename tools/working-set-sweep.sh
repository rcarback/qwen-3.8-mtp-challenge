#!/bin/bash
# Measures working-set points one per process, for the same reason
# tools/gemm-point-sweep.sh does: position in the process dominates the numbers.
#
# The square bf16 control reads about 14.7 TFLOPS when its block runs first in a
# fresh process and about 6.5 once any other block has run before it. That effect
# survives Memory.clearCache() and does not survive a process boundary, so a
# process per point is the only reliable control -- and here it matters twice
# over, because the process-state effect is itself what is being measured.
#
# Each point runs one cell of QuantizedMatmulWorkingSetTests:
#   reuse    W weight sets held = 1, used = 1   (low allocation, perfect reuse)
#   stream   W held, all W used in rotation     (high allocation, no reuse)
#   pressure W held, only index 0 used          (high allocation, perfect reuse)
#
# Data reuse moves `stream` alone. Allocation and residency pressure move
# `stream` and `pressure` together. A clock or power-state effect moves all three
# and shows in the sampled GPU frequency.
#
# Usage: tools/working-set-sweep.sh <points-file> <out.tsv>
# Each line of the points file: <shape> <arm> <width>
# Blank lines and lines starting with # are skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POINTS=${1:?points file}
OUT=${2:?output tsv}

printf "shape\tarm\twidth\tms\ttflops\theld_gb\tactive_gb\tspread_pct\tmhz_before\tmhz_after\n" >"$OUT"

while read -r shape arm width; do
  case "$shape" in '' | '#'*) continue ;; esac
  quiet=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ./tools/host-quiet-gate.sh >/dev/null 2>&1; then
      quiet=1
      break
    fi
  done
  [ "$quiet" = "1" ] || {
    echo "host never went quiet before $shape:$arm:$width, aborting" >&2
    exit 1
  }
  line=$(MLXFAST_QWEN_WS_POINT="$shape:$arm:$width" \
    swift test -c release --force-resolved-versions \
    --filter workingSetSinglePoint </dev/null 2>&1 |
    grep '^WSPOINT' | head -1)
  if [ -z "$line" ]; then
    echo "no measurement for $shape:$arm:$width" >&2
    exit 1
  fi
  printf "%s\n" "${line#WSPOINT	}" >>"$OUT"
  echo "$shape $arm $width ${line#WSPOINT	}"
done <"$POINTS"

echo "done"
