#!/usr/bin/env bash
# Summarise a local MTP round trace (MLX_QWEN_MTP_TRACE=1) for the depth-0
# leg: lower medians per field for the parent, worker and session lines, then
# the bucket table the decode-host-sync plan reads. BSD awk only; no gawk.
set -euo pipefail

trace="${1:-}"
if [ -z "$trace" ] || [ ! -r "$trace" ]; then
  echo "usage: $0 <trace-file>" >&2
  exit 2
fi

# Lines of one prefix, depth-0 rounds only, as key=value pairs. Absolute
# timestamps and identifiers are dropped; only durations remain.
extract() {
  grep "^$1 " "$trace" | grep -F 'offered=0 ' | tr ' ' '\n' \
    | grep -E '^[a-z0-9_]+=[0-9]+$' \
    | grep -vE '^(round|id|offered|bytes|t0_ns|t1_ns|t_read_ns|t_written_ns|t_eval_done_ns)=' \
    || true
}

# stdin: key=value lines. stdout: key<TAB>lower-median<TAB>count, one per key.
median_by_key() {
  sort -t= -k1,1 -k2,2n | awk -F= '
    { n[$1]++; v[$1, n[$1]] = $2 }
    END {
      for (k in n) {
        m = int((n[k] + 1) / 2)
        printf "%s\t%d\t%d\n", k, v[k, m], n[k]
      }
    }' | sort
}

get() {
  printf '%s\n' "$1" | awk -F'\t' -v k="$2" '$1 == k { print $2; exit }'
}

count() {
  printf '%s\n' "$1" | awk -F'\t' -v k="$2" '$1 == k { print $3; exit }'
}

parent=$(extract 'mtp-parent:' | median_by_key)
worker=$(extract 'mtp-worker:' | median_by_key)
session=$(extract 'mtp-trace0:' | median_by_key)

p_round=$(get "$parent" round_us)
if [ -z "$p_round" ] || [ "$p_round" -eq 0 ]; then
  echo "$0: no depth-0 mtp-parent rounds in $trace" >&2
  exit 1
fi
p_ew=$(get "$parent" encode_write_us)
p_wait=$(get "$parent" wait_us)
p_dec=$(get "$parent" decode_us)
p_gap=$(get "$parent" gap_us)
w_dec=$(get "$worker" decode_us)
w_handle=$(get "$worker" handle_us)
w_enc=$(get "$worker" encode_us)
w_write=$(get "$worker" write_us)
w_total=$(get "$worker" worker_us)
s_build=$(get "$session" build_us)
s_eval=$(get "$session" eval_wall_us)
s_read=$(get "$session" readout_us)
s_round=$(get "$session" round_us)
s_cpu_ns=$(get "$session" host_thread_cpu_ns)

for v in p_ew p_wait p_dec p_gap w_dec w_handle w_enc w_write w_total \
         s_build s_eval s_read s_round s_cpu_ns; do
  if [ -z "${!v}" ]; then
    echo "$0: field $v is missing from $trace (is every side traced?)" >&2
    exit 1
  fi
done

transport=$((p_wait - w_total))
w_outside=$((w_handle - s_round))
closure=$((p_round - p_ew - p_wait - p_dec))
# The in-process reference (39,655 us forward + 2,900 us lm_head) was timed
# around graph build AND eval (QwenPhaseBreakdownTests.swift:69-80), so the
# comparable session quantity is build + eval wall + readout, not eval alone.
# This reference is a debug-build best-of-5 figure (every raw log under
# docs/perf/raw-phase-*.txt:6 says "Building for debugging..."); the worker
# session below runs a release build, so forward_excess includes some of
# that build-mode gap alongside any real host-thread cost.
reference_us=42600
s_forward=$((s_build + s_eval + s_read))
forward_excess=$((s_forward - reference_us))

echo "rounds: parent=$(count "$parent" round_us) worker=$(count "$worker" worker_us) session=$(count "$session" round_us)"
echo
printf '%-44s %9s %7s\n' "bucket (lower median over depth-0 rounds)" "us" "share"
row() {
  printf '%-44s %9d %6.1f%%\n' "$1" "$2" "$(awk -v a="$2" -v b="$p_round" 'BEGIN { printf "%.1f", 100 * a / b }')"
}
row "parent encode + pipe write" "$p_ew"
row "transport + scheduling (derived)" "$transport"
row "worker JSON decode" "$w_dec"
row "worker outside the session (derived)" "$w_outside"
row "session graph build" "$s_build"
row "session eval wall" "$s_eval"
row "session readout" "$s_read"
row "worker JSON encode" "$w_enc"
row "worker pipe write" "$w_write"
row "parent JSON decode" "$p_dec"
row "parent round (measured)" "$p_round"
row "parent gap between rounds (outside the round)" "$p_gap"
echo
row "parent closure: round - (encode_write + wait + decode)" "$closure"
row "session build+eval+readout" "$s_forward"
row "forward excess over in-process 42.6 ms" "$forward_excess"
row "session host thread cpu (us, from cpu_ns)" "$((s_cpu_ns / 1000))"

# A derived bucket below zero means a stamp is missing or the sides are not
# from the same run; the table above is still printed so the cause is visible.
status=0
for pair in "transport=$transport" "worker_outside=$w_outside" "closure=$closure"; do
  if [ "${pair#*=}" -lt 0 ]; then
    echo "FAIL: derived bucket ${pair%%=*} is negative (${pair#*=} us)" >&2
    status=1
  fi
done
exit "$status"
