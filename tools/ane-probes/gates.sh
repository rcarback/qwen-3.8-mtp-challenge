# shellcheck shell=bash
# Measurement gates for timed arms on the local box. Source, do not execute.
#
#   cool_gate      wait for the GPU below 40C (macmon), 900 s ceiling
#   quiet_gate     wait until no Time Machine backup is copying and the media,
#                  photo and Spotlight analysers are idle, 900 s ceiling
#   ensure_metallib  put mlx.metallib beside a freshly built test bundle
#
# Time Machine and the analysers churn the page cache the n-gram table
# lives in; a prompt then takes minutes outside the model's own timers.
# Pause the analysers with SIGSTOP for the timed phase and SIGCONT after.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cool_gate() {
  # The GPU sensor intermittently reads 1.6C (implausible); then fall back to
  # the CPU average below 41C, which idles near 40C, and wait at least 45 s.
  local t c start elapsed
  start=$(date +%s)
  while :; do
    read -r t c < <(macmon pipe -s 1 2>/dev/null | jq -r '[.temp.gpu_temp_avg, .temp.cpu_temp_avg] | @tsv' | head -1)
    elapsed=$(( $(date +%s) - start ))
    if [ -n "$t" ] && awk -v t="$t" 'BEGIN{exit !(t >= 5.0)}'; then
      if awk -v t="$t" 'BEGIN{exit !(t < 40.0)}' && [ "$elapsed" -ge 45 ]; then echo "cool_gate: gpu ${t}C ok after ${elapsed}s $(date +%T)"; return 0; fi
    else
      if awk -v c="$c" 'BEGIN{exit !(c < 41.0)}' && [ "$elapsed" -ge 45 ]; then echo "cool_gate: gpu sensor implausible (${t}C), cpu ${c}C ok after ${elapsed}s $(date +%T)"; return 0; fi
    fi
    if [ "$elapsed" -gt 900 ]; then echo "cool_gate: ceiling hit (gpu ${t}C cpu ${c}C)"; return 1; fi
    sleep 10
  done
}

quiet_gate() {
  local start busy tm elapsed
  start=$(date +%s)
  while :; do
    elapsed=$(( $(date +%s) - start ))
    tm=$(tmutil status 2>/dev/null | rg -c "Running = 1" || true)
    busy=$(ps -A -o pcpu,comm | rg -i "mediaanalysisd$|photoanalysisd$|backupd$|mdworker_shared|mds_stores" | awk '{s+=$1} END {print int(s)}')
    if [ "${tm:-0}" -eq 0 ] && [ "${busy:-0}" -lt 20 ]; then echo "quiet_gate: ok (analysers ${busy}% cpu, no backup) after ${elapsed}s $(date +%T)"; return 0; fi
    if [ "$elapsed" -gt 900 ]; then echo "quiet_gate: ceiling hit (analysers ${busy}% cpu, backup running=${tm})"; return 1; fi
    sleep 15
  done
}

ensure_metallib() {
  # swift test -c release rebuilds the xctest bundle without mlx.metallib
  # beside its binary; MLX looks there first.
  local b="$REPO/.build/arm64-apple-macosx/release/mlxfast-challenge-devPackageTests.xctest/Contents/MacOS"
  [ -d "$b" ] && [ ! -f "$b/mlx.metallib" ] && cp "$REPO/.build/arm64-apple-macosx/release/mlx.metallib" "$b/mlx.metallib" && echo "ensure_metallib: provisioned"
  return 0
}
