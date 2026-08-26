#!/bin/bash
# Refuses to proceed while the host is too noisy to time on.
#
# The screensaver is the specific hazard this exists for. Its idle timer is a
# separate clock from display sleep, so `caffeinate -d` does not suppress it,
# and `loginwindow` respawns it within seconds of a `pkill`, so a watchdog does
# not either. The only durable fix is idleTime 0, which this script verifies
# rather than attempts: writing it needs the user's own shell.
#
# THE BINDING SIGNAL IS THE GPU, NOT THE CPU (revised 2026-08-26). The first
# version vetoed on any unrelated process above 25% CPU, and that veto fired on
# `WindowServer` and the terminal emulator while the GPU sat at its 0.35 W idle
# floor -- including on the terminal this harness prints into, which makes the
# gate refuse partly because of its own output. Compositing a desktop is not
# contention for a prefill that pulls the GPU to full power. The screensaver
# was never dangerous because it used a CPU. It was dangerous because it drove
# GPU power from 0.3 W to 9.2 W. So the GPU is sampled over a window, and the
# CPU check is kept only for a genuine compute hog outside the windowing stack.
#
# Be honest about which check stops which hazard. An eight-second window sees a
# 300-second screensaver cycle about three percent of the time, so the window is
# NOT what defends against that spike -- the idleTime assertion at the top is,
# because it is a state check rather than a sample. What the window does catch
# is sustained load, which it does well: it refused at 11.5 W and 17.6 W in
# negative testing. Both checks are needed and neither substitutes for the
# other.
set -uo pipefail

fail() {
  echo "host-quiet-gate: $1" >&2
  exit 1
}

idle=$(defaults -currentHost read com.apple.screensaver idleTime 2>/dev/null || echo unset)
if [ "$idle" != "0" ]; then
  echo "host-quiet-gate: screensaver idleTime is '$idle', must be 0" >&2
  echo "  run this in your own shell, then retry:" >&2
  echo "    defaults -currentHost write com.apple.screensaver idleTime 0" >&2
  exit 1
fi

if pgrep -x legacyScreenSaver >/dev/null 2>&1; then
  fail "legacyScreenSaver is running"
fi

# Compute hogs only. WindowServer, the terminal emulator, and the browser
# render process are excluded BY NAME because they are windowing-stack load
# whose GPU cost the sampled window below measures directly and far better.
#
# MATCH THE WHOLE COMMAND, NOT $2 (fixed 2026-08-26). `comm=` prints a full
# path, and awk splits it on whitespace, so any bundle whose name contains a
# space arrives as several fields. "/Applications/Google Chrome.app/.../Google
# Chrome Helper (Renderer)" therefore had $2 == "/Applications/Google", and the
# `Renderer` exclusion could never match it -- inverting the guard from "never
# veto on the browser" to "always veto on the browser" for the one entry in the
# list whose path has a space. Every other excluded name is space-free, which is
# why this survived. Reconstruct the command from the first field's offset and
# test that.
busy=$(ps -Ao pcpu=,comm= -r | head -12 |
  awk '{ cmd = substr($0, index($0, $2)) }
       $1 > 40.0 && cmd !~ /mlxfast|swift|clang|ld$|WindowServer|ghostty|Terminal|iTerm|Renderer|plugin-container|claude|Google Chrome|launchd/ { print cmd " " $1 }')
if [ -n "$busy" ]; then
  echo "host-quiet-gate: unrelated compute over 40% CPU:" >&2
  echo "$busy" >&2
  exit 1
fi

if ! command -v macmon >/dev/null 2>&1; then
  fail "macmon is absent, so the GPU cannot be checked. Install it with 'brew install macmon'"
fi

# Check these here, by name. Both are used below and both fail closed when
# absent, but they fail with a message about macmon giving zero samples or a
# peak of 0W, which sends the reader after the wrong tool.
if ! command -v jq >/dev/null 2>&1; then
  fail "jq is absent, so the macmon samples cannot be parsed. Install it with 'brew install jq'"
fi
if ! command -v bc >/dev/null 2>&1; then
  fail "bc is absent, so the thresholds cannot be compared. Install it with 'brew install bc'"
fi

# Sample across a window rather than once, to catch sustained load that a
# single reading could land either side of. See the header for why this window
# is not the defence against the five-minute screensaver cycle.
peak_power=0
peak_temp=0
power_samples=0
temp_samples=0
while read -r p t; do
  # A field that arrives as "null" or empty must not be silently treated as
  # zero: that would make its threshold vacuous and pass a hot GPU on the
  # strength of the OTHER field. Count valid samples per field and require
  # both below.
  case "$p" in '' | null) : ;; *)
    power_samples=$((power_samples + 1))
    over=$(echo "$p > $peak_power" | bc -l 2>/dev/null || echo 0)
    [ "$over" = "1" ] && peak_power=$p
    ;;
  esac
  case "$t" in '' | null) : ;; *)
    temp_samples=$((temp_samples + 1))
    over=$(echo "$t > $peak_temp" | bc -l 2>/dev/null || echo 0)
    [ "$over" = "1" ] && peak_temp=$t
    ;;
  esac
done < <(macmon pipe -s 8 -i 1000 2>/dev/null |
  jq -r '"\(.gpu_power) \(.temp.gpu_temp_avg)"')

[ "$power_samples" -ge 4 ] || fail "macmon gave only $power_samples gpu_power samples of 8"
[ "$temp_samples" -ge 4 ] || fail "macmon gave only $temp_samples gpu_temp samples of 8"

ok=$(echo "$peak_power < 2.0 && $peak_temp < 53.0" | bc -l 2>/dev/null || echo 0)
[ "$ok" = "1" ] || fail "gpu not idle over 8s: peak ${peak_power}W ${peak_temp}C"
printf "host-quiet-gate: ok (gpu peak %.2fW %.1fC over 8s)\n" "$peak_power" "$peak_temp"
