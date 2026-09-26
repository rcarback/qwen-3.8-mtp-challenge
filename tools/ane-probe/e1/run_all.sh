#!/usr/bin/env bash
# E1 kill-switch: convert -> correctness -> measure shape sweep.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(dirname "$0")/../venv/bin/python"
export PYTHONUNBUFFERED=1

echo "=== E1 convert ==="
"$PY" convert.py --seqs "${SEQS:-16,128,512,1024,2048}"

echo "=== E1 correctness ==="
"$PY" correctness.py --seqs "${CORR_SEQS:-16,128}"

echo "=== E1 measure ==="
"$PY" measure.py --seqs "${SEQS:-16,128,512,1024,2048}"

echo "=== E1 done ==="
ls -la artifacts/
