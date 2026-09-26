#!/usr/bin/env bash
# E3 characterisation matrix: convert -> correctness -> ANE -> Metal -> GPU -> sustained.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(dirname "$0")/../venv/bin/python"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=== E3 catalog ==="
"$PY" catalog.py

echo "=== E3 convert ==="
"$PY" convert.py "$@"

echo "=== E3 correctness (small seqs) ==="
"$PY" correctness.py --only-small

echo "=== E3 measure ANE ==="
"$PY" measure_coreml.py --backends ane --out measure_coreml.json

echo "=== E3 measure Metal ==="
"$PY" measure_metal.py --out measure_metal.json

echo "=== E3 measure GPU ==="
"$PY" measure_coreml.py --backends gpu --out measure_coreml.json

echo "=== E3 sustained ==="
"$PY" sustained.py --duration-s 60

echo "=== E3 done ==="
ls -la artifacts/*.json
