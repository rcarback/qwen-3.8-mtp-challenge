# Qwen3.8-Flash-Next port record, September 2026

Local perf fork only. This page records what the port verified and measured.
Every number below names the command that produced it.

## Identity

| Item | Value |
|---|---|
| Source | `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Source size | 131 safetensors shards, 360,023,349,944 bytes with tokenizer files (`tools/qwen38-flash/manifest.json`) |
| Runtime model type | `qwen4_exp_text` (flattened `text_config`), registered next to `qwen4_exp` |
| Runtime tree | dense tensors bf16 byte copies, routed experts affine 4-bit group 32, n-gram table bf16 in `ngram/` (memory-mapped) |
| Reference | mlx-lm PR 1788 (`tools/qwen38-flash/qwen4_exp.py`), llama.cpp PR 27742 and 27836 |

## Status

Now:

- Model code, transform, and generate verb are in the tree with 26 unit tests
  on small synthetic configurations (`swift test --force-resolved-versions --filter Qwen4Exp`).
- The n-gram hash constants come from the checkpoint tensors. The mlx-lm
  reference recomputes different multipliers from its seed formula, so the
  runtime never recomputes them.

Next (waits on the 335 GiB download, which waits on disk):

- Transform of the real checkpoint (`qwen4exp-transform`).
- Greedy continuations of the public prompts, prefill seconds, and decode
  tokens per second at 64 and 512 prompt tokens (`qwen4exp-generate`).
- Serve smoke through the headless MTP session.

Later:

- ANE dense lane A/B at 512 and 2048 prompt tokens (identical-token gate).
- Native MTP head accept rate at depth 2.

## Measurements

None yet. The download has not run.
