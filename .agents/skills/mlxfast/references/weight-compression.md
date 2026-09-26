# Compress selected weight families

Weight compression can reduce artifact size, residency, or bytes read during
inference. It is an approximate target change when effective weights change.
Confirm the deployment permits it and declare a quality budget before tuning.
For proposal-only heads, use [speculation](speculation.md) instead.

## Work from the deployed representation

1. Inventory eligible families and their actual bit width, group size, scales,
   biases, and copies. Separate resident experts from experts touched per step.
2. Confirm the transform runs. Log changed module paths and inspect resulting
   tensor formats. A nonzero module visit count does not prove a precision change.
3. Prefer original higher-precision weights when available. If requantizing an
   existing quantized checkpoint, record that lineage and accumulated error.
4. Change one family or precision parameter at a time. Do not assume attention,
   embeddings, or shared experts have the same sensitivity across models.
5. Measure the selected runtime kernel as well as payload size. Unpacking,
   metadata, or a slower kernel can cancel the nominal bandwidth saving.
6. Combine families that individually passed quality checks only after rechecking the combined
   policy. Errors can compound across layers and tensor families.

`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift` has a quantization hook that
skips modules already conforming to `Quantized`. Requesting q4 on an already-q8
module through that hook can do nothing. Inspect the transformed model.

Tied embeddings and output heads are constraints to preserve, not automatic
disqualifiers. Update all consumers consistently or retain their shared baseline.
A pinned checkpoint may permit lossless packing while forbidding changed target
values. Read the actual editable-surface and correctness contract.

## Bytes and quality

At fixed parameter count, moving group-32 to group-64 makes groups **larger**
and approximately halves group metadata. It leaves packed weight bits unchanged.
Smaller groups use more metadata and can reduce quantization error. Count both
payload and metadata using [budget](budget.md).

Compare INT4 against the actual baseline: fp16, q8, and q4 imply different savings.
A fourfold nominal payload reduction from fp16 is not a guaranteed speedup.
Check whether weights stay compressed in the executing runtime.

Run held-out NLL/perplexity and behavior checks on the same candidate artifact.
Record decision disagreements and logit gaps when relevant to fidelity. Reconstruction
error alone cannot establish model quality. For strict target contracts, a failed
required gate rejects the candidate even if aggregate perplexity is close.

## Historical evidence

Local M4 Max MoE results in `docs/perf/qwen38-flash-2026-09.md`:

| Policy against bf16 dense weights | Decode | Perplexity | Interpretation |
| --- | --- | --- | --- |
| Blanket dense q4 g32 | 64.0 → 42.5 ms/token | 4.340 → 5.043 | 33% latency reduction with 16% PPL increase, rejected |
| Blanket dense q8 g32 | 64.0 → 57.1 ms/token | 4.340 → 4.327 | Adopted local dense policy, limited quality evidence |
| Mixed q8/q4 policies | See source | 7.7% and 15% PPL increases | Sensitive families still affected |

Commit `802160cb` halved expert group metadata with group-64. Reconstruction
metrics passed its threshold, but median disagreements reached 43.5 per 512
positions. Bead `mc6` proposes narrower family selection. Recheck current status
before repeating it.

Use [measurement](measurement.md). Reject policies outside the quality budget.
Keep capacity or disk savings only when they serve the stated objective and
justify any latency cost. Remove abandoned policy flags after recording results.
