# Reuse prefix state safely

Prefix reuse saves repeated prefill in workloads with matching token prefixes.
It changes request cost rather than the per-token forward. Exact state reuse
needs the deployment's fidelity gate. Lossy state snapshots also need a
quality budget and tests across repeated restore cycles.

## State and identity

Start with live-session reuse. Add shared or disk checkpoints when the workload
needs persistence or reuse across sessions. Validate the exact token sequence and
all other state-producing inputs, including adapters and positional settings.
For multimodal models, include the nontext inputs.

For each checkpoint, retain its own token sequence and state boundary. A hash
can locate a candidate, but confirm token identity before reuse where practical.
An edited request may reuse its matching prefix if a valid checkpoint exists.
A strict-extension-only path must fall back on edits.

Equal-length requests can reuse state when the runtime retains enough information
to produce the next logits/token correctly. Otherwise replay the required suffix.
Do not turn one session API's strict-extension rule into a universal restriction.

A persistent cache identity should cover effective weights, state format/version,
position semantics, precision policy, and other state-producing configuration.
Tokenizer/template identity is useful for text-derived keys. If keys contain
exact token IDs and all model inputs, determine whether tokenizer identity adds
needed protection rather than assuming its absence alone corrupts the state.

## Memory and lifetime

Count unique retained buffers, attention KV, recurrent snapshots, and metadata.
Copy-on-write views can share storage but can also retain a much larger backing
allocation. Evicting one owner must not subtract bytes still retained by another.
Bound in-memory and disk retention separately.

Choose eviction based on recomputation cost and the deployment's session policy.
Oldest-first within a protected conversation is the current repository behavior.
Evicting the oldest entry last is not a general requirement.

Persisted session state can be useful after restart when identity, ownership, and
state reconstruction remain valid. Content-addressed checkpoints are one design,
not the only legitimate persistent form.

## Validation

Compare fresh prefill against extension, matched checkpoint, and disk restore.
Exercise attention and recurrent state, edited prefixes, equal lengths, branching,
eviction, cancellation, worker restart, and fingerprint mismatches where supported.
For lossy snapshots, repeat restore/continue cycles to reveal accumulated error.

Measure actual saved and recomputed tokens, TTFT, request wall time, retention,
and misses. Use a no-match request and forced-fresh run as controls. A replay
model can guide cache sizing, but concurrency claims need live serving evidence.

## Repository evidence

- `Sources/MLXFastModel/QwenSessionCacheStore.swift` separates append-only KV from
  recurrent state. Its documented example has 64 KiB/token KV and 144 MiB of
  recurrent state per checkpoint. Recompute for the current model and precision.
- `Sources/MLXFastModel/QwenPrefillDiskCache.swift` fingerprints weights identity,
  chunk size, and KV policy. Audit key construction and state inputs together.
- Commit `2fdee6bf` retracted an offline 12.8x concurrency-cost prediction after
  the live server measured 1.00x. The predictor's baseline was not enough.
- Beads `c44` and `6tb` cover byte ownership and restore fidelity. Read their
  current status and tests before treating an open question as a fixed defect.

If the benchmark provides no reusable prefixes, this mechanism cannot improve
that benchmark. Keep it for authorized multi-request serving when it provides
measured benefit. Use [measurement](measurement.md) for the evidence standard.
