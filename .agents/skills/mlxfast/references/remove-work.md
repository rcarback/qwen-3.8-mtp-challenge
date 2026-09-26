# Remove redundant work

Use this path when the forward repeats computation, allocation, copies, or
synchronization that the required result does not need. Simplification can be
worth keeping even when timing stays unchanged. Claim speed only when measured.

## Order of attack

1. Trace the actual call path. Check whether compilation already fuses or removes
   the suspected work. Reuse profiling before introducing counters.
2. Delete duplicate operations or reuse an existing value. Prefer a model-owned
   constant or a per-forward value over a global memoization layer.
3. For caching, enumerate every input that changes the value: geometry, dtype,
   device, offset, positional encoding, and mutable model state as applicable.
4. Bound cache lifetime and cardinality. Share one positional table across layers
   only when those layers use the same positions and encoding parameters.
5. Remove unnecessary host readbacks and barriers. Preserve stream ordering and
   buffer lifetime. Check an uninstrumented forward because barriers change overlap.
6. Remove branches and configuration made obsolete by the simpler code.

A request-dependent value can be safely reused within that request. It needs a
content-complete key for longer reuse. A rollback to an old offset may reuse a
correctly keyed entry. Rollback alone does not require invalidation.

## Validation

Apply the execution gate in [measurement](measurement.md). Test changed keys,
cache lifetime, concurrent model instances, rollback, and stream selection where
those behaviors changed. Exact operations should match exactly. A numerical
reordering still needs the deployment's required fidelity checks.

Use hit/miss or operation counts when engagement is otherwise ambiguous. A fully
warmed cache can correctly have zero new misses. Force recomputation as a control
without introducing cleanup overhead into only one timed arm.

Cover affected decode, verify, and prefill regimes. A simple shape predicate can
keep a local win without imposing it on other shapes. Reject an unbounded global
cache when a scoped value provides the same reuse.

## Repository examples

- `Qwen35.swift` memoizes the epsilon scalar by its bit pattern. Commit `9a3ea918`
  removed 127 scalar constructions per width-1 forward.
- Commit `4e2c4122` reported byte-exact packed prework and removed a special case.
  Its forward measured 35.75 ms versus 35.72 ms: no supported latency gain.
- `docs/perf/mlx-serve-attribution-2026-09-06.md` proposes hoisting RoPE tables.
  Its 0.25–0.50 ms estimate is unmeasured, not an accepted speedup.
- `Vendor/mlx-swift/Source/MLX/Stream.swift` had an argument-drop fix in
  `StreamOrDevice.stream(_:)`. A caller can depend on a utility outside the
  ranked editable surface. Check packaging before claiming the fix ships.

Keep a reduction in code or retained memory on its own merits. Remove a cache
whose bookkeeping exceeds its benefit and has no other justified use.
