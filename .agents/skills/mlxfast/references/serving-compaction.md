# Bound serving compaction

Compaction changes the prompt. Evaluate task success and total request cost,
including recovery of removed content. Report reduced prompt work separately from
a faster kernel or unchanged-input inference throughput.

## Ownership and behavior

1. Identify every layer that rewrites the conversation. Prefer one owner of each
   compacted result and its retrieval handle. Multiple owners need an explicit protocol.
2. Define which results may be compacted from actual conversation semantics.
   A later assistant message is a useful heuristic, not proof that content is
   irrelevant. Branches, edits, and resumed sessions can need older results.
3. Preserve active instructions, tool-call associations, and template rendering.
   Check the exact serialized prompt when JSON ordering or whitespace affects tokens.
4. Provide retrieval when the product promises recoverable content. Define missing,
   evicted, and expired handles explicitly. Irreversible summaries are a separate
   product choice with their own task-quality tests.
5. Bound stored originals in memory and on disk. Provide an effective off switch
   and include retrieval rounds, retries, and persistence cost in measurements.

Compaction can apply to a single long request as well as a multi-turn session.
The deciding factor is authorized input reduction and task quality, not the
number of serving layers or whether every tool result has been answered.

## Tests and measurements

Compare the same initial transcript with compaction disabled and enabled. Include
an input below the threshold, a result needed later, an edited conversation,
multiple expansions, eviction, and restart behavior where supported.

Record which owner compacted each result, tokens removed, retrieval count, failed
handles, task success, total wall time, and memory/disk growth. Synthetic fixtures
are useful boundary tests. Representative sessions support product claims.

## Current repository cautions

- `QwenRuntimeServe.swift` provides native compaction and expansion.
  `tools/serve-compactor/compactor.py` provides another owner. Check the deployment
  topology before enabling both on the same conversation.
- `CompactionStore.swift` bounds native in-memory retention. The proxy README
  documents disk entries without automatic eviction. A RAM cap does not bound disk.
- A raised token threshold is an imperfect substitute for an explicit off switch.
  Verify the effective configuration in the running process.
- `OrderedJSON.swift` preserves ordering required by the prompt template. Reuse the
  template's rendering semantics instead of normalizing tool arguments casually.

These source-review findings still need task-quality evaluation. Recheck the
current code and configuration before changing them. Trusted serving
files and a Python proxy do not become ranked editable code because they help locally.

Keep compaction only when the declared task-quality and resource budgets pass.
If one owner already meets the goal, delete redundant rewriting before adding
coordination. Use [measurement](measurement.md) for matched arms and records.
