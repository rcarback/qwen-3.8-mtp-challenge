# serve-compactor

A sidecar proxy that compacts large, settled tool results before they reach
`mlxfast-swift serve`. It modifies no file of the serve path.

Design: `scratchpad/compaction-design.md`.

## Run

```bash
# serve on 8080, proxy on 8099
MLXFAST_COMPACT_UPSTREAM=http://127.0.0.1:8080 \
MLXFAST_COMPACT_PORT=8099 \
python3 tools/serve-compactor/compactor.py
```

Point the coding CLI at `http://127.0.0.1:8099/v1`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `MLXFAST_COMPACT` | `1` | master switch; `0` is a pure passthrough |
| `MLXFAST_COMPACT_S_MIN` | `2048` | stub threshold, characters |
| `MLXFAST_COMPACT_EXPAND` | `1` | `expand()` tool, interception, and store |
| `MLXFAST_COMPACT_UPSTREAM` | `http://127.0.0.1:8080` | serve base URL |
| `MLXFAST_COMPACT_HOST` / `_PORT` | `127.0.0.1` / `8099` | listen address |
| `MLXFAST_COMPACT_CACHE_DIR` | `~/.cache/mlxfast-compactor` | expand store |
| `MLXFAST_COMPACT_EXPAND_BUDGET` | `1073741824` | in-memory LRU bytes |
| `MLXFAST_COMPACT_EXPAND_MAX_ROUNDS` | `4` | expand hops per request |

## What it does

A `role == "tool"` message is replaced by a stub when both hold:

* its content is at least `S_min` characters, and
* at least one assistant message follows it in this request ("settled").

Message arrays are append-only, so `settled` flips false to true exactly once
per result and never back. Nothing else is touched: user turns, assistant
turns, small results, and results the model has not yet read stay raw. Whole
results only; a result is never truncated.

The stub is a pure function of the content plus the paired assistant
`tool_call`, so it is byte-identical across turns, sessions, and restarts.

## expand()

When enabled, the proxy appends an `expand(handle)` tool to a non-empty client
`tools` array, intercepts calls to it, and answers them itself by appending the
stored text as a tool message and re-posting upstream. The client never sees
the tool.

Storage lifetime: in-memory byte-budgeted LRU plus write-through files under
the cache directory, named by content hash. Disk entries are never evicted
automatically; delete the directory to reclaim. A miss returns a loud
`expand-miss:` tool result and logs to stderr. It never fails the request and
never returns empty content.

## Tests

```bash
python3 tools/serve-compactor/test_compactor.py
```

No model, no GPU: the upstream is a stub HTTP server in-process.
