#!/usr/bin/env python3
"""Prefix-stable tool-result compaction as a sidecar proxy.

Sits between a coding CLI and `mlxfast-swift serve`. It rewrites the
`messages` array of an OpenAI-compatible /v1/chat/completions request --
replacing large, settled tool results with a deterministic content-derived
stub -- forwards the request upstream, and relays the response unchanged.

It touches no file of the serve path. Every prefix-stability property holds
through the existing renderer, tokenizer and checkpoint store by construction
of the rewritten message array.

Design: scratchpad/compaction-design.md. The load-bearing rules:

  STUB(R) iff size(R.content) >= S_min AND settled(R)
  settled(R) iff an assistant message appears after R in this request

`settled` is a function of successor existence only. Agent message arrays are
append-only, so it flips false->true exactly once per result and never back.
Anything that depended on total length or distance-from-the-end would rewrite
history every turn and make every turn a cold prefill.

Configuration (environment):

  MLXFAST_COMPACT           1/0    master switch                  (default 1)
  MLXFAST_COMPACT_S_MIN     int    stub threshold, characters     (default 2048)
  MLXFAST_COMPACT_EXPAND    1/0    expand() tool + store          (default 1)
  MLXFAST_COMPACT_UPSTREAM  url    serve base url    (default http://127.0.0.1:8080)
  MLXFAST_COMPACT_PORT      int    listen port                    (default 8099)
  MLXFAST_COMPACT_HOST      str    listen address           (default 127.0.0.1)
  MLXFAST_COMPACT_CACHE_DIR path   expand store root
                                   (default ~/.cache/mlxfast-compactor)
  MLXFAST_COMPACT_EXPAND_BUDGET  bytes  in-memory LRU budget  (default 1 GiB)
  MLXFAST_COMPACT_EXPAND_MAX_ROUNDS int expand hops per request  (default 4)
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
import threading
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_S_MIN = 2048
EXPAND_TOOL_NAME = "expand"


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Config:
    def __init__(self, **overrides):
        self.enabled = overrides.get(
            "enabled", _env_bool("MLXFAST_COMPACT", True))
        self.s_min = overrides.get(
            "s_min", _env_int("MLXFAST_COMPACT_S_MIN", DEFAULT_S_MIN))
        self.expand_enabled = overrides.get(
            "expand_enabled", _env_bool("MLXFAST_COMPACT_EXPAND", True))
        self.upstream = overrides.get(
            "upstream",
            os.environ.get("MLXFAST_COMPACT_UPSTREAM", "http://127.0.0.1:8080"))
        self.host = overrides.get(
            "host", os.environ.get("MLXFAST_COMPACT_HOST", "127.0.0.1"))
        self.port = overrides.get(
            "port", _env_int("MLXFAST_COMPACT_PORT", 8099))
        self.cache_dir = overrides.get(
            "cache_dir",
            os.environ.get(
                "MLXFAST_COMPACT_CACHE_DIR",
                os.path.expanduser("~/.cache/mlxfast-compactor")))
        self.expand_budget = overrides.get(
            "expand_budget",
            _env_int("MLXFAST_COMPACT_EXPAND_BUDGET", 1 << 30))
        self.expand_max_rounds = overrides.get(
            "expand_max_rounds",
            _env_int("MLXFAST_COMPACT_EXPAND_MAX_ROUNDS", 4))


# --------------------------------------------------------------------------
# stub rendering -- content-derived, therefore byte-identical across turns,
# sessions and server restarts. No timestamps, no counters, no ids.
# --------------------------------------------------------------------------

HANDLE_HEX = 16          # 64 bits of SHA-256
CALL_LINE_LIMIT = 200
EDGE_LINE_LIMIT = 100


def handle_for(content):
    """Stable 16-hex handle over the exact content bytes."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:HANDLE_HEX]


def _first_last_nonempty(content):
    head = tail = ""
    for line in content.splitlines():
        if line.strip():
            head = line.strip()
            break
    for line in reversed(content.splitlines()):
        if line.strip():
            tail = line.strip()
            break
    return head[:EDGE_LINE_LIMIT], tail[:EDGE_LINE_LIMIT]


def render_call_line(tool_call):
    """`Read(file_path=/x/y.swift, offset=1, limit=400)` from a tool_call.

    Arguments keep the tool_call's own key order, so the line is a pure
    function of the request bytes.
    """
    if not isinstance(tool_call, dict):
        return None
    fn = tool_call.get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    raw = fn.get("arguments")
    args = None
    if isinstance(raw, str) and raw.strip():
        try:
            args = json.loads(raw, object_pairs_hook=OrderedDict)
        except (ValueError, TypeError):
            args = None
    elif isinstance(raw, dict):
        args = raw
    if isinstance(args, dict):
        rendered = ", ".join(
            "%s=%s" % (k, v if isinstance(v, str) else json.dumps(v))
            for k, v in args.items())
    else:
        rendered = raw if isinstance(raw, str) else ""
    line = "%s(%s)" % (name, rendered)
    return line[:CALL_LINE_LIMIT]


def render_stub(content, tool_call=None, expand_enabled=True):
    """The exact replacement text for one tool result."""
    handle = handle_for(content)
    head, tail = _first_last_nonempty(content)
    lines = [
        "[compacted tool result | sha=%s | %d chars | %d lines]"
        % (handle, len(content), len(content.splitlines()))
    ]
    call_line = render_call_line(tool_call)
    if call_line:
        lines.append("call: " + call_line)
    if head:
        lines.append("head: " + head)
    if tail:
        lines.append("tail: " + tail)
    if expand_enabled:
        if call_line:
            lines.append(
                'To retrieve the full text call expand(handle="%s"), or '
                "re-run the call above." % handle)
        else:
            lines.append(
                'To retrieve the full text call expand(handle="%s").' % handle)
    else:
        if call_line:
            lines.append("To retrieve the full text re-run the call above.")
        else:
            lines.append("The full text is not retained; re-run the original call.")
    return "\n".join(lines)


EXPAND_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": EXPAND_TOOL_NAME,
        "description":
            "Retrieve the full text of a compacted tool result by its handle.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description":
                        "16-hex handle from a [compacted tool result] stub",
                },
            },
            "required": ["handle"],
        },
    },
}


# --------------------------------------------------------------------------
# the predicate
# --------------------------------------------------------------------------

def settled_flags(messages):
    """settled[i] == an assistant message exists after messages[i]."""
    flags = [False] * len(messages)
    seen_assistant = False
    for i in range(len(messages) - 1, -1, -1):
        flags[i] = seen_assistant
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            seen_assistant = True
    return flags


def _content_text(message):
    """The tool result's text, or None if this message carries no plain text.

    Only `role == "tool"` messages reach here. A parts array is joined the
    same way the serve renderer joins it, so size and stub are computed over
    what actually gets rendered.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                return None      # unknown shape: leave it alone
        return "".join(parts)
    return None


def _tool_call_index(messages):
    """tool_call_id -> the assistant tool_call that produced it.

    Ids never reach the token stream, so using them for *pairing* is safe;
    what lands in the stub is derived from the call's name and arguments.
    """
    index = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                index[call["id"]] = call
    return index


class CompactionResult:
    def __init__(self, messages, stubbed, stubbed_chars, stub_chars):
        self.messages = messages
        self.stubbed = stubbed              # [(handle, original_text)]
        self.stubbed_chars = stubbed_chars
        self.stub_chars = stub_chars


def compact_messages(messages, s_min=DEFAULT_S_MIN, expand_enabled=True):
    """Pure rewrite of the message array. Never mutates the input."""
    flags = settled_flags(messages)
    calls = _tool_call_index(messages)
    out = []
    stubbed = []
    stubbed_chars = 0
    stub_chars = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            out.append(message)          # user and assistant turns stay raw
            continue
        if not flags[i]:
            out.append(message)          # fresh: the model must read it now
            continue
        text = _content_text(message)
        if text is None or len(text) < s_min:
            out.append(message)
            continue
        stub = render_stub(
            text, calls.get(message.get("tool_call_id")), expand_enabled)
        replacement = dict(message)
        replacement["content"] = stub
        out.append(replacement)
        stubbed.append((handle_for(text), text))
        stubbed_chars += len(text)
        stub_chars += len(stub)
    return CompactionResult(out, stubbed, stubbed_chars, stub_chars)


def inject_expand_tool(tools):
    """Append the expand schema to a non-empty client tools array.

    A client that sends no tools has no tool loop and could never recover a
    stub, so nothing is injected there (the caller disables compaction too).
    """
    if not tools:
        return tools
    if any(isinstance(t, dict)
           and (t.get("function") or {}).get("name") == EXPAND_TOOL_NAME
           for t in tools):
        return tools
    return list(tools) + [EXPAND_TOOL_SCHEMA]


def compact_request(payload, config):
    """Rewrite a whole chat-completions payload. Returns (payload, result)."""
    messages = payload.get("messages")
    if not config.enabled or not isinstance(messages, list):
        return payload, CompactionResult(messages or [], [], 0, 0)
    tools = payload.get("tools")
    expand_available = config.expand_enabled and bool(tools)
    if config.expand_enabled and not tools:
        # No tool loop: a stub would be unrecoverable. Pass through raw.
        return payload, CompactionResult(messages, [], 0, 0)
    result = compact_messages(messages, config.s_min, expand_available)
    if not result.stubbed:
        return payload, result
    new_payload = dict(payload)
    new_payload["messages"] = result.messages
    if expand_available:
        new_payload["tools"] = inject_expand_tool(tools)
    return new_payload, result


# --------------------------------------------------------------------------
# expand storage
# --------------------------------------------------------------------------

class ExpandMiss(Exception):
    """Raised when a handle is not in the store. Never swallowed silently."""


class ExpandStore:
    """Content-addressed store for stubbed tool-result text.

    Lifetime, deliberately chosen and documented:

      * in-memory OrderedDict, byte-budgeted LRU (default 1 GiB). Eviction is
        least-recently-used, counted in content bytes.
      * write-through to `<cache_dir>/<handle>`. Writes are idempotent because
        the filename IS the content hash, so concurrent writers cannot produce
        a wrong file. Disk entries are never evicted by this process: they
        survive proxy restarts and are the recovery path for non-refetchable
        output (Bash, test runs). Reclaim them by deleting the directory --
        an operator action, not an automatic one, because a silent eviction of
        a Bash transcript is a permanent loss of detail.
      * a miss raises ExpandMiss. The proxy turns that into a loud tool result
        and a stderr line; it never returns empty content and never fails the
        HTTP request.
    """

    def __init__(self, cache_dir=None, budget=1 << 30):
        self.cache_dir = cache_dir
        self.budget = budget
        self._lock = threading.Lock()
        self._mem = OrderedDict()
        self._bytes = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, handle):
        # Callers gate on `self.cache_dir` before reaching here; assert so a
        # future caller that forgets fails loudly rather than joining None.
        assert self.cache_dir is not None, "_path requires a cache_dir"
        return os.path.join(self.cache_dir, handle)

    def put(self, handle, content):
        size = len(content.encode("utf-8"))
        with self._lock:
            if handle in self._mem:
                self._mem.move_to_end(handle)
            else:
                self._mem[handle] = content
                self._bytes += size
                while self._bytes > self.budget and len(self._mem) > 1:
                    _, evicted = self._mem.popitem(last=False)
                    self._bytes -= len(evicted.encode("utf-8"))
        if not self.cache_dir:
            return
        path = self._path(handle)
        if os.path.exists(path):
            return
        tmp = "%s.%d.tmp" % (path, os.getpid())
        try:
            with open(tmp, "w", encoding="utf-8") as handle_file:
                handle_file.write(content)
            os.replace(tmp, path)
        except OSError as error:
            _log("expand-store write failed for %s: %s" % (handle, error))
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def get(self, handle):
        with self._lock:
            if handle in self._mem:
                self._mem.move_to_end(handle)
                return self._mem[handle]
        if self.cache_dir:
            try:
                with open(self._path(handle), "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                content = None
            if content is not None:
                self.put(handle, content)
                return content
        raise ExpandMiss(handle)


def expand_miss_text(handle):
    return ('expand-miss: unknown handle "%s". The full text is not in the '
            "compaction store (evicted, or produced by a different machine). "
            "Re-run the original call shown in the stub." % handle)


# --------------------------------------------------------------------------
# response inspection
# --------------------------------------------------------------------------

def _accumulate_tool_calls(acc, deltas):
    for delta in deltas or []:
        if not isinstance(delta, dict):
            continue
        idx = delta.get("index", 0)
        slot = acc.setdefault(
            idx, {"id": None, "type": "function",
                  "function": {"name": "", "arguments": ""}})
        if delta.get("id"):
            slot["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


def expand_handles(tool_calls):
    """The handles requested by expand() calls, in call order."""
    handles = []
    for call in tool_calls or []:
        fn = (call or {}).get("function") or {}
        if fn.get("name") != EXPAND_TOOL_NAME:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        handles.append((call, args.get("handle")))
    return handles


class StreamRelay:
    """Relays an SSE stream byte-for-byte, with one decision point.

    Chunks are buffered only until the first tool-call name or the first
    content delta is known. If the turn is not a pure expand() call the buffer
    is flushed verbatim and every later byte is written straight through, so
    the client sees the upstream byte stream unchanged. If it IS an expand
    call the stream is swallowed and handed back for interception.
    """

    def __init__(self, write, flush=None):
        self._write = write
        self._flush = flush or (lambda: None)
        self._buffer = []
        self.passthrough = False
        self.tool_calls = {}
        self.saw_content = False

    def _decide(self):
        if self.passthrough:
            return
        if self.saw_content:
            self._open()
            return
        names = [c["function"]["name"] for c in self.tool_calls.values()
                 if c["function"]["name"]]
        if names and any(n != EXPAND_TOOL_NAME for n in names):
            self._open()

    def _open(self):
        self.passthrough = True
        for chunk in self._buffer:
            self._write(chunk)
        self._buffer = []
        self._flush()

    def feed(self, chunk):
        if self.passthrough:
            self._write(chunk)
            self._flush()
            return
        self._buffer.append(chunk)
        for line in chunk.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                self._open()
                return
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    self.saw_content = True
                _accumulate_tool_calls(self.tool_calls, delta.get("tool_calls"))
        self._decide()

    def flush_all(self):
        """Give up on interception and emit everything buffered."""
        self._open()

    def finish(self):
        """Returns the swallowed expand tool_calls, or None after passthrough."""
        if self.passthrough:
            return None
        calls = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        if calls and all(
                c["function"]["name"] == EXPAND_TOOL_NAME for c in calls):
            return calls
        self._open()
        return None


# --------------------------------------------------------------------------
# proxy
# --------------------------------------------------------------------------

def _log(message):
    sys.stderr.write("[compactor] %s\n" % message)
    sys.stderr.flush()


class Upstream:
    def __init__(self, base_url):
        parsed = urllib.parse.urlsplit(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.https = parsed.scheme == "https"
        self.prefix = parsed.path.rstrip("/")

    def request(self, method, path, body, headers):
        cls = (http.client.HTTPSConnection if self.https
               else http.client.HTTPConnection)
        conn = cls(self.host, self.port, timeout=3600)
        conn.request(method, self.prefix + path, body=body, headers=headers)
        return conn, conn.getresponse()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Bound by `serve()` on the handler subclass before the server accepts a
    # connection, so these are never None in a request. Declared rather than
    # assigned None so the attribute type is Config/ExpandStore, not None.
    config: "Config"
    store: "ExpandStore"

    def log_message(self, fmt, *args):      # quieter than the default
        pass

    # -- plumbing ---------------------------------------------------------

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _upstream_headers(self, body):
        headers = {}
        for key, value in self.headers.items():
            if key.lower() in ("host", "content-length", "connection",
                               "transfer-encoding", "accept-encoding"):
                continue
            headers[key] = value
        headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        return headers

    def _relay_plain(self, response):
        body = response.read()
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() in ("transfer-encoding", "content-length",
                               "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream_headers(self, response):
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() in ("transfer-encoding", "content-length",
                               "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        # No Content-Length on a relayed stream: the client reads to EOF.
        self.close_connection = True

    # -- routing ----------------------------------------------------------

    def do_GET(self):
        self._forward_opaque("GET", b"")

    def do_POST(self):
        body = self._read_body()
        if self.path.rstrip("/").endswith("/chat/completions"):
            self._handle_completion(body)
        else:
            self._forward_opaque("POST", body)

    def _forward_opaque(self, method, body):
        upstream = Upstream(self.config.upstream)
        conn, response = upstream.request(
            method, self.path, body, self._upstream_headers(body))
        try:
            self._relay_plain(response)
        finally:
            conn.close()

    # -- the interesting path ---------------------------------------------

    def _handle_completion(self, body):
        try:
            payload = json.loads(body)
        except ValueError:
            self._forward_opaque("POST", body)
            return
        payload, result = compact_request(payload, self.config)
        for handle, text in result.stubbed:
            if self.config.expand_enabled:
                self.store.put(handle, text)
        if result.stubbed:
            _log("stubbed %d results: %d chars -> %d chars"
                 % (len(result.stubbed), result.stubbed_chars,
                    result.stub_chars))

        streaming = bool(payload.get("stream"))
        rounds = 0
        while True:
            outbound = json.dumps(payload).encode("utf-8")
            upstream = Upstream(self.config.upstream)
            conn, response = upstream.request(
                "POST", self.path, outbound, self._upstream_headers(outbound))
            try:
                if streaming:
                    calls = self._relay_streaming(response)
                else:
                    calls = self._relay_buffered(response)
            finally:
                conn.close()
            if calls is None:
                return
            rounds += 1
            if rounds > self.config.expand_max_rounds:
                _log("expand round limit reached; returning last response")
                return
            payload = dict(payload)
            payload["messages"] = list(payload["messages"]) + \
                self._expand_turn(calls)

    def _relay_streaming(self, response):
        if response.status != 200:
            self._relay_plain(response)
            return None
        opened = [False]

        def write(chunk):
            if not opened[0]:
                self._send_stream_headers(response)
                opened[0] = True
            self.wfile.write(chunk)

        relay = StreamRelay(write, self.wfile.flush)
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            relay.feed(chunk)
        calls = relay.finish()
        if calls is None:
            if not opened[0]:
                self._send_stream_headers(response)
            self.wfile.flush()
            return None
        if not self.config.expand_enabled:
            # Nothing will service the call; emit the stream we held back
            # rather than dropping it on the floor.
            relay.flush_all()
            self.wfile.flush()
            return None
        return calls

    def _relay_buffered(self, response):
        raw = response.read()
        calls = None
        if response.status == 200 and self.config.expand_enabled:
            try:
                parsed = json.loads(raw)
                message = (parsed.get("choices") or [{}])[0].get("message") or {}
                tool_calls = message.get("tool_calls") or []
                if tool_calls and all(
                        (c.get("function") or {}).get("name") == EXPAND_TOOL_NAME
                        for c in tool_calls):
                    calls = tool_calls
            except (ValueError, AttributeError, IndexError, TypeError):
                calls = None
        if calls is not None:
            return calls
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() in ("transfer-encoding", "content-length",
                               "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        return None

    def _expand_turn(self, calls):
        """Assistant tool_call turn plus its tool results, as a strict suffix."""
        messages = [{"role": "assistant", "content": None, "tool_calls": calls}]
        for call, handle in expand_handles(calls):
            call_id = call.get("id") or ("expand-%s" % handle)
            try:
                content = self.store.get(handle) if handle else None
                if content is None:
                    raise ExpandMiss(handle)
            except ExpandMiss:
                _log('EXPAND MISS handle=%s -- returning a loud tool result'
                     % handle)
                content = expand_miss_text(handle)
            else:
                _log("expand hit handle=%s (%d chars)" % (handle, len(content)))
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": content})
        return messages


def serve(config=None):
    config = config or Config()
    Handler.config = config
    Handler.store = ExpandStore(
        config.cache_dir if config.expand_enabled else None,
        config.expand_budget)
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    _log("listening on %s:%d -> %s (compaction=%s, S_min=%d, expand=%s)"
         % (config.host, config.port, config.upstream,
            "on" if config.enabled else "off", config.s_min,
            "on" if config.expand_enabled else "off"))
    server.serve_forever()


if __name__ == "__main__":
    serve()
