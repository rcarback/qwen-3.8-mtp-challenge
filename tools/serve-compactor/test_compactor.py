#!/usr/bin/env python3
"""Tests for the tool-result compaction proxy.

No model, no GPU: the upstream is a stub HTTP server in this process.

Run:  python3 tools/serve-compactor/test_compactor.py
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compactor as C


# --------------------------------------------------------------------------
# a synthetic session, grown the way an agent harness grows one: append-only
# --------------------------------------------------------------------------

BIG_A = "".join("line %d of the first big file\n" % i for i in range(400))
BIG_B = "".join("row %d of a long command transcript\n" % i for i in range(300))
SMALL = "ok (312 bytes)\n" * 3


def session_messages():
    """The full end state. Requests are prefixes of this list."""
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the parser."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "Read",
                "arguments": '{"file_path": "/x/Parser.swift", "limit": 400}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": BIG_A},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {
                "name": "Bash", "arguments": '{"command": "swift test"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": SMALL},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c3", "type": "function", "function": {
                "name": "Bash", "arguments": '{"command": "swift build -v"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "content": BIG_B},
        {"role": "assistant", "content": "Here is the fix."},
        {"role": "user", "content": "Now run the benchmark."},
    ]


def render_message(message):
    """Stand-in for the serve renderer: enough framing that a byte-comparison
    over the concatenation is a real prefix comparison."""
    body = message.get("content")
    if body is None:
        body = ""
    if isinstance(body, list):
        body = "".join(b.get("text", "") for b in body)
    calls = ""
    for call in message.get("tool_calls") or []:
        fn = call["function"]
        calls += "\n<tool_call>%s %s</tool_call>" % (fn["name"], fn["arguments"])
    return "<|im_start|>%s\n%s%s<|im_end|>\n" % (message["role"], body, calls)


def render(messages):
    return "".join(render_message(m) for m in messages)


# --------------------------------------------------------------------------
# 1. prefix stability -- the property the whole design rests on
# --------------------------------------------------------------------------

class PrefixStabilityTests(unittest.TestCase):
    def test_each_message_flips_at_most_once_and_only_raw_to_stub(self):
        full = session_messages()
        # Requests: every prefix that ends at a tool result or a user turn,
        # i.e. what the server actually sees. Six of them.
        cuts = [2, 4, 6, 8, 9, 10]
        self.assertGreaterEqual(len(cuts), 5)

        rendered_history = {}     # message index -> [distinct renderings, in order]
        for cut in cuts:
            result = C.compact_messages(full[:cut], s_min=2048)
            for i, message in enumerate(result.messages):
                text = render_message(message)
                seen = rendered_history.setdefault(i, [])
                if not seen or seen[-1] != text:
                    seen.append(text)

        flipped = []
        for i, versions in rendered_history.items():
            self.assertLessEqual(
                len(versions), 2,
                "message %d rendered %d distinct ways; the predicate is not "
                "monotone" % (i, len(versions)))
            if len(versions) == 2:
                flipped.append(i)
                self.assertNotIn("[compacted tool result", versions[0])
                self.assertIn("[compacted tool result", versions[1])
                self.assertEqual(full[i]["role"], "tool")
                self.assertGreaterEqual(len(full[i]["content"]), 2048)

        # Exactly the two large tool results flip; nothing else moves.
        self.assertEqual(sorted(flipped), [3, 7])

    def test_settled_never_goes_back_to_false(self):
        """The latch itself: over every growing prefix, no message's stub
        decision ever reverts. Catches predicates that rank results against
        each other (e.g. "stub the largest two"), which un-stub old bytes."""
        full = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "s" * 3000},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "b", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "m" * 9000},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "l" * 27000},
            {"role": "assistant", "content": "done"},
        ]
        previous = {}
        for cut in range(1, len(full) + 1):
            out = C.compact_messages(full[:cut], s_min=2048).messages
            for i, message in enumerate(out):
                stubbed = isinstance(message.get("content"), str) and \
                    message["content"].startswith("[compacted tool result")
                if previous.get(i):
                    self.assertTrue(
                        stubbed,
                        "message %d un-stubbed at prefix length %d" % (i, cut))
                previous[i] = stubbed

    def test_prefix_is_byte_identical_except_at_the_flip(self):
        """Turn N's stream equals turn N-1's stream with at most one message
        rewritten, and everything before and after that message untouched."""
        full = session_messages()
        cuts = [2, 4, 6, 8, 9, 10]
        previous = None
        flips = 0
        for cut in cuts:
            current = [render_message(m)
                       for m in C.compact_messages(full[:cut], s_min=2048).messages]
            if previous is not None:
                overlap = len(previous)
                differing = [i for i in range(overlap)
                             if previous[i] != current[i]]
                self.assertLessEqual(len(differing), 1, "more than one flip")
                flips += len(differing)
                # everything else is byte-identical, in place
                for i in range(overlap):
                    if i not in differing:
                        self.assertEqual(previous[i], current[i])
            previous = current
        self.assertEqual(flips, 2)      # one per large result, never repeated


# --------------------------------------------------------------------------
# 2. determinism
# --------------------------------------------------------------------------

class DeterminismTests(unittest.TestCase):
    def test_same_content_same_bytes_in_process(self):
        call = {"function": {"name": "Read",
                             "arguments": '{"file_path": "/x/a.swift"}'}}
        a = C.render_stub(BIG_A, call)
        b = C.render_stub("" + BIG_A, dict(call))
        self.assertEqual(a, b)

    def test_same_content_same_bytes_across_processes(self):
        here = os.path.dirname(os.path.abspath(__file__))
        code = (
            "import sys; sys.path.insert(0, %r); import compactor as C;"
            "big = ''.join('line %%d of the first big file\\n' %% i "
            "for i in range(400));"
            "sys.stdout.write(C.render_stub(big, {'function': {'name': 'Read',"
            " 'arguments': '{\"file_path\": \"/x/a.swift\"}'}}))" % here)
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, check=True).stdout
        call = {"function": {"name": "Read",
                             "arguments": '{"file_path": "/x/a.swift"}'}}
        self.assertEqual(out.decode("utf-8"), C.render_stub(BIG_A, call))

    def test_handle_is_content_derived_not_id_derived(self):
        m1 = {"role": "tool", "tool_call_id": "call_AAA", "content": BIG_A}
        m2 = {"role": "tool", "tool_call_id": "call_ZZZ", "content": BIG_A}
        a = C.compact_messages([m1, {"role": "assistant", "content": "x"}])
        b = C.compact_messages([m2, {"role": "assistant", "content": "x"}])
        self.assertEqual(a.messages[0]["content"], b.messages[0]["content"])

    def test_stub_carries_bytes_and_handle(self):
        stub = C.render_stub(BIG_A, None)
        self.assertIn("%d chars" % len(BIG_A), stub)
        self.assertIn(C.handle_for(BIG_A), stub)


# --------------------------------------------------------------------------
# 3. never-stub invariants
# --------------------------------------------------------------------------

class InvariantTests(unittest.TestCase):
    def test_fresh_result_is_untouched(self):
        msgs = session_messages()[:4]      # ends with the big tool result
        out = C.compact_messages(msgs).messages
        self.assertEqual(out[3]["content"], BIG_A)
        self.assertIs(out[3], msgs[3])

    def test_small_settled_result_is_untouched_forever(self):
        msgs = session_messages()
        out = C.compact_messages(msgs).messages
        self.assertEqual(out[5]["content"], SMALL)

    def test_user_and_assistant_messages_are_untouched(self):
        msgs = session_messages()
        # a user message big enough to trip the size test
        msgs.insert(1, {"role": "user", "content": BIG_A})
        out = C.compact_messages(msgs).messages
        for i, message in enumerate(msgs):
            if message["role"] in ("user", "system", "assistant"):
                self.assertEqual(render_message(out[i]), render_message(message))

    def test_disabled_config_is_a_pure_passthrough(self):
        payload = {"messages": session_messages(), "tools": [{"function": {
            "name": "Read"}}]}
        out, result = C.compact_request(payload, C.Config(enabled=False))
        self.assertIs(out, payload)
        self.assertEqual(result.stubbed, [])

    def test_no_client_tools_means_no_compaction(self):
        payload = {"messages": session_messages()}
        out, result = C.compact_request(payload, C.Config(enabled=True))
        self.assertIs(out, payload)
        self.assertEqual(result.stubbed, [])

    def test_expand_tool_is_injected_once(self):
        payload = {"messages": session_messages(),
                   "tools": [{"type": "function",
                              "function": {"name": "Read"}}]}
        out, _ = C.compact_request(payload, C.Config(enabled=True))
        names = [(t.get("function") or {}).get("name") for t in out["tools"]]
        self.assertEqual(names.count(C.EXPAND_TOOL_NAME), 1)
        again, _ = C.compact_request(out, C.Config(enabled=True))
        names = [(t.get("function") or {}).get("name") for t in again["tools"]]
        self.assertEqual(names.count(C.EXPAND_TOOL_NAME), 1)


# --------------------------------------------------------------------------
# 4. expand storage
# --------------------------------------------------------------------------

class ExpandStoreTests(unittest.TestCase):
    def test_round_trip_is_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = C.ExpandStore(tmp)
            store.put(C.handle_for(BIG_A), BIG_A)
            self.assertEqual(store.get(C.handle_for(BIG_A)), BIG_A)

    def test_survives_a_fresh_store_via_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            C.ExpandStore(tmp).put(C.handle_for(BIG_B), BIG_B)
            self.assertEqual(C.ExpandStore(tmp).get(C.handle_for(BIG_B)), BIG_B)

    def test_miss_raises_loudly(self):
        store = C.ExpandStore(None)
        with self.assertRaises(C.ExpandMiss):
            store.get("deadbeefdeadbeef")
        text = C.expand_miss_text("deadbeefdeadbeef")
        self.assertIn("expand-miss", text)
        self.assertIn("deadbeefdeadbeef", text)

    def test_memory_budget_evicts_but_disk_still_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = C.ExpandStore(tmp, budget=len(BIG_A.encode()) + 16)
            store.put(C.handle_for(BIG_A), BIG_A)
            store.put(C.handle_for(BIG_B), BIG_B)
            self.assertEqual(store.get(C.handle_for(BIG_A)), BIG_A)


# --------------------------------------------------------------------------
# 5. the proxy, against a stub upstream
# --------------------------------------------------------------------------

SSE_PLAIN = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n')

SSE_EXPAND = (
    b'data: {"choices":[{"delta":{"role":"assistant","tool_calls":['
    b'{"index":0,"id":"e1","type":"function","function":'
    b'{"name":"expand","arguments":""}}]}}]}\n\n'
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
    b'{"arguments":"{\\"handle\\": \\"HANDLE\\"}"}}]}}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    b'data: [DONE]\n\n')


class StubUpstream:
    """Records requests; replies with whatever the test queued."""

    def __init__(self):
        self.requests = []
        self.replies = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):   # match the base signature
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                outer.requests.append(json.loads(body))
                kind, payload = outer.replies.pop(0)
                if kind == "sse":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                    self.close_connection = True
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server.server_port

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.upstream = StubUpstream()
        self.tmp = tempfile.TemporaryDirectory()
        self.config = C.Config(
            enabled=True, s_min=2048, expand_enabled=True,
            upstream=self.upstream.url, host="127.0.0.1", port=0,
            cache_dir=self.tmp.name, expand_budget=1 << 20,
            expand_max_rounds=4)
        C.Handler.config = self.config
        C.Handler.store = C.ExpandStore(self.tmp.name, 1 << 20)
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), C.Handler)
        self.proxy.daemon_threads = True
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()

    def tearDown(self):
        self.proxy.shutdown()
        self.proxy.server_close()
        self.upstream.stop()
        self.tmp.cleanup()

    def post(self, payload):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.server_port, timeout=30)
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return response.status, body

    def request_payload(self, messages=None):
        return {
            "model": "qwen",
            "stream": True,
            "messages": messages or session_messages(),
            "tools": [{"type": "function", "function": {"name": "Read"}}],
        }

    def test_sse_passthrough_is_byte_identical(self):
        self.upstream.replies.append(("sse", SSE_PLAIN))
        status, body = self.post(self.request_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body, SSE_PLAIN)

    def test_upstream_sees_the_compacted_array(self):
        self.upstream.replies.append(("sse", SSE_PLAIN))
        self.post(self.request_payload())
        sent = self.upstream.requests[0]["messages"]
        self.assertIn("[compacted tool result", sent[3]["content"])
        self.assertEqual(sent[5]["content"], SMALL)       # small: raw
        self.assertIn("[compacted tool result", sent[7]["content"])
        self.assertEqual(sent[1]["content"], "Fix the parser.")

    def test_expand_is_intercepted_and_round_trips_verbatim(self):
        handle = C.handle_for(BIG_A)
        self.upstream.replies.append(
            ("sse", SSE_EXPAND.replace(b"HANDLE", handle.encode())))
        self.upstream.replies.append(("sse", SSE_PLAIN))
        status, body = self.post(self.request_payload())
        self.assertEqual(status, 200)
        # The client never sees the expand turn.
        self.assertEqual(body, SSE_PLAIN)
        # The second upstream request carries the original text, verbatim.
        second = self.upstream.requests[1]["messages"]
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["content"], BIG_A)
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertEqual(
            second[-2]["tool_calls"][0]["function"]["name"], "expand")

    def test_expand_miss_is_loud_and_not_silent(self):
        self.upstream.replies.append(
            ("sse", SSE_EXPAND.replace(b"HANDLE", b"0123456789abcdef")))
        self.upstream.replies.append(("sse", SSE_PLAIN))
        status, body = self.post(self.request_payload())
        self.assertEqual(status, 200)
        second = self.upstream.requests[1]["messages"]
        self.assertIn("expand-miss", second[-1]["content"])
        self.assertIn("0123456789abcdef", second[-1]["content"])

    def test_non_streaming_response_is_relayed(self):
        reply = json.dumps({"choices": [
            {"message": {"role": "assistant", "content": "hi"}}]}).encode()
        self.upstream.replies.append(("json", reply))
        payload = self.request_payload()
        payload["stream"] = False
        status, body = self.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"],
                         "hi")

    def test_expand_disabled_leaves_the_stream_alone(self):
        self.config.expand_enabled = False
        C.Handler.store = C.ExpandStore(None)
        self.upstream.replies.append(
            ("sse", SSE_EXPAND.replace(b"HANDLE", b"0123456789abcdef")))
        status, body = self.post(self.request_payload())
        self.assertEqual(status, 200)
        self.assertEqual(
            body, SSE_EXPAND.replace(b"HANDLE", b"0123456789abcdef"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
