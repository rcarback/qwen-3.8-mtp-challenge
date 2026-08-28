import Foundation
import MLXFastCore
import Tokenizers

// An OpenAI-compatible HTTP front end for the native-MTP decode session.
//
// LOCAL DEVELOPER TOOLING. Nothing here is on a scored path and it lives outside
// `editablePaths`, so `yukon submit` cannot package it. It exists so external
// harnesses can drive the same worker the ranked benchmark drives, on real
// workloads, and read the accept rate the run actually produced.
//
// SAMPLING. `temperature`, `top_p`, and `seed` are carried to the session's
// `mtp_decode_begin`. Absent or zero temperature selects greedy decode, which is
// the ranked path and stays byte-identical. `top_k`, `presence_penalty`, and
// `frequency_penalty` are not honoured and are not decoded. `n > 1` IS rejected,
// because silently returning one choice where the caller asked for four is a
// wrong answer rather than a missing knob.
//
// LOOPBACK ONLY, NO AUTHENTICATION. One request at a time: the worker holds one
// ~14 GiB model and one KV session.
extension QwenRuntime {
    public struct QwenServeOptions: Sendable {
        public let targetWeightsPath: String
        public let mtpHeadPath: String
        public let depth: Int
        public let maxNewTokens: Int
        public let port: UInt16
        public let modelName: String

        public init(
            targetWeightsPath: String,
            mtpHeadPath: String,
            depth: Int,
            maxNewTokens: Int,
            port: UInt16,
            modelName: String
        ) {
            self.targetWeightsPath = targetWeightsPath
            self.mtpHeadPath = mtpHeadPath
            self.depth = depth
            self.maxNewTokens = maxNewTokens
            self.port = port
            self.modelName = modelName
        }
    }

    public static func qwenServe(
        options: QwenServeOptions,
        workerOptions: RuntimeWorkerOptions
    ) throws {
        guard options.depth >= MLXFastConstants.qwenMTPSerialControlDepth,
              options.depth <= MLXFastConstants.qwenMTPMaxDraftDepth
        else {
            throw MLXFastError.invalidInput(
                "--mtp-depth must be between "
                    + "\(MLXFastConstants.qwenMTPSerialControlDepth) and "
                    + "\(MLXFastConstants.qwenMTPMaxDraftDepth) "
                    + "(0 is the serial control: MTP off)")
        }
        guard options.maxNewTokens > 0 else {
            throw MLXFastError.invalidInput("--max-tokens must be positive")
        }

        let tokenizer = try loadLocalTokenizer(at: options.targetWeightsPath)
        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            mtpHeadPath: options.mtpHeadPath
        )
        defer { client.close() }

        serveNote("loading the target and MTP head...")
        let warm = try client.warmMTPDecode()
        guard warm.ok else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the untimed warm: "
                    + (warm.error ?? "no reason reported"))
        }

        // One worker, one KV session: every generation runs to completion before
        // the next begins. Two interleaved `mtp_decode_round` streams on one
        // session would corrupt both.
        let generationQueue = DispatchQueue(label: "mlxfast.serve.generate")
        let context = ServeContext(client: client, tokenizer: tokenizer)

        let server = try MinimalHTTPServer(port: options.port)
        try server.start { request, responder in
            switch (request.method, request.path) {
            case ("GET", "/v1/models"), ("GET", "/models"):
                respondWithModelList(options: options, responder: responder)
            case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
                generationQueue.async {
                    handleCompletion(
                        request: request, responder: responder,
                        context: context, options: options)
                }
            case ("GET", "/health"):
                responder.sendJSON(status: 200, body: Data(#"{"ok":true}"#.utf8))
            default:
                responder.sendError(
                    status: 404,
                    message: "no route for \(request.method) \(request.path)")
            }
        }

        let depthNote = options.depth == MLXFastConstants
            .qwenMTPSerialControlDepth ? " (serial control: MTP off)" : ""
        serveNote("""
            listening on http://127.0.0.1:\(options.port)/v1
              model name       \(options.modelName)
              draft depth      \(options.depth)\(depthNote)
              max new tokens   \(options.maxNewTokens)
              sampling         greedy unless the request sets temperature > 0
            """)
        server.waitForever()
    }

    /// Everything the generation path owns: the worker, its tokenizer, and the
    /// token history of the session the worker currently holds.
    ///
    /// `@unchecked Sendable`: `RuntimeWorkerClient` and `Tokenizer` are not
    /// `Sendable`, and neither is touched anywhere except on the single serial
    /// `mlxfast.serve.generate` queue created in `qwenServe`. The accept queue
    /// only forwards this reference; it never dereferences it.
    final class ServeContext: @unchecked Sendable {
        let client: RuntimeWorkerClient
        let tokenizer: any Tokenizer

        /// The seed the worker's live session holds plus everything it emitted.
        /// A new request is compared against this to decide whether it can
        /// extend the session or has to restart it. Guarded by a lock even
        /// though every mutation happens on the generation queue, so a future
        /// reader on another queue cannot race silently.
        private let lock = NSLock()
        private var history: [Int] = []

        init(client: RuntimeWorkerClient, tokenizer: any Tokenizer) {
            self.client = client
            self.tokenizer = tokenizer
        }

        func decide(for incoming: [Int]) -> ServePrefixDecision {
            lock.lock()
            defer { lock.unlock() }
            return ServePrefixDecision.make(previous: history, incoming: incoming)
        }

        func record(prompt: [Int], emitted: [Int]) {
            lock.lock()
            defer { lock.unlock() }
            history = prompt + emitted
        }

        /// A failed request leaves the worker's session in an unknown state, so
        /// the next request must not try to extend it.
        func invalidate() {
            lock.lock()
            defer { lock.unlock() }
            history = []
        }
    }

    /// Bucket key for the worker's resume-point store: a hash of the leading
    /// tokens, which a rewind or an edited tail does not disturb.
    /// Token offsets for the given CHARACTER offsets, by tokenizing each
    /// prefix.
    ///
    /// Reports every turn boundary it finds and applies no policy. How many of
    /// these deserve a checkpoint depends on the chunk size, which is the
    /// worker's to know: the trusted binary links no model code, and pushing
    /// the decision down keeps it that way.
    ///
    /// The final offset is dropped. It sits at the end of the prompt, where a
    /// checkpoint would leave no row to read the next token from.
    static func tokenBoundaries(
        forCharacterOffsets offsets: [Int], in prompt: String,
        tokenizer: any Tokenizer, totalTokens: Int
    ) -> [Int] {
        var boundaries: [Int] = []
        for offset in offsets where offset > 0 && offset < prompt.count {
            let index = prompt.index(prompt.startIndex, offsetBy: offset)
            let count = tokenizer.encode(
                text: String(prompt[prompt.startIndex ..< index]),
                addSpecialTokens: false).count
            // Ordered so the `>` comparison comes first: written the other
            // way, `< totalTokens, count >` parses as a generic argument list.
            guard count > (boundaries.last ?? 0), count < totalTokens else {
                continue
            }
            boundaries.append(count)
        }
        return boundaries
    }

    static func conversationKey(for tokens: [Int]) -> String {
        var hasher = Hasher()
        for token in tokens.prefix(128) { hasher.combine(token) }
        return String(UInt(bitPattern: hasher.finalize()), radix: 36)
    }

    private static func respondWithModelList(
        options: QwenServeOptions,
        responder: HTTPResponder
    ) {
        let payload = ModelListResponse(data: [
            .init(
                id: options.modelName,
                created: Int(Date().timeIntervalSince1970),
                ownedBy: "local"
            ),
        ])
        guard let body = try? JSONEncoder().encode(payload) else {
            responder.sendError(
                status: 500, message: "could not encode the model list")
            return
        }
        responder.sendJSON(status: 200, body: body)
    }

    private static func handleCompletion(
        request: HTTPRequest,
        responder: HTTPResponder,
        context: ServeContext,
        options: QwenServeOptions
    ) {
        let decoded: ChatCompletionRequest
        do {
            decoded = try JSONDecoder().decode(
                ChatCompletionRequest.self, from: request.body)
        } catch {
            responder.sendError(
                status: 400, message: "could not decode the request body: \(error)")
            return
        }
        if let choices = decoded.n, choices != 1 {
            responder.sendError(
                status: 400,
                message: "this server decodes one continuation per request; "
                    + "n=\(choices) is not supported")
            return
        }

        // Tools have to survive the trip in original key order, which
        // `JSONDecoder` cannot promise, so they are re-parsed from the raw body.
        let tools = parseToolsFromRawBody(request.body)

        let prompt: String
        let turnEnds: [Int]
        do {
            (prompt, turnEnds) = try OpenAIPromptRendering
                .renderPromptWithTurnBoundaries(
                    messages: decoded.messages, tools: tools)
        } catch {
            responder.sendError(status: 400, message: "\(error)")
            return
        }

        var budget = decoded.maxTokens ?? options.maxNewTokens
        if budget > options.maxNewTokens {
            serveNote("clamping max_tokens \(budget) to \(options.maxNewTokens)")
            budget = options.maxNewTokens
        }
        budget = max(1, budget)

        let seedTokens = context.tokenizer.encode(
            text: prompt, addSpecialTokens: false)
        let turnBoundaries = Self.tokenBoundaries(
            forCharacterOffsets: turnEnds, in: prompt,
            tokenizer: context.tokenizer, totalTokens: seedTokens.count)
        let streaming = decoded.stream ?? false
        let completionID = "chatcmpl-" + UUID().uuidString
            .replacingOccurrences(of: "-", with: "").prefix(24)
        let created = Int(Date().timeIntervalSince1970)

        do {
            // PREFIX REUSE. An agent turn is almost always the previous
            // conversation plus a tool result plus a new message -- a strict
            // extension of what the worker already holds. Forwarding only the
            // tail turns a full re-prefill into a short one. Anything that is
            // not a strict extension restarts, because the session's recurrent
            // layers cannot rewind (see `ServePrefixDecision`).
            let decision = context.decide(for: seedTokens)
            switch decision {
            case .extend(let tail):
                serveNote(
                    "reusing \(seedTokens.count - tail.count) cached tokens, "
                        + "prefilling \(tail.count)")
            case .restart:
                context.invalidate()
                let reset = try context.client.resetMTPDecode()
                guard reset.ok else {
                    throw MLXFastError.invalidInput(
                        "the MTP worker refused the session reset: "
                            + (reset.error ?? "no reason reported"))
                }
            }

            let outcome = try generate(
                context: context,
                seedTokens: seedTokens,
                turnBoundaries: turnBoundaries,
                decision: decision,
                sampling: samplingFor(decoded),
                budget: budget,
                stopStrings: decoded.stop?.values ?? [],
                depth: options.depth,
                tools: tools,
                onDelta: streaming
                    ? { delta in
                        emitChunk(
                            responder: responder, id: String(completionID),
                            created: created, model: options.modelName,
                            delta: .init(role: nil, content: delta, toolCalls: nil),
                            finishReason: nil, openStream: true)
                    }
                    : nil
            )

            context.record(prompt: seedTokens, emitted: outcome.emittedTokens)
            // Refresh this conversation's resume point to the END of the turn.
            // Recorded only at `begin`, a resume point pins to the prompt
            // boundary and a later switch replays the whole reply; refreshed
            // here, the tail a switch must replay is whatever arrives next.
            //
            // Best-effort by design: a failed snapshot costs a slower resume,
            // never a wrong answer, so it must not fail the request that just
            // succeeded.
            let history = seedTokens + outcome.emittedTokens
            if let snapshot = try? context.client.snapshotMTPDecode(
                conversationId: Self.conversationKey(for: seedTokens),
                tokens: history), snapshot.ok
            {
                serveNote("recorded resume point at \(history.count) tokens")
            }
            FileHandle.standardError.write(Data(
                (renderStatsBar(outcome.stats, depth: options.depth) + "\n").utf8))

            if streaming {
                finishStream(
                    responder: responder, id: String(completionID),
                    created: created, model: options.modelName, outcome: outcome)
            } else {
                sendCompletion(
                    responder: responder, id: String(completionID),
                    created: created, model: options.modelName,
                    promptTokens: seedTokens.count, outcome: outcome,
                    depth: options.depth)
            }
        } catch {
            // The worker's session state is now unknown, so the next request
            // must not try to extend it.
            context.invalidate()
            // Once SSE headers are out the only honest signal left is an abrupt
            // end of stream; before that a 500 still reaches the client.
            // Before the headers are out a real 500 still reaches the client,
            // and the error text is the only way the fault is visible at all.
            // Only an already-open stream has to end abruptly.
            if streaming, responder.didSendHeaders {
                responder.endSSE()
            } else {
                responder.sendError(status: 500, message: "\(error)")
            }
            serveNote("request failed: \(error)")
        }
    }

    struct ServeOutcome {
        var text: String
        var toolCalls: [ToolCallPayload]
        var finishReason: String
        var stats: QwenChatTurnStats
        /// Needed by `ServeContext` so the next request can tell whether it
        /// extends this one.
        var emittedTokens: [Int]
    }

    /// Read the sampling knobs off the request. Absent or zero temperature is
    /// greedy, which is the session's untouched argmax path.
    private static func samplingFor(
        _ request: ChatCompletionRequest
    ) -> (temperature: Double, topP: Double, seed: UInt64?)? {
        guard let temperature = request.temperature, temperature > 0 else {
            return nil
        }
        return (temperature, request.topP ?? 1.0, request.seed)
    }

    private static func generate(
        context: ServeContext,
        seedTokens: [Int],
        turnBoundaries: [Int],
        decision: ServePrefixDecision,
        sampling: (temperature: Double, topP: Double, seed: UInt64?)?,
        budget: Int,
        stopStrings: [String],
        depth: Int,
        tools: [OrderedJSON]?,
        onDelta: ((String) -> Void)?
    ) throws -> ServeOutcome {
        let tokenizer = context.tokenizer
        var stats = QwenChatTurnStats()
        let started = Date()
        let begin: RuntimeWorkerResponse
        switch decision {
        case .extend(let tail):
            // The session already carries the shared prefix; only the tail is
            // forwarded. Sampling policy was set at the original `begin` and
            // persists for the life of the session.
            begin = try context.client.extendMTPDecode(tokens: tail)
        case .restart:
            // A restart is the expensive path -- a full re-prefill, ~200-300 s
            // at 20k. Hand the worker a conversation key so it can consult its
            // resume-point store first: an interrupt, a failed request, or a
            // rewound message then costs the tail, not the prompt.
            //
            // The key is a stable PREFIX hash, not an identity. A collision is
            // harmless by construction: the store re-verifies the full token
            // prefix before resuming, so a wrong bucket simply misses and falls
            // through to the ordinary prefill.
            begin = try context.client.beginMTPDecode(
                seedTokens: seedTokens,
                temperature: sampling?.temperature,
                topP: sampling?.topP,
                seed: sampling?.seed,
                conversationId: Self.conversationKey(for: seedTokens),
                turnBoundaries: turnBoundaries)
        }
        // Make the resume-point store VISIBLE. Without this a cache hit is
        // indistinguishable from a lucky fast prefill, which is exactly how a
        // cache silently stops working and nobody notices.
        if let resumed = begin.resumedTokens {
            serveNote(
                "resumed \(resumed) tokens from the session store, "
                    + "prefilled \(seedTokens.count - resumed)")
        } else if case .restart = decision {
            serveNote(
                "no resume point: prefilling all \(seedTokens.count) tokens")
        }
        stats.seedPrefillSeconds = Date().timeIntervalSince(started)
        guard begin.ok, begin.seedToken != nil else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the seed prefill: "
                    + (begin.error ?? "no seed token returned"))
        }

        // `emitted` starts EMPTY even though begin returned a seed argmax. That
        // token is the reply's first token, but the session holds it as the
        // pending primary and commits it at the top of round 1 -- so it arrives
        // in round 1's `tokens` and appending it here would emit it twice.
        var emitted: [Int] = []
        var full = ""
        var gate = OpenAIPromptRendering.ToolCallGate()
        var finishReason = "stop"
        var done = false
        // Local diagnostic: the stats bar reports how many tokens a turn
        // emitted but not why it stopped, and EOS, a stop string and an empty
        // round are indistinguishable from the outside. Name the branch.
        var exitCause = "budget-or-loop-end"

        while !done, emitted.count < budget {
            let response = try context.client.mtpDecodeRound(depth: depth)
            guard response.ok, let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "the MTP worker failed a decode round: "
                        + (response.error ?? "no tokens returned"))
            }
            stats.rounds += 1
            stats.acceptedDrafts += response.acceptedDraftCount ?? 0
            stats.rejectedDrafts += response.rejectedDraftCount ?? 0

            for token in tokens {
                if let eos = tokenizer.eosTokenId, token == eos {
                    exitCause = "eos"
                    done = true
                    break
                }
                emitted.append(token)
                if emitted.count >= budget {
                    exitCause = "budget"
                    finishReason = "length"
                    done = true
                    break
                }
            }

            // Decode the whole prefix each round: Qwen uses byte-level BPE, so a
            // token can carry a fragment of a multi-byte character and decoding
            // tokens singly produces replacement characters at the seams.
            full = tokenizer.decode(tokens: emitted, skipSpecialTokens: true)

            if let stop = firstStopHit(in: full, stopStrings: stopStrings) {
                exitCause = "stop-string"
                full = String(full[full.startIndex..<stop])
                done = true
            }

            let admitted = gate.admit(full)
            if let onDelta, !admitted.delta.isEmpty {
                onDelta(admitted.delta)
            }
            if admitted.sawToolCall { finishReason = "tool_calls" }
            if tokens.isEmpty {
                exitCause = "empty-round"
                break
            }
        }

        stats.seconds = Date().timeIntervalSince(started)
        stats.emittedTokens = emitted.count

        let calls = OpenAIPromptRendering.parseToolCalls(full, tools: tools)
        if calls.isEmpty {
            // The gate may have stopped on a `<tool_call>` the model never
            // closed. Reporting `tool_calls` with no calls would strand a
            // harness waiting for a call it can never run.
            if finishReason == "tool_calls" { finishReason = "stop" }
        } else {
            finishReason = "tool_calls"
        }

        let visible = calls.isEmpty
            ? full
            : String(full[full.startIndex..<(
                full.range(of: OpenAIPromptRendering.ToolCallGate.marker)?
                    .lowerBound ?? full.endIndex)])

        serveNote(
            "turn ended: cause=\(exitCause) finish=\(finishReason) "
                + "emitted=\(emitted.count) rounds=\(stats.rounds) "
                + "budget=\(budget) stops=\(stopStrings.count) "
                + "raw=\(full.count)ch visible=\(visible.count)ch "
                + "calls=\(calls.count) head=\(String(full.prefix(120)).debugDescription)")
        return ServeOutcome(
            text: visible,
            toolCalls: calls,
            finishReason: finishReason,
            stats: stats,
            emittedTokens: emitted)
    }

    private static func firstStopHit(
        in text: String, stopStrings: [String]
    ) -> String.Index? {
        var earliest: String.Index?
        for stop in stopStrings where !stop.isEmpty {
            if let found = text.range(of: stop)?.lowerBound {
                earliest = earliest.map { Swift.min($0, found) } ?? found
            }
        }
        return earliest
    }

    private static func emitChunk(
        responder: HTTPResponder,
        id: String,
        created: Int,
        model: String,
        delta: ChatCompletionChunk.Delta,
        finishReason: String?,
        openStream: Bool
    ) {
        if openStream { responder.beginSSE() }
        let chunk = ChatCompletionChunk(
            id: id, created: created, model: model,
            choices: [.init(index: 0, delta: delta, finishReason: finishReason)])
        guard let body = try? JSONEncoder().encode(chunk) else { return }
        responder.sendSSE(body)
    }

    private static func finishStream(
        responder: HTTPResponder,
        id: String,
        created: Int,
        model: String,
        outcome: ServeOutcome
    ) {
        // Idempotent: a stream that already emitted a delta has its headers out.
        responder.beginSSE()
        if !outcome.toolCalls.isEmpty {
            let streamed = outcome.toolCalls.enumerated().map {
                ChatCompletionChunk.StreamedToolCall(
                    index: $0.offset, id: $0.element.id,
                    type: $0.element.type, function: $0.element.function)
            }
            emitChunk(
                responder: responder, id: id, created: created, model: model,
                delta: .init(role: nil, content: nil, toolCalls: streamed),
                finishReason: nil, openStream: false)
        }
        emitChunk(
            responder: responder, id: id, created: created, model: model,
            delta: .init(role: nil, content: nil, toolCalls: nil),
            finishReason: outcome.finishReason, openStream: false)
        responder.endSSE()
    }

    private static func sendCompletion(
        responder: HTTPResponder,
        id: String,
        created: Int,
        model: String,
        promptTokens: Int,
        outcome: ServeOutcome,
        depth: Int
    ) {
        let response = ChatCompletionResponse(
            id: id,
            created: created,
            model: model,
            choices: [.init(
                index: 0,
                message: .init(
                    content: outcome.text.isEmpty ? nil : outcome.text,
                    toolCalls: outcome.toolCalls.isEmpty
                        ? nil : outcome.toolCalls),
                finishReason: outcome.finishReason)],
            usage: .init(
                promptTokens: promptTokens,
                completionTokens: outcome.stats.emittedTokens,
                totalTokens: promptTokens + outcome.stats.emittedTokens,
                mtp: .init(
                    offeredDepth: depth,
                    rounds: outcome.stats.rounds,
                    acceptedDrafts: outcome.stats.acceptedDrafts,
                    rejectedDrafts: outcome.stats.rejectedDrafts,
                    acceptRate: outcome.stats.acceptRate,
                    effectiveDraftDepth: outcome.stats.meanDraftDepth,
                    decodeTokensPerSecond: outcome.stats.decodeTokensPerSecond,
                    seedPrefillSeconds: outcome.stats.seedPrefillSeconds)))
        guard let body = try? JSONEncoder().encode(response) else {
            responder.sendError(
                status: 500, message: "could not encode the response")
            return
        }
        responder.sendJSON(status: 200, body: body)
    }

    /// Re-parse `tools` straight from the request bytes so object key order
    /// survives. See `OrderedJSON` for why that matters to the prompt.
    private static func parseToolsFromRawBody(_ body: Data) -> [OrderedJSON]? {
        guard let text = String(data: body, encoding: .utf8),
              let root = try? OrderedJSON.parse(text),
              case .array(let items)? = root["tools"],
              !items.isEmpty
        else { return nil }
        return items
    }

    private static func serveNote(_ message: String) {
        FileHandle.standardError.write(
            Data("mlxfast-swift serve: \(message)\n".utf8))
    }
}
