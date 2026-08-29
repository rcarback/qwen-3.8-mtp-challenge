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
        let context = ServeContext(
            client: client,
            tokenizer: tokenizer,
            stopTokens: Self.serveStopTokens(
                directory: URL(fileURLWithPath: options.targetWeightsPath),
                tokenizer: tokenizer))

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
        /// EVERY id that terminates a turn, not just `tokenizer.eosTokenId`.
        /// `generation_config.json` lists two for this model (248046
        /// <|im_end|> and 248044 <|endoftext|>) and the worker's accept walk
        /// already stops on the full set, so a parent that recognised one was
        /// letting a committed 248044 run the turn on to its token budget.
        let stopTokens: Set<Int>

        /// The seed the worker's live session holds plus everything it emitted.
        /// A new request is compared against this to decide whether it can
        /// extend the session or has to restart it. Guarded by a lock even
        /// though every mutation happens on the generation queue, so a future
        /// reader on another queue cannot race silently.
        private let lock = NSLock()
        private var history: [Int] = []

        init(
            client: RuntimeWorkerClient,
            tokenizer: any Tokenizer,
            stopTokens: Set<Int>
        ) {
            self.client = client
            self.tokenizer = tokenizer
            self.stopTokens = stopTokens
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
            // The session consumed the stop token too, so the resume point
            // has to describe it. This is also what makes the array matchable:
            // the chat template writes the same terminator into the next
            // turn's prompt, so seed + emitted + [terminator] is a genuine
            // prefix of what arrives next.
            let history = seedTokens + outcome.emittedTokens
                + (outcome.terminalToken.map { [$0] } ?? [])
            // Three outcomes, three distinct log lines, because they mean
            // different things and the old form collapsed all of them into
            // silence. `ok == false` is the worker REFUSING to file a
            // checkpoint whose snapshot does not describe the array handed
            // with it; a nil result is the snapshot call itself failing.
            // Distinguishing them matters: a guard that refused everything
            // and a guard that never fired both produce zero errors, and
            // only the log tells them apart.
            //
            // Logged parent-side because worker stderr forwarding is BROKEN.
            // Proven, not inferred: each parent-side "resumed N tokens from
            // the session store" line is a witness that a worker stderr write
            // executed, because every `resumedTokens` assignment in
            // mtp_decode_begin (:801/:817/:841/:897) is immediately followed
            // by that write with no branch between, and that local is the
            // only feed for the field at :951. Across the serve logs: 51
            // witnesses, 0 forwarded "mlxfast-worker: " lines, 0 redacted
            // "token-validation-failed" lines. So the branches ran 51 times
            // and forwarded nothing.
            // Keyed on the FILED COUNT, not on `ok`. An earlier version
            // branched on `snapshot.ok` and its refusal arm was DEAD CODE:
            // `send()` throws on !ok, so an ok:false refusal never returned as
            // a value and the parent logged it as a call failure. The worker
            // now reports a refusal as ok:true having filed zero tokens --
            // a refusal is a normal outcome, not a protocol failure -- so the
            // three cases are distinguishable and the failure label means
            // only what it says.
            if let snapshot = try? context.client.snapshotMTPDecode(
                conversationId: Self.conversationKey(for: seedTokens),
                tokens: history)
            {
                if (snapshot.resumedTokens ?? 0) > 0 {
                    serveNote("recorded resume point at \(history.count) tokens")
                } else {
                    serveNote(
                        "skipped resume point at \(history.count) tokens "
                        + "(worker refused: snapshot does not describe this array)")
                }
            } else {
                serveNote(
                    "skipped resume point at \(history.count) tokens "
                    + "(snapshot call failed)")
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
        /// The stop token the session COMMITTED and the client never saw.
        ///
        /// The emit loop stops at a stop token without emitting it, so the
        /// session's state runs one token ahead of `emittedTokens`. A resume
        /// point has to describe the state, so the filed array needs this
        /// token appended -- and appending it also makes the array MORE
        /// matchable, because the chat template writes the same terminator
        /// into the next turn's prompt.
        var terminalToken: Int?
    }

    /// The serve round loop's text stage: decode the reply so far, test it
    /// against the stop strings, and hand the caller whatever is safe to
    /// stream.
    ///
    /// It exists as a type so it can be driven by a fake decoder in a test.
    /// The loop it came from could only be exercised with a real tokenizer, a
    /// real worker and a real model, which is why the quadratic cost inside it
    /// went unmeasured for so long.
    struct ServeRoundText {
        struct Outcome {
            var full: String
            var delta: String
            var hitStop: Bool
            var sawToolCall: Bool
            var decodeSeconds: Double
            var stopSeconds: Double
            var gateSeconds: Double
        }

        let stopStrings: [String]
        let streaming: Bool
        private var gate = OpenAIPromptRendering.ToolCallGate()

        /// The decoded text has exactly two consumers: the stop-string scan
        /// and the streaming delta. With neither present nothing reads it
        /// until the turn ends, so the whole stage collapses to one call at
        /// the end. That is exact, not an approximation: the final string is
        /// a decode of the same token array either way, `firstStopHit` is
        /// unreachable with no stop strings, and the gate's only remaining
        /// effect is `sawToolCall`, which latches identically whether it sees
        /// the reply once or in pieces.
        var runsPerRound: Bool { streaming || !stopStrings.isEmpty }

        private var lastFull = ""
        private var ranAtLeastOnce = false

        init(stopStrings: [String], streaming: Bool) {
            self.stopStrings = stopStrings
            self.streaming = streaming
        }

        /// `decode` yields the WHOLE reply so far, not an increment: Qwen uses
        /// byte-level byte-pair encoding, so a token can carry a fragment of a
        /// multi-byte character and decoding tokens singly produces
        /// replacement characters at the seams.
        mutating func advance(decode: () -> String) -> Outcome {
            guard runsPerRound else {
                return Outcome(
                    full: lastFull, delta: "", hitStop: false,
                    sawToolCall: false, decodeSeconds: 0, stopSeconds: 0,
                    gateSeconds: 0)
            }
            ranAtLeastOnce = true
            let decodeStarted = Date()
            var full = decode()
            let decodeSeconds = Date().timeIntervalSince(decodeStarted)

            let stopStarted = Date()
            let stopHit = QwenRuntime.firstStopHit(
                in: full, stopStrings: stopStrings)
            let stopSeconds = Date().timeIntervalSince(stopStarted)
            if let stop = stopHit {
                full = String(full[full.startIndex ..< stop])
            }

            let gateStarted = Date()
            let admitted = gate.admit(full)
            let gateSeconds = Date().timeIntervalSince(gateStarted)

            lastFull = full
            return Outcome(
                full: full,
                delta: streaming ? admitted.delta : "",
                hitStop: stopHit != nil,
                sawToolCall: admitted.sawToolCall,
                decodeSeconds: decodeSeconds,
                stopSeconds: stopSeconds,
                gateSeconds: gateSeconds)
        }

        /// One last pass for a stage that skipped the loop. A stage that ran
        /// per round already holds its final text and re-running the gate on
        /// it would be a second sighting of the same marker.
        mutating func finish(decode: () -> String) -> Outcome {
            guard !ranAtLeastOnce else {
                return Outcome(
                    full: lastFull, delta: "", hitStop: false,
                    sawToolCall: gate.stopped, decodeSeconds: 0,
                    stopSeconds: 0, gateSeconds: 0)
            }
            ranAtLeastOnce = true
            let decodeStarted = Date()
            let full = decode()
            let decodeSeconds = Date().timeIntervalSince(decodeStarted)
            let gateStarted = Date()
            let admitted = gate.admit(full)
            let gateSeconds = Date().timeIntervalSince(gateStarted)
            lastFull = full
            return Outcome(
                full: full, delta: "", hitStop: false,
                sawToolCall: admitted.sawToolCall,
                decodeSeconds: decodeSeconds, stopSeconds: 0,
                gateSeconds: gateSeconds)
        }
    }

    /// Whole-prefix decode, computed from a bounded token suffix.
    ///
    /// The whole-prefix decode at `QwenRuntimeServe`'s round loop is quadratic
    /// in the reply length. It cannot become a per-token decode, because Qwen
    /// uses byte-level byte-pair encoding and a character can straddle tokens.
    /// It CAN become a suffix decode plus a splice: re-decode the last
    /// `window` tokens each round and freeze everything before that as
    /// `committedText`, so only a bounded tail is ever re-decoded. The
    /// default `window` is sixteen.
    ///
    /// `window` alone is a MARGIN, not a proof: a character spans at most
    /// four bytes, so a large-enough window makes it likely that
    /// `tokens.count - window` lands after any in-progress character, but
    /// "likely" is not exact -- a boundary can still fall inside a
    /// multi-byte sequence for any window size, including the default.
    /// CORRECTNESS instead comes from checking the candidate commit itself:
    /// decoding a slice that ends mid-character renders a trailing
    /// replacement character (U+FFFD), so a commit is only ever frozen when
    /// its decode does not end in one (see `text(for:decode:)`). A rejected
    /// candidate is not an error -- the pending decode below still spans the
    /// whole reply, so the round costs a larger-than-ideal re-decode rather
    /// than a wrong answer, and the next call (a longer `tokens`) tries
    /// again at a larger boundary.
    ///
    /// The committed prefix is only ever extended by the part of a suffix
    /// decode that a LATER decode can no longer change, which is everything
    /// except the last `window` tokens' worth of text. Nothing is committed
    /// until it is out of reach of a seam repair.
    struct IncrementalDetokenizer {
        let window: Int
        /// Text for `tokens[0 ..< committedTokens]`, known stable.
        private var committedText = ""
        private var committedTokens = 0

        init(window: Int = 16) {
            self.window = Swift.max(1, window)
        }

        mutating func text(
            for tokens: [Int],
            decode: (ArraySlice<Int>) -> String
        ) -> String {
            guard !tokens.isEmpty else { return "" }
            // Everything from `committedTokens` on is re-decoded each call, so
            // the committed point must never advance past `tokens.count -
            // window`; beyond that a later token could still repair a seam.
            let safeCommit = Swift.max(0, tokens.count - window)
            if safeCommit > committedTokens {
                let candidate = decode(tokens[0 ..< safeCommit])
                // A token-count margin alone does not guarantee `safeCommit`
                // lands on a character boundary: it only bounds how far a
                // FUTURE token can still repair a seam, not whether THIS
                // slice, decoded alone, already ends mid-character. Decoding
                // an incomplete trailing byte sequence renders one or more
                // trailing replacement characters (the same signal
                // `QwenRuntimeServe.generate`'s round loop comment already
                // relies on above), so a trailing replacement character
                // means this candidate is not yet safe to freeze. Skip the
                // commit and retry at the next call's larger boundary rather
                // than caching a wrong interpretation permanently -- the
                // pending decode below still covers the full reply either
                // way, so this only costs extra re-decoding, never a wrong
                // answer.
                if !candidate.hasSuffix("\u{FFFD}") {
                    committedText = candidate
                    committedTokens = safeCommit
                }
            }
            return committedText + decode(tokens[committedTokens...])
        }
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
        var stage = ServeRoundText(
            stopStrings: stopStrings, streaming: onDelta != nil)
        var detokenizer = IncrementalDetokenizer()
        var finishReason = "stop"
        var done = false
        // Local diagnostic: the stats bar reports how many tokens a turn
        // emitted but not why it stopped, and EOS, a stop string and an empty
        // round are indistinguishable from the outside. Name the branch.
        var exitCause = "budget-or-loop-end"
        // Set only when the loop stops ON a stop token. The session committed
        // it; the client never sees it; the resume point must carry it.
        var terminalToken: Int?

        while !done, emitted.count < budget {
            let roundStarted = Date()
            let response = try context.client.mtpDecodeRound(depth: depth)
            stats.workerRoundSeconds += Date().timeIntervalSince(roundStarted)
            guard response.ok, let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "the MTP worker failed a decode round: "
                        + (response.error ?? "no tokens returned"))
            }
            stats.rounds += 1
            stats.acceptedDrafts += response.acceptedDraftCount ?? 0
            stats.rejectedDrafts += response.rejectedDraftCount ?? 0

            for token in tokens {
                if context.stopTokens.contains(token) {
                    exitCause = "eos"
                    terminalToken = token
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

            let text = stage.advance {
                detokenizer.text(for: emitted) {
                    tokenizer.decode(tokens: Array($0), skipSpecialTokens: true)
                }
            }
            stats.detokenizeSeconds += text.decodeSeconds
            stats.stopScanSeconds += text.stopSeconds
            stats.gateSeconds += text.gateSeconds
            full = text.full
            if text.hitStop {
                exitCause = "stop-string"
                done = true
            }
            if let onDelta, !text.delta.isEmpty {
                let emitStarted = Date()
                onDelta(text.delta)
                stats.streamEmitSeconds += Date().timeIntervalSince(emitStarted)
            }
            if text.sawToolCall { finishReason = "tool_calls" }
            if tokens.isEmpty {
                exitCause = "empty-round"
                break
            }
        }

        let finalText = stage.finish {
            detokenizer.text(for: emitted) {
                tokenizer.decode(tokens: Array($0), skipSpecialTokens: true)
            }
        }
        stats.detokenizeSeconds += finalText.decodeSeconds
        stats.gateSeconds += finalText.gateSeconds
        full = finalText.full
        if finalText.sawToolCall { finishReason = "tool_calls" }

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
            emittedTokens: emitted,
            terminalToken: terminalToken)
    }

    /// Every id that terminates a turn: the union of `eos_token_id` and
    /// `pad_token_id` across `config.json` and `generation_config.json` (each
    /// accepting a scalar or a list), plus the tokenizer's own EOS id.
    ///
    /// DUPLICATED ON PURPOSE. The worker resolves the same set with
    /// `resolveQwenMTPStopTokens`, but that whole file sits inside
    /// `#if !MLXFAST_TRUSTED_HARNESS` and so compiles to nothing in this
    /// binary -- the trusted harness excludes participant-facing worker code
    /// by construction. Sharing it would mean moving it across that boundary,
    /// which is a bigger change than restating twenty lines. The two must
    /// agree: the worker stops ACCEPTING drafts on this set, and if the parent
    /// recognised a smaller one it would keep decoding past a token the
    /// session already treated as terminal.
    static func serveStopTokens(
        directory: URL, tokenizer: any Tokenizer
    ) -> Set<Int> {
        var ids = Set<Int>()
        for name in ["config.json", "generation_config.json"] {
            guard let data = try? Data(
                contentsOf: directory.appendingPathComponent(name)),
                let root = (try? JSONSerialization.jsonObject(with: data))
                    as? [String: Any]
            else { continue }
            for key in ["eos_token_id", "pad_token_id"] {
                switch root[key] {
                case let value as Int:
                    ids.insert(value)
                case let values as [Any]:
                    ids.formUnion(values.compactMap { $0 as? Int })
                default:
                    continue
                }
            }
        }
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
        }
        return ids
    }

    static func firstStopHit(
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
