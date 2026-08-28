import Foundation
import MLXFastCore
import Tokenizers

// Interactive chat REPL over the native-MTP decode session.
//
// LOCAL DEVELOPER TOOLING. Nothing here is on a scored path, no gate reads it,
// and it lives outside `editablePaths` so `yukon submit` cannot package it. It
// exists to make the speculative decoder observable by hand: you type, the
// model answers, and the accept rate and effective draft depth for that turn
// are reported next to the tokens they produced.
//
// WHY IT REUSES THE BENCHMARK'S WORKER PROTOCOL RATHER THAN A SIDE PATH. The
// numbers are only worth reading if they come from the same machinery the
// ranked run drives. This spawns the same sandboxed worker, sends the same
// `mtp_decode_warm` / `mtp_decode_begin` / `mtp_decode_round` requests, and
// reads accept counts out of the same round responses the trusted driver
// audits. The one addition is `mtp_decode_reset`, which no scored verb issues.
//
// GREEDY ONLY. The session commits the target's argmax; there is no sampler.
// Replies will be more repetitive than a sampled chat, which is a property of
// the decoder being measured, not a defect in this REPL.
extension QwenRuntime {
    public struct QwenChatOptions {
        public let targetWeightsPath: String
        public let mtpHeadPath: String
        public let depth: Int
        public let maxNewTokens: Int
        public let systemPrompt: String?

        public init(
            targetWeightsPath: String,
            mtpHeadPath: String,
            depth: Int,
            maxNewTokens: Int,
            systemPrompt: String?
        ) {
            self.targetWeightsPath = targetWeightsPath
            self.mtpHeadPath = mtpHeadPath
            self.depth = depth
            self.maxNewTokens = maxNewTokens
            self.systemPrompt = systemPrompt
        }
    }

    /// One turn's decode statistics, derived from the round responses.
    struct QwenChatTurnStats {
        var emittedTokens = 0
        var rounds = 0
        var acceptedDrafts = 0
        var rejectedDrafts = 0
        var seconds = 0.0
        var seedPrefillSeconds = 0.0

        // PER-ROUND PARENT SEGMENTS. `seconds` covers the whole turn and
        // `workerRoundSeconds` covers the part of it the worker owned, so the
        // difference between them is the parent's own between-round work --
        // the window in which the worker is blocked on read and the GPU is
        // idle. The four segments below name where that window goes.
        var workerRoundSeconds = 0.0
        var detokenizeSeconds = 0.0
        var stopScanSeconds = 0.0
        var gateSeconds = 0.0
        var streamEmitSeconds = 0.0

        /// Parent work between rounds, by subtraction rather than by summing
        /// the four segments: anything unaccounted for belongs here rather
        /// than disappearing.
        var hostTailSeconds: Double {
            Swift.max(0, seconds - seedPrefillSeconds - workerRoundSeconds)
        }

        /// The host tail as a share of the decode window. This is the number
        /// this plan moves.
        var hostTailShare: Double? {
            // Without a worker-round accumulation the "tail" would be the
            // whole decode window; omit the field rather than print that.
            guard workerRoundSeconds > 0 else { return nil }
            let decodeSeconds = seconds - seedPrefillSeconds
            return decodeSeconds > 0 ? hostTailSeconds / decodeSeconds : nil
        }

        var proposedDrafts: Int { acceptedDrafts + rejectedDrafts }

        /// Share of proposed drafts the target kept. This is THE number the
        /// speculative decoder lives or dies by: verify rows cost nearly a
        /// serial step each, so rejected drafts are work thrown away.
        var acceptRate: Double? {
            proposedDrafts > 0
                ? Double(acceptedDrafts) / Double(proposedDrafts) : nil
        }

        /// Drafts actually proposed per round. This is the EFFECTIVE depth, not
        /// the depth the parent offered: the session may draft fewer (or, near
        /// a boundary, more) than the offer.
        var meanDraftDepth: Double? {
            rounds > 0 ? Double(proposedDrafts) / Double(rounds) : nil
        }

        /// End-to-end rate, prefill included -- what the turn actually felt
        /// like.
        var tokensPerSecond: Double? {
            seconds > 0 ? Double(emittedTokens) / seconds : nil
        }

        /// Rate with the seed prefill excluded. Reported alongside the
        /// end-to-end rate because every turn re-prefills the whole
        /// conversation, so the end-to-end number decays as the chat grows and
        /// says less and less about the decoder itself.
        var decodeTokensPerSecond: Double? {
            let decodeSeconds = seconds - seedPrefillSeconds
            return decodeSeconds > 0
                ? Double(emittedTokens) / decodeSeconds : nil
        }

        /// Tokens committed per round. At depth 0 this is exactly 1; above it,
        /// the amount by which speculation is actually beating serial decode.
        var tokensPerRound: Double? {
            rounds > 0 ? Double(emittedTokens) / Double(rounds) : nil
        }
    }

    public static func qwenChatREPL(
        options: QwenChatOptions,
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
        guard options.maxNewTokens > 0,
              options.maxNewTokens
                  <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
        else {
            throw MLXFastError.invalidInput(
                "--max-tokens must be between 1 and "
                    + "\(MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens) "
                    + "(the worker's own per-session decode ceiling)")
        }

        let tokenizer = try loadLocalTokenizer(at: options.targetWeightsPath)
        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            mtpHeadPath: options.mtpHeadPath
        )
        defer { client.close() }

        FileHandle.standardError.write(Data(
            "mlxfast-swift chat: loading the target and MTP head...\n".utf8))
        let warm = try client.warmMTPDecode()
        guard warm.ok else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the untimed warm: "
                    + (warm.error ?? "no reason reported"))
        }

        printChatBanner(options: options)

        var transcript: [QwenChatMessage] = []
        var turnIndex = 0
        while let line = readLine(strippingNewline: true) {
            let prompt = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if prompt.isEmpty {
                printChatPrompt()
                continue
            }
            if prompt == "/exit" || prompt == "/quit" {
                break
            }
            if prompt == "/reset" {
                transcript.removeAll()
                print("(conversation cleared)\n")
                printChatPrompt()
                continue
            }

            // Every turn re-prefills the whole conversation. There is no cross-
            // turn KV reuse here on purpose: the session's KV state belongs to
            // one `begin`, and stitching turns onto it would mean maintaining
            // cache state the worker's own accounting does not model.
            if turnIndex > 0 {
                let reset = try client.resetMTPDecode()
                guard reset.ok else {
                    throw MLXFastError.invalidInput(
                        "the MTP worker refused the session reset: "
                            + (reset.error ?? "no reason reported"))
                }
            }
            turnIndex += 1

            transcript.append(QwenChatMessage(role: .user, text: prompt))
            let seedText = renderChatML(
                transcript, systemPrompt: options.systemPrompt)
            let seedTokens = tokenizer.encode(
                text: seedText, addSpecialTokens: false)

            let (reply, stats) = try runChatTurn(
                client: client,
                tokenizer: tokenizer,
                seedTokens: seedTokens,
                options: options
            )
            transcript.append(
                QwenChatMessage(role: .assistant, text: reply))

            print("")
            print(renderStatsBar(stats, depth: options.depth))
            print("")
            // Replies go to stdout, the prompt to stderr. stdout is block-
            // buffered under a pipe, so without this the whole transcript
            // arrives after the interleaved stderr chrome.
            fflush(stdout)
            printChatPrompt()
        }
        FileHandle.standardError.write(Data("\nbye\n".utf8))
    }

    /// Drive one prompt to a stop token, streaming text as rounds commit.
    private static func runChatTurn(
        client: RuntimeWorkerClient,
        tokenizer: any Tokenizer,
        seedTokens: [Int],
        options: QwenChatOptions
    ) throws -> (String, QwenChatTurnStats) {
        var stats = QwenChatTurnStats()
        let started = Date()
        let begin = try client.beginMTPDecode(seedTokens: seedTokens)
        stats.seedPrefillSeconds = Date().timeIntervalSince(started)
        guard begin.ok, begin.seedToken != nil else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the seed prefill: "
                    + (begin.error ?? "no seed token returned"))
        }

        // `emitted` starts EMPTY even though begin returned a seed argmax. That
        // token is the turn's first token, but the session holds it as the
        // pending primary and commits it at the top of round 1 -- so it arrives
        // in round 1's `tokens` and appending it here would emit it twice.
        var emitted: [Int] = []
        var rendered = ""
        var stopped = false

        while !stopped, emitted.count < options.maxNewTokens {
            let response = try client.mtpDecodeRound(depth: options.depth)
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
                    stopped = true
                    break
                }
                emitted.append(token)
                if emitted.count >= options.maxNewTokens { break }
            }
            renderDelta(&rendered, emitted: emitted, tokenizer: tokenizer)
            if tokens.isEmpty { break }
        }

        stats.seconds = Date().timeIntervalSince(started)
        stats.emittedTokens = emitted.count
        return (rendered, stats)
    }

    /// Re-decode the whole emitted prefix and print only the new suffix.
    ///
    /// WHY NOT DECODE TOKEN BY TOKEN. Qwen uses byte-level BPE, so a single
    /// token can carry a fragment of a multi-byte UTF-8 character (and of an
    /// emoji or CJK glyph routinely does). Decoding each token alone therefore
    /// produces replacement characters at the seams. Decoding the full prefix
    /// each time is O(n) per round but always yields well-formed text, and n
    /// here is a chat reply, not a benchmark window.
    private static func renderDelta(
        _ rendered: inout String,
        emitted: [Int],
        tokenizer: any Tokenizer
    ) {
        let full = tokenizer.decode(tokens: emitted, skipSpecialTokens: true)
        guard full.hasPrefix(rendered) else {
            // A retokenization boundary rewrote earlier text. Rare, but print
            // the whole thing again rather than emit a corrupted delta.
            print(full, terminator: "")
            rendered = full
            fflush(stdout)
            return
        }
        let delta = String(full.dropFirst(rendered.count))
        if !delta.isEmpty {
            print(delta, terminator: "")
            fflush(stdout)
        }
        rendered = full
    }

    struct QwenChatMessage {
        enum Role { case user, assistant }
        let role: Role
        let text: String
    }

    /// Multi-turn ChatML with thinking pre-closed.
    ///
    /// `QwenChatTemplate.userTurnDisablingThinking` covers the single-user-turn
    /// case the GPQA capture needs. A conversation needs the assistant turns
    /// closed with `<|im_end|>` too, so the framing is built here rather than
    /// widening the gate's helper for a REPL that no gate reads.
    static func renderChatML(
        _ messages: [QwenChatMessage],
        systemPrompt: String?
    ) -> String {
        var rendered = ""
        if let systemPrompt, !systemPrompt.isEmpty {
            rendered += "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
        }
        for message in messages {
            switch message.role {
            case .user:
                rendered += "<|im_start|>user\n\(message.text)<|im_end|>\n"
            case .assistant:
                rendered += "<|im_start|>assistant\n\(message.text)<|im_end|>\n"
            }
        }
        // Thinking pre-closed, matching the sidecar's enable_thinking=false
        // branch. With thinking open the model reasons past any sensible chat
        // budget -- the same failure the GPQA gate measured.
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered
    }

    static func renderStatsBar(
        _ stats: QwenChatTurnStats, depth: Int
    ) -> String {
        var fields: [String] = []
        if let rate = stats.decodeTokensPerSecond {
            fields.append(String(format: "%.1f tok/s decode", rate))
        }
        if let rate = stats.tokensPerSecond {
            fields.append(String(format: "%.1f end-to-end", rate))
        }
        if let accept = stats.acceptRate {
            fields.append(String(
                format: "accept %.0f%% (%d/%d)",
                accept * 100, stats.acceptedDrafts, stats.proposedDrafts))
        } else {
            fields.append("accept n/a (no drafts)")
        }
        if let meanDepth = stats.meanDraftDepth {
            fields.append(String(
                format: "depth %.2f/%d", meanDepth, depth))
        }
        if let perRound = stats.tokensPerRound {
            fields.append(String(format: "%.2f tok/round", perRound))
        }
        fields.append("\(stats.emittedTokens) tok in \(stats.rounds) rounds")
        if let share = stats.hostTailShare {
            fields.append(String(
                format: "host tail %.2fs (%.0f%%) [detok %.2f stop %.2f "
                    + "gate %.2f stream %.2f]",
                stats.hostTailSeconds, share * 100,
                stats.detokenizeSeconds, stats.stopScanSeconds,
                stats.gateSeconds, stats.streamEmitSeconds))
        }
        fields.append(String(
            format: "prefill %.2fs", stats.seedPrefillSeconds))
        let bar = fields.joined(separator: " │ ")
        let rule = String(repeating: "─", count: min(bar.count, 78))
        return rule + "\n" + bar
    }

    private static func printChatBanner(options: QwenChatOptions) {
        let depthNote = options.depth == MLXFastConstants
            .qwenMTPSerialControlDepth
            ? " (serial control: MTP off)" : ""
        FileHandle.standardError.write(Data("""
            mlxfast-swift chat -- greedy, native-MTP speculative decode
              draft depth offered: \(options.depth)\(depthNote)
              max tokens per turn: \(options.maxNewTokens)
              /reset clears the conversation, /exit quits

            """.utf8))
        printChatPrompt()
    }

    private static func printChatPrompt() {
        FileHandle.standardError.write(Data("you> ".utf8))
    }
}
