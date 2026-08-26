import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

/// The worker's per-session output ceiling.
///
/// Defaults to the pinned constant, which is what every ranked and benchmark
/// path uses because none of them set the variable. `serve` raises it so an
/// agent turn is not truncated at 1,536 tokens. Input length is NOT capped here
/// or anywhere else: the 16 full-attention layers use an unbounded
/// `KVCacheSimple` and the other 48 carry constant-size recurrent state.
/// The worker's per-session OUTPUT ceiling, supplied by the trusted parent on
/// argv.
///
/// WHY ARGV AND NOT AN ENVIRONMENT VARIABLE. `sanitizedRuntimeWorkerEnvironment`
/// is a strict allowlist that starts from an empty environment, and its
/// maintainer contract forbids adding an `MLXFAST_` allowance: worker
/// configuration travels on argv, the way `--weights` and `--mtp-head` already
/// do. An env var would have been silently dropped at spawn and the override
/// would have looked applied while doing nothing.
///
/// Defaults to the pinned constant, which is what every benchmark and ranked
/// path gets because none of them pass the flag. Input length is NOT capped
/// here or anywhere else: the 16 full-attention layers use an unbounded
/// `KVCacheSimple` and the other 48 carry constant-size recurrent state.
nonisolated(unsafe) private var qwenMTPDecodeCeiling =
    MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens

/// Validated `mtp_decode_round` request.
struct QwenMTPRoundRequest: Equatable {
    let depth: Int
}

/// Cheap identity for the running BUILD: the loaded weights tree plus the
/// worker binary itself.
///
/// WHY THE BINARY IS PART OF THIS. A persisted prefill checkpoint stores
/// per-layer cache rows -- the output of the attention path -- so kernel
/// numerics propagate into every stored row. A checkpoint written by one
/// binary and read back by another (a kernel, model, or transform edit)
/// silently splices old numerics into a new run; this repo's own guidance
/// puts near-tie argmax flips at roughly the magnitude a fused-vs-unfused
/// attention kernel can differ by. A hand-bumped "kernel generation" constant
/// would rot the first time an edit forgets to bump it, so instead this folds
/// in the running worker binary's own size and modification time: any edit
/// that changes the binary's bytes changes both, so every kernel, model, or
/// transform edit invalidates the cache by construction -- no bump required.
/// It over-invalidates on a no-op rebuild (same source, new mtime); that
/// costs one re-prefill, against a failure mode that is silent and
/// unbounded.
///
/// Hashing 15 GB of weights on every start is not affordable, and is not what
/// the guard needs: the question is whether the tree (or the binary) CHANGED,
/// and name, size and modification time answer that. FNV-1a rather than
/// `Hasher` because this value must be stable across processes -- `Hasher` is
/// randomly reseeded per process, so its keys would not survive a restart.
func qwenBuildIdentity(weightsPath: String) -> String {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    func mix(_ bytes: some Sequence<UInt8>) {
        for byte in bytes {
            hash = (hash ^ UInt64(byte)) &* 0x100_0000_01b3
        }
    }
    let manager = FileManager.default
    let names = ((try? manager.contentsOfDirectory(atPath: weightsPath)) ?? [])
        .sorted()
    for name in names {
        mix(Array(name.utf8))
        let attributes = try? manager.attributesOfItem(
            atPath: weightsPath + "/" + name)
        let size = (attributes?[.size] as? Int) ?? 0
        let modified = (attributes?[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        mix(Array(String(size).utf8))
        mix(Array(String(Int(modified)).utf8))
    }
    // The running worker binary's own size and mtime. A component that
    // silently vanishes when the path cannot be resolved is the exact
    // failure this function exists to prevent, so fold in an explicit
    // marker rather than skipping it.
    let binaryPath = CommandLine.arguments.first
        ?? ProcessInfo.processInfo.arguments.first
    if let binaryPath,
       let attributes = try? manager.attributesOfItem(atPath: binaryPath)
    {
        let size = (attributes[.size] as? Int) ?? 0
        let modified = (attributes[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        mix(Array("bin:\(binaryPath)".utf8))
        mix(Array(String(size).utf8))
        mix(Array(String(Int(modified)).utf8))
    } else {
        mix(Array("bin:unresolved".utf8))
    }
    return String(hash, radix: 36)
}

/// Stable string for the KV basis a checkpoint was written under.
///
/// Rows written at one bit width, group size, or rotation basis decode into
/// plausible numbers under another rather than failing, so the basis has to be
/// part of the fingerprint. `nil` is the bf16 path and gets its own name so it
/// cannot collide with a quantized policy that happens to stringify short.
func qwenKVPolicyIdentity() -> String {
    guard let policy = Qwen36MTPBlockSession.KVQuantization.fromEnvironment()
    else { return "bf16" }
    return "b\(policy.bits)g\(policy.groupSize)"
        + "m\(policy.minimumOffset)r\(policy.rotate ? 1 : 0)"
        + "s\(policy.seed)"
}

/// Flatten a snapshot's heterogeneous per-layer caches into named arrays.
///
/// The layer stack mixes `ArraysCache` for the 48 gated-delta layers with
/// attention KV caches for the other 16, and each carries a different number
/// of state arrays. `layerTags` and `stateCounts` are what let the reader put
/// them back in the right classes in the right order.
func qwenMTPCacheEntry(
    snapshot: Qwen36MTPBlockSession.SessionSnapshot, tokens: [Int]
) throws -> QwenPrefillDiskCache.CacheEntry {
    var arrays: [String: MLXArray] = [:]
    var tags: [String] = []
    var counts: [Int] = []
    var offsets: [Int] = []
    for (layer, cache) in snapshot.cache.enumerated() {
        tags.append(try QwenPrefillDiskCache.tag(for: cache))
        offsets.append(cache.offset)
        let state = cache.state
        counts.append(state.count)
        for (index, array) in state.enumerated() {
            arrays["L\(layer).S\(index)"] = array
        }
    }
    return QwenPrefillDiskCache.CacheEntry(
        tokens: tokens, layerTags: tags, stateCounts: counts, offsets: offsets,
        arrays: arrays,
        kvBytes: snapshot.kvBytes, recurrentBytes: snapshot.recurrentBytes,
        seedTokenCount: snapshot.seedTokenCount,
        committedTokenCount: snapshot.committedTokenCount)
}

/// Resume points, keyed by conversation, shared across session rebuilds.
///
/// Deliberately OUTSIDE `QwenMTPWorkerState`: a reset replaces that struct
/// wholesale, and the whole point of the store is to survive exactly that.
/// Budget defaults to 64 GiB clamped to a quarter of physical RAM; a ranked run
/// never populates it because no ranked request carries a `conversationId`.
let qwenMTPResumeStore =
    QwenSessionCacheStore<Qwen36MTPBlockSession.SessionSnapshot>()

/// Cache root and fingerprint for disk-persisted prefill checkpoints.
///
/// `DARKBLOOM_PREFILL_CACHE_DIR` is opt-in and off by default until the
/// on-disk path is proven; attached once, before the request loop starts,
/// because the fingerprint (weights identity, chunk size, KV basis) is fixed
/// for the process lifetime and does not vary per request.
func attachQwenMTPDiskCache(targetWeightsPath: String) {
    guard let root = ProcessInfo.processInfo
        .environment["DARKBLOOM_PREFILL_CACHE_DIR"], !root.isEmpty
    else { return }
    qwenMTPResumeStore.attachDisk(
        root: URL(fileURLWithPath: root),
        fingerprint: QwenPrefillDiskCache.Fingerprint(
            weightsIdentity: qwenBuildIdentity(weightsPath: targetWeightsPath),
            chunkSize: QwenPrefillChunking.chunkSize,
            kvPolicy: qwenKVPolicyIdentity()))
}

struct QwenMTPWorkerState {
    var began = false
    /// Set by `mtp_decode_warm`. The warm is input-independent, so the trusted
    /// parent runs it BEFORE starting its clock; `began` then skips it.
    var warmed = false
    var poisoned = false
    var seedTokenCount = 0
    var decodedTokenCount = 0
    /// Reference-side only, created lazily: the candidate worker never receives a
    /// reference request, so it never allocates the second cache stack.
    var referenceSession: Qwen36MTPReferenceSession?
}

/// Strict validation for an MTP round request.
///
/// Like the DFlash equivalent this deliberately does NOT bound the depth by any
/// remaining-token count: the worker is never told how much of the decode window
/// is left, so it cannot special-case the tail. The trusted parent always asks
/// for a full round and truncates the scored prefix itself.
///
/// The depth rides on the existing `max_block_size` wire field rather than a new
/// one. That field is documented as "the parent-chosen block width for this
/// round" and is already the ONLY block-shaped field on the request; adding a
/// second spelling for the same idea would give a submission two places to look
/// for the window shape and give the protocol two fields that must agree.
func validateQwenMTPRoundRequest(
    _ request: RuntimeWorkerRequest,
    decodedTokenCount: Int
) throws -> QwenMTPRoundRequest {
    guard request.id > 0, request.kind == "mtp_decode_round" else {
        throw MLXFastError.invalidInput(
            "MTP round request has an invalid id or kind")
    }
    guard request.promptTokens == nil,
          request.seedTokens == nil,
          request.token == nil,
          request.steps == nil,
          request.topK == nil,
          request.expectedToken == nil,
          request.prefixTokens == nil,
          request.startOffset == nil,
          request.rowCount == nil,
          request.declaredBlockWidth == nil,
          request.seedTokenCount == nil,
          request.verifyBlockTokens == nil,
          request.temperature == nil,
          request.topP == nil,
          request.samplingSeed == nil,
          let depth = request.maxBlockSize,
          // 0 is legal and is the TRUE SERIAL CONTROL the paired score divides
          // by -- MTP off, one token per target forward -- served by the same
          // worker, the same protocol and the same forward. 1 is also legal and
          // is a labelled speculative-depth-1 diagnostic, never a denominator.
          depth >= Qwen36MTPLimits.serialControlDepth,
          depth <= Qwen36MTPLimits.maxDepth
    else {
        throw MLXFastError.invalidInput(
            "MTP round request has invalid or cross-kind fields")
    }
    guard decodedTokenCount >= 0 else {
        throw MLXFastError.invalidInput(
            "MTP worker has a negative committed token count")
    }
    // Depth 0 commits exactly one token per round; every other depth commits at
    // most `depth + 1`.
    let (requestedTotal, overflow) =
        decodedTokenCount.addingReportingOverflow(Swift.max(depth + 1, 1))
    guard !overflow,
          requestedTotal <= qwenMTPDecodeCeiling
    else {
        throw MLXFastError.invalidInput(
            "MTP round request exceeds the configured decode ceiling")
    }
    return QwenMTPRoundRequest(depth: depth)
}

/// Resolve the stop set the way MTPLX's `_default_stop_tokens` does: the union of
/// `eos_token_id` / `pad_token_id` over `config.json` and `generation_config.json`
/// (each accepting a scalar or a list) plus the tokenizer's own EOS id.
///
/// Applied identically to the speculative loop and to the serial reference, which
/// is what keeps an EOS branch comparable between them.
func resolveQwenMTPStopTokens(
    directory: URL,
    tokenizer: (any MLXLMCommon.Tokenizer)?
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
    if let eos = tokenizer?.eosTokenId {
        ids.insert(eos)
    }
    return ids
}

extension QwenRuntime {
    /// Runtime worker for the Qwen 3.6 native-MTP track.
    ///
    /// Loads the organizer-pinned backbone with the separately pinned MTP head
    /// merged at load, warms the round shapes, and then serves the `mtp_*`
    /// request kinds. Everything expensive happens before the protocol hello,
    /// i.e. outside every scored window.
    public static func runQwenMTPWorker(
        targetWeightsPath: String,
        mtpHeadPath: String,
        decodeCeiling: Int? = nil
    ) throws {
        // Installed before anything reads it, and only when the parent asked
        // for a different bound. No caller on the ranked path passes one.
        if let decodeCeiling, decodeCeiling > 0 {
            qwenMTPDecodeCeiling = decodeCeiling
        }
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        applyQwenMTPStartupMemoryProfile()

        let targetURL = URL(fileURLWithPath: targetWeightsPath)
        // An EMPTY head path is the headless local-research configuration --
        // a sibling `qwen3_5_text` tower with no published MTP head, decoded
        // serially. The parent only ever produces it behind the geometry
        // escape; the ranked path always passes a real directory.
        let headURL = mtpHeadPath.isEmpty
            ? nil
            : URL(fileURLWithPath: mtpHeadPath)
        // The layout is read from the backbone's OWN config and decides both
        // which class the factory builds and whether the transform's
        // `language_model.` text-tower prefix has to be stripped from the
        // primary tree. Both Qwen 3.6 classes are accepted, because
        // `Qwen35Model` is a pure pass-through wrapper around the same
        // `Qwen35TextModel` for every call this session makes -- see
        // `Qwen36MTPTarget`. Accepting only one of them is what made this worker
        // unloadable in both directions: the transformed tree builds the bare
        // text model, and the raw pinned reference builds the wrapper.
        var backboneLayout = Qwen36MTPHeadAttachment.BackboneLayout.textModel
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL,
            headDirectory: headURL
        ) { layout in
            backboneLayout = layout
            return try waitForQwenMTPAsync {
                try await LLMModelFactory.shared.load(
                    from: targetURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        }
        guard let model = context.model as? any Qwen36MTPTarget else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone loaded as \(type(of: context.model)), "
                    + "which is not an MTP-capable Qwen 3.6 model. This track "
                    + "serves Qwen35TextModel (the transformed weights/ tree, "
                    + "model_type qwen3_5_text) and Qwen35Model (the raw pinned "
                    + "reference, model_type qwen3_5) and nothing else.")
        }
        fputs(
            "mlxfast-worker: qwen-mtp backbone layout=\(backboneLayout.rawValue) "
                + "class=\(type(of: context.model)) "
                + "key_prefix_strip="
                + "\(backboneLayout.primaryKeyPrefixStrip ?? "<none>")\n",
            stderr
        )
        // Fail here rather than at the first draft. `_qwen35MTPEnabled` is set
        // and cleared around the load, and the head only attaches when the
        // configuration also declares `mtp_num_hidden_layers > 0`; a tree whose
        // config lost that field would otherwise load, never draft, and report a
        // perfectly exact run at zero acceptance.
        guard model.hasMTPHead || headURL == nil else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head did not attach to the loaded backbone: the "
                    + "runtime config.json must declare mtp_num_hidden_layers > 0 "
                    + "and the head tree must carry its "
                    + "\(Qwen36MTPHeadAttachment.expectedHeadTensorCount) tensors")
        }
        eval(context.model)

        let stopTokens = resolveQwenMTPStopTokens(
            directory: targetURL, tokenizer: context.tokenizer)

        // Warm every legal round shape on throwaway cache state, before the
        // hello. The real begin request performs the trusted allocator clear and
        // re-warms the working set it frees.
        let warmup = try Qwen36MTPBlockSession(
            model: model, stopTokens: stopTokens)
        try warmup.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
        var session = try Qwen36MTPBlockSession(
            model: model, stopTokens: stopTokens)
        attachQwenMTPDiskCache(targetWeightsPath: targetWeightsPath)

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))

        var state = QwenMTPWorkerState()
        var expectedRequestID = 1
        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else { continue }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(
                    RuntimeWorkerRequest.self,
                    from: Data(line.utf8)
                )
                guard request.id == expectedRequestID else {
                    throw MLXFastError.invalidInput(
                        "MTP request id must be monotonic; expected "
                            + "\(expectedRequestID), got \(request.id)")
                }
                expectedRequestID += 1
                // LOCAL INTERACTIVE TOOLING ONLY -- no scored verb issues this.
                //
                // `mtp_decode_begin` is one-shot by design: a measured window is
                // exactly one seed and one decode pass, and a parent able to
                // re-begin mid-window could reset KV state the row ledger is
                // closed against. A chat turn needs precisely that, so the reset
                // is handled HERE, in the read loop that owns `session`, rather
                // than inside `handleQwenMTPWorkerRequest`, which receives the
                // session by value and so can mutate it but never replace it.
                //
                // Rebuilding the session drops all KV state; the next begin
                // re-prefills. The ~14 GiB `model` is untouched, so a turn costs
                // a prefill, not a reload.
                if request.kind == "mtp_decode_reset" {
                    // A separate binding rather than the loop's `response`:
                    // this branch answers and `continue`s, and Swift cannot
                    // prove the single-assignment of a `let` across that jump.
                    let resetResponse: RuntimeWorkerResponse
                    do {
                        session = try Qwen36MTPBlockSession(
                            model: model, stopTokens: stopTokens)
                        // `warmed` is carried, not cleared. Leaving it false
                        // makes the next begin re-run the allocator clear and
                        // `warmAllDepths`, measured at 14.8s against a 0.37s
                        // warmed prefill -- i.e. the warm, not the model, would
                        // dominate every turn after the first. The warm is
                        // input-independent (it compiles round shapes; it never
                        // sees the seed), so skipping it changes what the turn
                        // COSTS, never what it decodes.
                        state = QwenMTPWorkerState()
                        state.warmed = true
                        resetResponse = RuntimeWorkerResponse(
                            id: request.id, nonce: sessionNonce, ok: true)
                    } catch {
                        state.poisoned = true
                        resetResponse = RuntimeWorkerResponse(
                            id: request.id,
                            nonce: sessionNonce,
                            ok: false,
                            error: "\(error)"
                        )
                    }
                    try protocolIO.writeLine(try encoder.encode(resetResponse))
                    continue
                }
                do {
                    response = try handleQwenMTPWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        model: model,
                        session: session,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(
                    id: -1,
                    nonce: sessionNonce,
                    ok: false,
                    error: "\(error)"
                )
            }
            try protocolIO.writeLine(try encoder.encode(response))
        }
    }

    static func handleQwenMTPWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        model: any Qwen36MTPTarget,
        session: Qwen36MTPBlockSession,
        state: inout QwenMTPWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard !state.poisoned else {
            throw MLXFastError.invalidInput(
                "the MTP decode session is poisoned after an earlier failure")
        }

        switch request.kind {
        case "mtp_decode_warm":
            guard !state.began,
                  request.id > 0,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.temperature == nil,
                  request.topP == nil,
                  request.samplingSeed == nil
            else {
                throw MLXFastError.invalidInput(
                    "MTP warm request is malformed or arrived after begin")
            }
            do {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
                state.warmed = true
                return RuntimeWorkerResponse(
                    id: request.id, nonce: sessionNonce, ok: true)
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_decode_snapshot":
            // Record a resume point for the session as it stands, under the
            // caller's token history.
            //
            // WHY THE PARENT DRIVES THIS. The worker knows its cache state but
            // not the token history that names it: emitted tokens live on the
            // parent side. Recording at `begin` alone would pin every resume
            // point to a PROMPT boundary, so a conversation switch would re-run
            // the whole previous reply. Recording at turn end keeps the resume
            // point current, and the tail a switch must replay stays short.
            guard state.began,
                  let conversationId = request.conversationId,
                  let tokens = request.seedTokens, !tokens.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "MTP snapshot request is malformed or arrived before begin")
            }
            // Capturing state cannot fail: it copies caches the session
            // already holds, so there is no error path to poison the session on.
            let snapshot = session.snapshotState()
            qwenMTPResumeStore.record(
                conversation: conversationId,
                tokens: tokens,
                state: snapshot,
                roundBytes: snapshot.recurrentBytes,
                kvBytes: snapshot.kvBytes)
            return RuntimeWorkerResponse(
                id: request.id, nonce: sessionNonce, ok: true,
                resumedTokens: tokens.count)

        case "mtp_decode_begin":
            guard !state.began,
                  request.id > 0,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "MTP begin request is repeated or malformed")
            }
            if !state.warmed {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
                state.warmed = true
            }
            // Temperature is opt-in and absent means greedy, which is the
            // ranked path: `setSampling(nil)` restores it explicitly so a
            // reused worker cannot inherit a previous request's policy.
            if let temperature = request.temperature, temperature > 0 {
                session.setSampling(Qwen36MTPSampling(
                    temperature: Float(temperature),
                    topP: Float(request.topP ?? 1.0),
                    seed: request.samplingSeed ?? UInt64.random(in: 0 ... .max)))
            } else {
                session.setSampling(nil)
            }
            do {
                // RESUME BEFORE PREFILL. `restoreState` reinstates the 48
                // gated-delta layers' recurrent state, which is the thing
                // `extend`'s own documentation says `trim()` cannot roll back --
                // so restore-then-extend performs the rewind that a bare
                // `extend` cannot. A miss falls through to an ordinary begin.
                var seedToken: Int
                var resumedTokens: Int?
                // Content-addressed checkpoint keys for this prompt. Derived
                // from the tokens alone, so a prompt shared with ANOTHER
                // connection resolves to the same entries -- which is the case
                // that matters when several agent sessions run at once and
                // agree on their first several thousand tokens.
                // Turn boundaries when the parent supplied them, fixed stride
                // otherwise. The parent knows where its messages end; the
                // worker only sees a flat token array and would have to guess.
                var chunkKeys = (request.turnBoundaries?.isEmpty == false)
                    ? QwenPrefillChunking.chainKeys(
                        for: seedTokens, boundaries: request.turnBoundaries!)
                    : QwenPrefillChunking.chainKeys(for: seedTokens)
                // Divergence-learned boundary: the position where this prompt
                // stops agreeing with a recently completed stream, which is
                // exactly where the NEXT prompt sharing the same harness
                // boilerplate will stop agreeing too. Inserted as a
                // first-class boundary so the existing checkpoint, in-memory
                // match, and disk match paths all see it: `prefixKey` derives
                // the same running-FNV key `chainKeys` would at that position,
                // so a checkpoint recorded here is found by any later request
                // that discovers the same divergence.
                if let boundary = qwenMTPResumeStore.learnedBoundary(
                        incoming: seedTokens),
                    let key = QwenPrefillChunking.prefixKey(
                        for: seedTokens, count: boundary)
                {
                    chunkKeys = QwenPrefillChunking.insertingBoundary(
                        boundary, key: key, into: chunkKeys)
                }

                /// Prefill `seedTokens` from absolute position `base`, taking a
                /// checkpoint at every chunk boundary beyond it.
                ///
                /// Boundaries are absolute positions -- stride multiples,
                /// caller-supplied turn ends, or a divergence-learned
                /// boundary -- never relative to `base`. A checkpoint is only
                /// reusable if the next request lands on the same boundary,
                /// and the next request will not share this one's resume
                /// position.
                func prefillCheckpointed(from base: Int) throws -> Int {
                    var position = base
                    var token: Int?
                    // Every chunk boundary strictly beyond `base`, then the end
                    // of the prompt. The final segment is usually a partial
                    // chunk and gets no checkpoint.
                    var stops = chunkKeys.map(\.tokenCount).filter { $0 > base }
                    if stops.last != seedTokens.count {
                        stops.append(seedTokens.count)
                    }
                    for stop in stops {
                        let segment = Array(seedTokens[position ..< stop])
                        if segment.isEmpty { continue }
                        if position == 0 {
                            token = try session.begin(
                                seedTokens: segment,
                                expectedTotalTokens: seedTokens.count)
                        } else {
                            token = try session.extend(tokens: segment)
                        }
                        position = stop
                        // Checkpoint only at a real boundary, and only when one
                        // is not already retained -- concurrent conversations
                        // sharing a prefix would otherwise each pay 144 MiB to
                        // store the same state.
                        guard let key = chunkKeys.first(
                            where: { $0.tokenCount == stop })?.key,
                            !qwenMTPResumeStore.hasChunk(key: key)
                        else { continue }
                        let checkpoint = session.snapshotState()
                        qwenMTPResumeStore.recordChunk(
                            key: key,
                            tokens: Array(seedTokens.prefix(stop)),
                            state: checkpoint,
                            roundBytes: checkpoint.recurrentBytes,
                            kvBytes: checkpoint.kvBytes)
                        if let entry = try? qwenMTPCacheEntry(
                            snapshot: checkpoint, tokens: Array(seedTokens.prefix(stop)))
                        {
                            try? qwenMTPResumeStore.recordChunkPersisting(
                                key: key, entry: entry)
                        }
                    }
                    guard let token else {
                        throw MLXFastError.invalidInput(
                            "chunked prefill produced no seed token for "
                                + "\(seedTokens.count) tokens from \(base)")
                    }
                    return token
                }

                if let conversationId = request.conversationId,
                   let hit = qwenMTPResumeStore.bestMatch(
                       conversation: conversationId, incoming: seedTokens),
                   (try? session.restoreState(hit.round.state)) != nil
                {
                    seedToken = try session.extend(tokens: hit.tail)
                    resumedTokens = hit.round.tokenCount
                    FileHandle.standardError.write(Data(
                        ("qwen-mtp: resumed \(hit.round.tokenCount) cached "
                            + "tokens, prefilling \(hit.tail.count)\n").utf8))
                } else if let hit = qwenMTPResumeStore.chunkMatch(
                    keys: chunkKeys, incoming: seedTokens),
                    (try? session.restoreState(hit.round.state)) != nil
                {
                    // The conversation-scoped match missed but the prompt
                    // shares a prefix with something already prefilled. This is
                    // the ordinary case for a chat client, which re-renders its
                    // history each turn and so never reproduces the exact token
                    // stream the previous turn ended on.
                    seedToken = try prefillCheckpointed(from: hit.round.tokenCount)
                    resumedTokens = hit.round.tokenCount
                    FileHandle.standardError.write(Data(
                        ("qwen-mtp: resumed \(hit.round.tokenCount) cached "
                            + "tokens from the shared prefix store, prefilling "
                            + "\(hit.tail.count)\n").utf8))
                } else if let hit = qwenMTPResumeStore.diskChunkMatch(
                    keys: chunkKeys, incoming: seedTokens) {
                    // Disk checkpoint. Tried only after the in-memory
                    // `chunkMatch` misses, so a warm process never pays the
                    // read; a cold one pays a few hundred milliseconds of
                    // safetensors load instead of minutes of prefill.
                    //
                    // Same environment the fingerprint was built from, and the
                    // fingerprint already refused any checkpoint written under
                    // a different policy, so this is guaranteed to be the
                    // policy the rows were written at.
                    //
                    // Unlike the two in-memory branches above, a failure here
                    // must not poison the worker: a malformed on-disk
                    // checkpoint (missing array, unknown tag, layer-count
                    // disagreement) is data corruption the process did not
                    // cause and cannot fix by refusing every later request. So
                    // this is scoped to its own `do`, matching the `try?`-and-
                    // fall-through shape of the two branches above it: any
                    // failure deletes the offending file and falls through to
                    // a full prefill instead of escaping to the outer `catch`.
                    do {
                        let policy = Qwen36MTPBlockSession.KVQuantization
                            .fromEnvironment()
                        let quantization = policy.map {
                            (groupSize: $0.groupSize, bits: $0.bits)
                        }
                        let caches = try QwenPrefillDiskCache.restoreCaches(
                            from: hit.entry, quantization: quantization)
                        try session.adoptRestoredCaches(
                            caches,
                            seedTokenCount: hit.entry.seedTokenCount,
                            committedTokenCount: hit.entry.committedTokenCount)
                        // Record the restored state in memory too, so a
                        // second identical request in this process resumes
                        // from RAM instead of paying the disk read again.
                        // Captured HERE, before the further prefill below,
                        // because the record is keyed on `hit.entry.tokens`
                        // and must describe the session at exactly that many
                        // committed tokens.
                        qwenMTPResumeStore.recordChunk(
                            key: hit.key, tokens: hit.entry.tokens,
                            state: session.snapshotState(),
                            roundBytes: hit.entry.recurrentBytes,
                            kvBytes: hit.entry.kvBytes)
                        QwenPrefillCacheDiagnostics.log(
                            "restore: key=\(hit.key) adopted, extending from "
                                + "\(hit.entry.tokens.count) tokens")
                        seedToken = try prefillCheckpointed(
                            from: hit.entry.tokens.count)
                        resumedTokens = hit.entry.tokens.count
                        FileHandle.standardError.write(Data(
                            ("qwen-mtp: resumed \(hit.entry.tokens.count) "
                                + "cached tokens from disk, prefilling "
                                + "\(seedTokens.count - hit.entry.tokens.count)\n")
                                .utf8))
                    } catch {
                        QwenPrefillCacheDiagnostics.log(
                            "restore: key=\(hit.key) FAILED: \(error)")
                        FileHandle.standardError.write(Data(
                            ("qwen-mtp: disk checkpoint \(hit.key) failed to "
                                + "restore (\(error)); deleting it and "
                                + "re-prefilling\n").utf8))
                        qwenMTPResumeStore.dropDiskEntry(key: hit.key)
                        seedToken = try prefillCheckpointed(from: 0)
                    }
                } else {
                    if let conversationId = request.conversationId {
                        // A refused restore means the stored basis no longer
                        // matches the running policy. Drop the whole
                        // conversation so the next turn does not pay the same
                        // failed match again.
                        qwenMTPResumeStore.drop(conversation: conversationId)
                    }
                    seedToken = try prefillCheckpointed(from: 0)
                }
                if let conversationId = request.conversationId {
                    let snapshot = session.snapshotState()
                    qwenMTPResumeStore.record(
                        conversation: conversationId,
                        tokens: seedTokens,
                        state: snapshot,
                        roundBytes: snapshot.recurrentBytes,
                        kvBytes: snapshot.kvBytes)
                }
                // Retain this prompt's token stream (tokens only, ~8
                // bytes/token) so the next request can discover where it
                // diverges. Recorded after a successful prefill: a stream that
                // failed to prefill proves nothing about a reusable boundary.
                qwenMTPResumeStore.recordStream(tokens: seedTokens)
                state.began = true
                state.seedTokenCount = seedTokens.count
                state.decodedTokenCount = 0
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    seedToken: seedToken,
                    resumedTokens: resumedTokens
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_decode_extend":
            // LOCAL INTERACTIVE TOOLING ONLY -- no scored verb issues this.
            //
            // Continue the live session with more input instead of paying a
            // fresh prefill. `state.decodedTokenCount` is deliberately NOT
            // reset: the ceiling caps EMITTED tokens over the whole
            // conversation, so a caller that extends forever still hits it.
            // `state.seedTokenCount` grows by the extension so the round
            // handler's `seed + decoded == targetCacheOffset` check keeps
            // holding.
            guard state.began,
                  request.id > 0,
                  let extendTokens = request.seedTokens,
                  !extendTokens.isEmpty,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.temperature == nil,
                  request.topP == nil,
                  request.samplingSeed == nil
            else {
                throw MLXFastError.invalidInput(
                    "MTP extend request arrived before begin or is malformed")
            }
            do {
                let seedToken = try session.extend(tokens: extendTokens)
                let (nextSeedCount, seedOverflow) =
                    state.seedTokenCount.addingReportingOverflow(
                        extendTokens.count)
                guard !seedOverflow else {
                    throw MLXFastError.invalidInput(
                        "MTP extend overflowed the session seed count")
                }
                state.seedTokenCount = nextSeedCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    seedToken: seedToken
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_decode_round":
            guard state.began else {
                throw MLXFastError.invalidInput(
                    "MTP round requested before begin")
            }
            let round = try validateQwenMTPRoundRequest(
                request, decodedTokenCount: state.decodedTokenCount)
            do {
                let result = try session.generateRound(depth: round.depth)
                let (nextCount, overflow) =
                    state.decodedTokenCount.addingReportingOverflow(
                        result.tokens.count)
                let (expectedOffset, offsetOverflow) =
                    state.seedTokenCount.addingReportingOverflow(nextCount)
                guard !overflow,
                      !offsetOverflow,
                      nextCount <= qwenMTPDecodeCeiling,
                      result.targetCacheOffset == expectedOffset,
                      // The ledger the parent audits has to close inside the
                      // worker too, so a broken round is caught at the boundary
                      // it was produced at rather than a round later.
                      result.acceptedDraftCount + result.rejectedDraftCount
                          + 1 == result.declaredRows,
                      // Depth 0 must commit exactly one token and declare exactly
                      // one row: a serial control that drafted anything would be
                      // an accelerated denominator, which is the specific defect
                      // this depth exists to remove.
                      round.depth != Qwen36MTPLimits.serialControlDepth
                          || (result.tokens.count == 1
                              && result.declaredRows == 1
                              && result.draftTokens.isEmpty),
                      result.perRowTop2Tokens.count == result.declaredRows,
                      result.perRowTop2Logits.count == result.declaredRows
                else {
                    throw MLXFastError.invalidInput(
                        "MTP round ledger or target cache offset diverged")
                }
                state.decodedTokenCount = nextCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    tokens: result.tokens,
                    declaredRows: result.declaredRows,
                    perRowTop2Tokens: result.perRowTop2Tokens,
                    perRowTop2Logits: result.perRowTop2Logits,
                    draftTokens: result.draftTokens,
                    acceptedDraftCount: result.acceptedDraftCount,
                    rejectedDraftCount: result.rejectedDraftCount,
                    targetCacheOffset: result.targetCacheOffset
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_reference_prefill":
            // REFERENCE SIDE ONLY. Establishes the run's seed token in the
            // candidate's own frame -- one bulk forward over the whole seed --
            // and primes the continuous width-1 frame at the end of the seed so
            // the first row request is a continuation rather than a rebuild.
            guard request.token == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  request.prefixTokens == nil,
                  request.startOffset == nil,
                  request.rowCount == nil,
                  request.declaredBlockWidth == nil,
                  request.seedTokenCount == nil,
                  request.verifyBlockTokens == nil,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "MTP reference-prefill request is malformed or has "
                        + "cross-kind fields")
            }
            let reference = Qwen36MTPReferenceSession(model: model)
            let seedToken = try reference.prefillSeed(seedTokens)
            state.referenceSession = reference
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "mtp_reference_rows":
            guard request.token == nil,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  request.declaredBlockWidth == nil,
                  let prefixTokens = request.prefixTokens,
                  !prefixTokens.isEmpty,
                  let seedTokenCount = request.seedTokenCount,
                  seedTokenCount > 0,
                  seedTokenCount <= prefixTokens.count,
                  let startOffset = request.startOffset,
                  startOffset >= seedTokenCount,
                  let rowCount = request.rowCount,
                  rowCount > 0,
                  // The verify block is the parent's reconstruction of the
                  // candidate's own verify input for a round: row 0 is the
                  // parent's committed primary, the rest are the journalled
                  // drafts. Width is bounded exactly like a round.
                  request.verifyBlockTokens.map({
                      !$0.isEmpty && $0.count <= Qwen36MTPLimits.maxDepth + 1
                  }) ?? true
            else {
                throw MLXFastError.invalidInput(
                    "MTP reference-rows request is malformed or has cross-kind "
                        + "fields")
            }
            let reference: Qwen36MTPReferenceSession
            if let existing = state.referenceSession {
                reference = existing
            } else {
                reference = Qwen36MTPReferenceSession(model: model)
                state.referenceSession = reference
            }
            let answer = try reference.rows(
                tokens: prefixTokens,
                seedTokenCount: seedTokenCount,
                startOffset: startOffset,
                count: rowCount,
                verifyBlockTokens: request.verifyBlockTokens
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                referenceK1Argmax: answer.rows.map(\.sequentialArgmax),
                referenceTop2Tokens: answer.rows.map(\.top2Tokens),
                referenceTop2Logits: answer.rows.map(\.top2Logits),
                referenceTop1Logits: answer.rows.map(\.top1Logit),
                referenceVerifyTop2Tokens: answer.verifyBlockTop2Tokens,
                referenceVerifyTop2Logits: answer.verifyBlockTop2Logits
            )

        default:
            throw MLXFastError.invalidInput(
                "MTP worker rejects request kind \(request.kind)")
        }
    }
}

/// Select the startup memory profile before the backbone and head loads, for the
/// same reason the DFlash worker does: this worker never constructs
/// `Qwen35RuntimeWeightCache`, so it is a startup path the policy would otherwise
/// never reach, and it is the path that needs it — it holds the whole text tower
/// plus the head and warms every legal round width before the protocol hello.
private func applyQwenMTPStartupMemoryProfile() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
        requestedProfile: ProcessInfo.processInfo.environment[
            RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
        ]
    )
    guard policy.isLowMemory else { return }
    setenv("MLX_MAX_MB_PER_BUFFER", String(policy.maxMegabytesPerCommandBuffer), 1)
    setenv("MLX_MAX_OPS_PER_BUFFER", String(policy.maxOperationsPerCommandBuffer), 1)
    for (name, value) in policy.environmentOverrides {
        setenv(name, value, 0)
    }
    Memory.cacheLimit = policy.cacheLimitBytes
    fputs(
        "mlxfast-worker: low-memory startup profile engaged ("
            + policy.selectionReason + ")\n",
        stderr
    )
}

private final class QwenMTPAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

/// Bridge the vendored async loader into the synchronous worker startup path.
/// Same reasoning as the DFlash bridge: the values crossing this boundary are
/// non-`Sendable` vendored types, and the hop is safe because it happens once,
/// on a single thread, before the protocol loop begins.
private func waitForQwenMTPAsync<T>(
    _ operation: @escaping () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = QwenMTPAsyncResultBox<T>()
    nonisolated(unsafe) let unsafeOperation = operation
    nonisolated(unsafe) let unsafeBox = box
    Task {
        do {
            unsafeBox.result = .success(try await unsafeOperation())
        } catch {
            unsafeBox.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "the MTP async model load completed without a result")
    }
    return try result.get()
}
