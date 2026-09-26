import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANECacheReplayProbeTests {
    @Test func rejectsInvalidQKPositions() {
        let qkv = [Float16](repeating: 0, count: 32 * 14336)
        for position in [-1, 1024, Int.max] {
            #expect(throws: QwenANEQKInputError.self) {
                try packQwenANEQKRows(qkv: qkv, positions: Array(repeating: position, count: 32))
            }
        }
    }

    @Test func realPrefixHistories() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_CACHE_REPLAY_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let withRerank = env["MLXFAST_ANE_CACHE_REPLAY_RERANK"] == "1"
        let withReusableDecoder = env["MLXFAST_ANE_CACHE_REPLAY_REUSABLE"] == "1"
        let withReadout = withRerank || env["MLXFAST_ANE_CACHE_REPLAY_READOUT"] == "1"
        let withEmbedding = withReusableDecoder || withReadout || env["MLXFAST_ANE_CACHE_REPLAY_EMBEDDING"] == "1"
        let withQK = withEmbedding || env["MLXFAST_ANE_CACHE_REPLAY_QK"] == "1"
        let withFront = withQK || env["MLXFAST_ANE_CACHE_REPLAY_FRONT"] == "1"
        let withTail = withFront || env["MLXFAST_ANE_CACHE_REPLAY_TAIL"] == "1"
        let withOutput = withTail || env["MLXFAST_ANE_CACHE_REPLAY_OUTPUT"] == "1"
        let stem = withReusableDecoder ? "retained-cache-reusable" : withRerank ? "retained-cache-rerank" : withReadout ? "retained-cache-readout" : withEmbedding ? "retained-cache-embedding" : withQK ? "retained-cache-qk" : withFront ? "retained-cache-front" : withTail ? "retained-cache-tail" : withOutput ? "retained-cache-output" : "retained-cache-attention"
        let qkPath = root.appendingPathComponent("checkpoint-prepared-qk.safetensors")
        let qkvPath = root.appendingPathComponent("checkpoint-qkv.safetensors")
        let referencePath = root.appendingPathComponent("checkpoint-attention.safetensors")
        let qk = try MLX.loadArrays(url: qkPath)
        let qkv = try MLX.loadArrays(url: qkvPath)
        let reference = try MLX.loadArrays(url: referencePath)
        let prepared = try #require(qk["ane_qk"])
        let projection = try #require(qkv["ane_qkv"])
        let previous = try #require(reference["ane_attention"])
        let gpu = try #require(reference["reference_attention"])
        let positions = try #require(qk["positions"])
        try #require(prepared.shape == [32, 28, 256] && projection.shape == [32, 14336])
        try #require(previous.shape == [32, 6144] && gpu.shape == previous.shape)
        try #require(positions.asArray(Int32.self) == (0..<32).map(Int32.init))
        let preparedValues = prepared.asArray(Float16.self), projectedValues = projection.asArray(Float16.self)
        try #require(preparedValues.allSatisfy { $0.isFinite } && projectedValues.allSatisfy { $0.isFinite })
        let capacity = 1024, channels = capacity * 2048
        func hash(_ path: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: path)).map { String(format: "%02x", $0) }.joined()
        }
        var report: [String: Any] = ["phase": "initialize", "lanes": 32, "capacity": capacity,
            "qk_sha256": try hash(qkPath), "qkv_sha256": try hash(qkvPath),
            "reference_sha256": try hash(referencePath),
            "scope": "32 stopped-prefix histories from one real coding seed; sequential retained ANE cache and attention; not independent corpus cases"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("\(stem).json"), options: .atomic)
        }
        try save()
        do {
            let loop = try ANEDirectDispatch.CacheLoop(batch: 32, capacity: capacity, attention: true)
            var outputLoop: ANEDirectDispatch.FP16Loop?
            var expectedOutput: [Float16] = []
            var tailLoop: ANEDirectDispatch.FP16Loop?
            var expectedHidden: [Float16] = []
            var residualValues: [Float16] = []
            var savedHidden = [Float16](repeating: 0, count: 32 * 5120)
            var inputLoop: ANEDirectDispatch.FP16Loop?
            var qkvLoop: ANEDirectDispatch.FP16Loop?
            var qkLoop: ANEDirectDispatch.FP16Loop?
            var seedInputs: [Float16] = []
            var embeddingLookup: QwenANEEmbedding?
            var seedTokens: [Int] = []
            if withOutput {
                let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
                let head = try MLX.loadArrays(url: source)
                if withFront {
                    func expandFront(_ prefix: String) throws -> MLXArray {
                        try aneProbeAffineWeight(#require(head[prefix + ".weight"]),
                            scales: #require(head[prefix + ".scales"]), biases: #require(head[prefix + ".biases"]), groupSize: 64, bits: 4)
                    }
                    let fc = try expandFront("fc")
                    try #require(fc.shape == [5120, 10240])
                    let fcBlob = buildConvWeightBlob(f16Bytes(fc))
                    let fcDigest = SHA256.hash(data: fcBlob).map { String(format: "%02x", $0) }.joined()
                    let eNorm = try #require(head["pre_fc_norm_embedding.weight"]).asArray(Float16.self)
                    let hNorm = try #require(head["pre_fc_norm_hidden.weight"]).asArray(Float16.self)
                    try #require(eNorm.count == 5120 && hNorm.count == 5120)
                    inputLoop = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEInputMILText(normWeights: eNorm + hNorm,
                        programTag: fcDigest), weightBlob: fcBlob, inputDim: 10240, outputDim: 5120, batch: 32)
                    let rows = try ["q", "k", "v"].map { try expandFront("layers.0.self_attn.\($0)_proj") }
                    let projection = concatenated(rows, axis: 0)
                    try #require(projection.shape == [14336, 5120])
                    let projectionBlob = buildConvWeightBlob(f16Bytes(projection))
                    let projectionDigest = SHA256.hash(data: projectionBlob).map { String(format: "%02x", $0) }.joined()
                    let norm = try #require(head["layers.0.input_layernorm.weight"]).asArray(Float16.self)
                    try #require(norm.count == 5120)
                    qkvLoop = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEProjectionMILText(norm: norm,
                        programTag: projectionDigest), weightBlob: projectionBlob, inputDim: 5120, outputDim: 14336, batch: 32)
                    let inputCapture = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-input.safetensors"))
                    let capturedInput = try #require(inputCapture["input"])
                    try #require(capturedInput.shape == [32, 10240])
                    seedInputs = capturedInput.asArray(Float16.self)
                    if withEmbedding {
                        seedTokens = try #require(inputCapture["input_tokens"]).asArray(Int32.self).map(Int.init)
                        try #require(seedTokens.count == 32 && seedTokens.allSatisfy { $0 >= 0 && $0 < 248320 })
                        let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
                        let indexData = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
                        let index = try #require(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
                        let mapping = try #require(index["weight_map"] as? [String: String])
                        let prefix = "language_model.model.embed_tokens"
                        let shard = try #require(mapping[prefix + ".weight"])
                        try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
                        let target = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
                        let packed = try #require(target[prefix + ".weight"])
                        let scales = try #require(target[prefix + ".scales"]), biases = try #require(target[prefix + ".biases"])
                        try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
                        let tiles = Set(seedTokens.map { $0 / 2048 }).sorted()
                        var blobs: [Int: Data] = [:]
                        for tile in tiles {
                            let start = tile * 2048, end = min(start + 2048, 248320)
                            var weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                                biases: biases[start..<end], groupSize: 64, bits: 4)
                            if end - start < 2048 {
                                weight = concatenated([weight, MLXArray.zeros([2048 - (end - start), 5120], dtype: .float16)], axis: 0)
                            }
                            blobs[tile] = buildConvWeightBlob(f16Bytes(weight.T))
                        }
                        embeddingLookup = try QwenANEEmbedding(tileWeightBlobs: blobs, maxLoadedTiles: 8)
                        report["embedding_tiles"] = tiles
                        report["embedding_source_scope"] = "Required seed tiles prepared offline; live ANE lookup each step; not full-vocabulary preload"
                    }
                    report["fc_weight_blob_sha256"] = fcDigest
                    report["qkv_weight_blob_sha256"] = projectionDigest
                    if withQK {
                        let qNorm = try #require(head["layers.0.self_attn.q_norm.weight"]).asArray(Float16.self)
                        let kNorm = try #require(head["layers.0.self_attn.k_norm.weight"]).asArray(Float16.self)
                        try #require(qNorm.count == 256 && kNorm.count == 256)
                        let weights = (0..<28).flatMap { $0 < 24 ? qNorm : kNorm }
                        let mil = buildQwenANEQKMILText(weights: weights)
                        qkLoop = try ANEDirectDispatch.FP16Loop(milText: mil, weightBlob: Data(),
                            inputDim: 7169, outputDim: 7168, batch: 32)
                        report["qk_mil_sha256"] = SHA256.hash(data: Data(mil.utf8)).map { String(format: "%02x", $0) }.joined()
                    }
                }
                let prefix = "layers.0.self_attn.o_proj"
                let weight = try aneProbeAffineWeight(#require(head[prefix + ".weight"]),
                    scales: #require(head[prefix + ".scales"]), biases: #require(head[prefix + ".biases"]),
                    groupSize: 64, bits: 4)
                try #require(weight.shape == [5120, 6144])
                let blob = buildConvWeightBlob(weight.asArray(Float16.self).withUnsafeBytes { Data($0) })
                let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
                outputLoop = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEGatedOutputMILText(programTag: digest),
                    weightBlob: blob, inputDim: 12288, outputDim: 5120, batch: 32)
                let capture = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-gated-output.safetensors"))
                expectedOutput = try #require(capture["ane_output"]).asArray(Float16.self)
                try #require(expectedOutput.count == 32 * 5120)
                report["output_weight_blob_sha256"] = digest
                if withTail {
                    func expand(_ name: String) throws -> MLXArray {
                        let key = "layers.0.mlp.\(name)_proj"
                        return try aneProbeAffineWeight(#require(head[key + ".weight"]),
                            scales: #require(head[key + ".scales"]), biases: #require(head[key + ".biases"]), groupSize: 64, bits: 4)
                    }
                    let gate = try expand("gate"), up = try expand("up"), down = try expand("down")
                    try #require(gate.shape == [17408, 5120] && up.shape == gate.shape && down.shape == [5120, 17408])
                    let (tailBlob, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gate), f16Bytes(up), f16Bytes(down)])
                    let tailDigest = SHA256.hash(data: tailBlob).map { String(format: "%02x", $0) }.joined()
                    let post = try #require(head["layers.0.post_attention_layernorm.weight"]).asArray(Float16.self)
                    let norm = try #require(head["norm.weight"]).asArray(Float16.self)
                    try #require(post.count == 5120 && norm.count == 5120)
                    tailLoop = try ANEDirectDispatch.FP16Loop(milText: buildQwenANEDecoderTailMILText(offsets: offsets,
                        postNorm: post, finalNorm: norm, programTag: tailDigest), weightBlob: tailBlob,
                        inputDim: 10240, outputDim: 5120, batch: 32, weightFileName: "weight.bin")
                    let hiddenCapture = try MLX.loadArrays(url: root.appendingPathComponent("checkpoint-decoder-tail.safetensors"))
                    expectedHidden = try #require(hiddenCapture["ane_hidden"]).asArray(Float16.self)
                    residualValues = try #require(qkv["residual"]).asArray(Float16.self)
                    try #require(expectedHidden.count == 32 * 5120 && residualValues.count == expectedHidden.count)
                    report["tail_weight_blob_sha256"] = tailDigest
                }
            }
            var savedProjection = [Float16](repeating: 0, count: 32 * 5120)
            var oracle = [Float16](repeating: 0, count: 32 * channels)
            var saved = [Float16](repeating: 0, count: 32 * 6144)
            var times: [Double] = []
            for step in 0..<32 {
                var currentFC: [Float16] = [], currentQKV: [Float16] = []
                if let inputLoop, let qkvLoop {
                    var input: [Float16] = []
                    let embeddings = try embeddingLookup?(tokens: (0..<32).map { seedTokens[min(step, $0)] })
                    for lane in 0..<32 {
                        let row = min(step, lane)
                        if let embeddings {
                            input.append(contentsOf: embeddings[(lane * 5120)..<((lane + 1) * 5120)])
                            input.append(contentsOf: seedInputs[(row * 10240 + 5120)..<((row + 1) * 10240)])
                        } else {
                            input.append(contentsOf: seedInputs[(row * 10240)..<((row + 1) * 10240)])
                        }
                    }
                    currentFC = try inputLoop(input)
                    currentQKV = try qkvLoop(currentFC)
                    for lane in 0..<32 {
                        let row = min(step, lane)
                        try #require(zip(currentFC[(lane * 5120)..<((lane + 1) * 5120)],
                            residualValues[(row * 5120)..<((row + 1) * 5120)]).allSatisfy { $0 == $1 })
                        try #require(zip(currentQKV[(lane * 14336)..<((lane + 1) * 14336)],
                            projectedValues[(row * 14336)..<((row + 1) * 14336)]).allSatisfy { $0 == $1 })
                    }
                }
                var currentQK: [Float16] = []
                if let qkLoop {
                    currentQK = try qkLoop(packQwenANEQKRows(qkv: currentQKV, positions: (0..<32).map { min(step, $0) }))
                    for lane in 0..<32 {
                        let row = min(step, lane)
                        try #require(zip(currentQK[(lane * 7168)..<((lane + 1) * 7168)],
                            preparedValues[(row * 7168)..<((row + 1) * 7168)]).allSatisfy { $0 == $1 })
                    }
                }
                var fresh: [Float16] = [], queries: [Float16] = []
                var slots: [Int] = [], lengths: [Int] = []
                for lane in 0..<32 {
                    let row = min(step, lane)
                    let qOffset = row * 28 * 256
                    let sourceQK = withQK ? currentQK : preparedValues
                    let liveOffset = withQK ? lane * 7168 : qOffset
                    queries.append(contentsOf: sourceQK[liveOffset..<(liveOffset + 6144)])
                    let keys = Array(sourceQK[(liveOffset + 6144)..<(liveOffset + 7168)])
                    let vOffset = row * 14336 + 13312
                    let values = withFront ? Array(currentQKV[(lane * 14336 + 13312)..<(lane * 14336 + 14336)])
                        : Array(projectedValues[vOffset..<(vOffset + 1024)])
                    fresh.append(contentsOf: keys + values)
                    slots.append(step <= lane ? step : -1)
                    lengths.append(row + 1)
                    if step <= lane {
                        let pair = keys + values
                        for head in 0..<8 {
                            let offset = lane * channels + (head * capacity + step) * 256
                            oracle.replaceSubrange(offset..<(offset + 256), with: pair[(head * 256)..<((head + 1) * 256)])
                        }
                    }
                }
                report["phase"] = "dispatch"; report["step"] = step; try save()
                let began = Date()
                let actual = try loop.step(newKV: fresh, slots: slots, queries: queries, validLengths: lengths)
                times.append(Date().timeIntervalSince(began))
                try #require(actual.count == 32 * 6144 && actual.allSatisfy { $0.isFinite })
                if let outputLoop {
                    var packed: [Float16] = []
                    for lane in 0..<32 {
                        packed.append(contentsOf: actual[(lane * 6144)..<((lane + 1) * 6144)])
                        let row = min(step, lane)
                        for head in 0..<24 {
                            let offset = row * 14336 + head * 512 + 256
                            if withFront {
                                let liveOffset = lane * 14336 + head * 512 + 256
                                packed.append(contentsOf: currentQKV[liveOffset..<(liveOffset + 256)])
                            } else {
                                packed.append(contentsOf: projectedValues[offset..<(offset + 256)])
                            }
                        }
                    }
                    let projected = try outputLoop(packed)
                    let range = (step * 5120)..<((step + 1) * 5120)
                    savedProjection.replaceSubrange(range, with: projected[range])
                    if let tailLoop {
                        var tailInput: [Float16] = []
                        for lane in 0..<32 {
                            let row = min(step, lane)
                            if withFront {
                                tailInput.append(contentsOf: currentFC[(lane * 5120)..<((lane + 1) * 5120)])
                            } else {
                                tailInput.append(contentsOf: residualValues[(row * 5120)..<((row + 1) * 5120)])
                            }
                            tailInput.append(contentsOf: projected[(lane * 5120)..<((lane + 1) * 5120)])
                        }
                        let hidden = try tailLoop(tailInput)
                        savedHidden.replaceSubrange(range, with: hidden[range])
                    }
                }
                for lane in 0...step {
                    let range = (lane * 6144)..<((lane + 1) * 6144)
                    if lane == step { saved.replaceSubrange(range, with: actual[range]) }
                    else { try #require(Array(actual[range]) == Array(saved[range])) }
                }
            }
            report["phase"] = "audit"; try save()
            let cache = try loop.snapshot()
            try #require(cache.count == oracle.count)
            let mismatches = zip(cache, oracle).filter { $0 != $1 }.count
            report["cache_mismatched_values"] = mismatches; try save()
            try #require(mismatches == 0)
            func errors(_ tensor: MLXArray) throws -> [String: Double] {
                let expected = tensor.asArray(Float.self)
                try #require(expected.count == saved.count && expected.allSatisfy { $0.isFinite })
                var squared = 0.0, energy = 0.0, maximum = 0.0
                for (a, e) in zip(saved, expected) {
                    let difference = Double(a) - Double(e)
                    squared += difference * difference; energy += Double(e) * Double(e)
                    maximum = max(maximum, abs(difference))
                }
                return ["relative_rms_error": sqrt(squared / max(energy, Double.leastNormalMagnitude)), "max_abs_error": maximum]
            }
            if withOutput {
                let differences = zip(savedProjection, expectedOutput).filter { $0 != $1 }.count
                report["projection_mismatched_values"] = differences; try save()
                try #require(differences == 0)
            }
            if withTail {
                let differences = zip(savedHidden, expectedHidden).filter { $0 != $1 }.count
                report["hidden_mismatched_values"] = differences; try save()
                try #require(differences == 0)
            }
            let local = try errors(previous)
            report["versus_batched_ane"] = local
            report["versus_gpu_reference"] = try errors(gpu)
            report["stage_dispatch_read_seconds"] = times
            var outputs = ["ane_attention": MLXArray(saved, [32, 6144]), "reference_attention": gpu]
            if withOutput { outputs["ane_output"] = MLXArray(savedProjection, [32, 5120]) }
            if withTail { outputs["ane_hidden"] = MLXArray(savedHidden, [32, 5120]) }
            if withReusableDecoder {
                report["phase"] = "reusable-decoder-initialize"; try save()
                let decoder = try QwenANEDecoder(inputProjection: #require(inputLoop),
                    qkvProjection: #require(qkvLoop), qkPreparation: #require(qkLoop),
                    outputProjection: #require(outputLoop), tail: #require(tailLoop))
                let lookup = try #require(embeddingLookup)
                #expect(throws: QwenANEDecoderError.self) {
                    try decoder(input: [], positions: [], slots: [], validLengths: [])
                }
                var collected = [Float16](repeating: 0, count: 32 * 5120)
                for step in 0..<32 {
                    let positions = (0..<32).map { min(step, $0) }
                    let embeddings = try lookup(tokens: positions.map { seedTokens[$0] })
                    var input: [Float16] = []
                    for lane in 0..<32 {
                        let row = positions[lane]
                        input.append(contentsOf: embeddings[(lane * 5120)..<((lane + 1) * 5120)])
                        input.append(contentsOf: seedInputs[(row * 10240 + 5120)..<((row + 1) * 10240)])
                    }
                    report["phase"] = "reusable-decoder-dispatch"; report["reusable_step"] = step; try save()
                    let hidden = try decoder(input: input, positions: positions,
                        slots: (0..<32).map { step <= $0 ? step : -1 }, validLengths: positions.map { $0 + 1 })
                    for lane in 0...step {
                        let range = (lane * 5120)..<((lane + 1) * 5120)
                        try #require(Array(hidden[range]) == Array(savedHidden[range]))
                    }
                    let range = (step * 5120)..<((step + 1) * 5120)
                    collected.replaceSubrange(range, with: hidden[range])
                }
                let differences = try zip(decoder.snapshot(), oracle).filter { $0 != $1 }.count
                try #require(differences == 0 && collected == savedHidden)
                report["reusable_cache_mismatched_values"] = differences
                report["reusable_hidden_exact_values"] = collected.count
            }
            if withReadout {
                report["phase"] = "readout-prepare"; try save()
                let source = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_PROJECTION_HEAD"]))
                let head = try MLX.loadArrays(url: source)
                let packed = try #require(head["draft_lm_head.weight"])
                let scales = try #require(head["draft_lm_head.scales"]), biases = try #require(head["draft_lm_head.biases"])
                try #require(packed.shape == [98336, 320] && scales.shape == [98336, 80] && biases.shape == scales.shape)
                var blobs: [Int: Data] = [:]
                for start in stride(from: 0, to: 98336, by: 2048) {
                    let end = min(start + 2048, 98336)
                    let weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                        biases: biases[start..<end], groupSize: 64, bits: 2)
                    blobs[start] = buildConvWeightBlob(f16Bytes(weight))
                }
                report["phase"] = "readout-load"; try save()
                let readout = try QwenANECoarseReadout(tileWeightBlobs: blobs)
                let selector = try QwenANEShortlistSelector()
                report["phase"] = "readout-dispatch"; try save()
                let logits = try readout(savedHidden)
                let selected = try selector(logits)
                let control = try MLX.loadArrays(url: root.appendingPathComponent("selection-readout.safetensors"))
                let expectedIDs = try #require(control["compact_ids"]).asArray(Int32.self)
                let expectedScores = try #require(control["scores"]).asArray(Float16.self)
                try #require(selected.ids.count == 1024 && selected.scores.count == 1024)
                try #require(selected.ids == expectedIDs && selected.scores == expectedScores)
                report["shortlist_exact_entries"] = 1024
                outputs["compact_ids"] = MLXArray(selected.ids, [32, 32])
                outputs["shortlist_scores"] = MLXArray(selected.scores, [32, 32])
                if withRerank {
                    report["phase"] = "rerank-prepare"; try save()
                    let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
                    let data = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
                    let index = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
                    let mapping = try #require(index["weight_map"] as? [String: String])
                    let prefix = "language_model.lm_head"
                    let shard = try #require(mapping[prefix + ".weight"])
                    try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
                    let target = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
                    let packed = try #require(target[prefix + ".weight"])
                    let scales = try #require(target[prefix + ".scales"]), biases = try #require(target[prefix + ".biases"])
                    try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
                    var targetBlobs: [Int: Data] = [:], digests: [String: String] = [:]
                    for start in stride(from: 0, to: 98336, by: 2048) {
                        let begin = start < 98304 ? start : 248044
                        let end = start < 98304 ? start + 2048 : 248070
                        var weight = aneProbeAffineWeight(packed[begin..<end], scales: scales[begin..<end],
                            biases: biases[begin..<end], groupSize: 64, bits: 4)
                        if start == 98304 {
                            weight = concatenated([weight, MLXArray.zeros([6, 5120], dtype: .float16)], axis: 0)
                        }
                        let blob = buildConvWeightBlob(f16Bytes(weight))
                        targetBlobs[start] = blob
                        digests[String(start)] = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
                        report["rerank_prepared_tiles"] = targetBlobs.count; try save()
                    }
                    report["target_tile_sha256"] = digests
                    report["phase"] = "rerank-load"; try save()
                    let reranker = try QwenANEReranker(targetTileWeightBlobs: targetBlobs)
                    #expect(throws: QwenANERerankerError.self) { try reranker(hidden: savedHidden, compactIDs: []) }
                    report["phase"] = "rerank-dispatch"; try save()
                    let result = try reranker(hidden: savedHidden, compactIDs: selected.ids)
                    let controlData = try Data(contentsOf: root.appendingPathComponent("checkpoint-rerank.json"))
                    let control = try #require(JSONSerialization.jsonObject(with: controlData) as? [String: Any])
                    let previousTokens = try #require(control["ane_tokens"] as? [Int]).map(Int32.init)
                    let referenceTokens = try #require(control["reference_tokens"] as? [Int]).map(Int32.init)
                    try #require(result.tokens.count == 32 && previousTokens.count == 32 && referenceTokens.count == 32)
                    report["final_tokens"] = result.tokens.map(Int.init)
                    report["final_matches_prior_ane"] = zip(result.tokens, previousTokens).filter { $0 == $1 }.count
                    report["final_matches_gpu_reference"] = zip(result.tokens, referenceTokens).filter { $0 == $1 }.count
                    try save()
                    try #require(result.tokens == previousTokens)
                    outputs["final_tokens"] = MLXArray(result.tokens)
                    outputs["rerank_scores"] = MLXArray(result.scores, [32, 32])
                }
            }
            try MLX.save(arrays: outputs,
                url: root.appendingPathComponent("\(stem).safetensors"))
            if withFront { report["live_fc_qkv_exact_checks"] = 32 * 32 }
            if withQK { report["live_qk_exact_checks"] = 32 * 32 }
            if withEmbedding { report["live_embedding_lookups"] = 32 * 32 }
            report["phase"] = "complete"; try save()
            try #require((local["relative_rms_error"] ?? .infinity) < 0.005)
        } catch {
            report["error"] = String(describing: error); try save(); throw error
        }
    }
}
