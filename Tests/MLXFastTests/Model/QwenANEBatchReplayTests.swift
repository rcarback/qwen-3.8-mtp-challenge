import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers
@testable import MLXLLM
@testable import MLXFastCore
@testable import MLXFastModel

@Suite(.serialized)
struct QwenANEBatchReplayTests {
    private struct Frame: Decodable {
        let inputTokens: [Int]
        let targetTop2Tokens: [Int]
    }
    private struct Archive: Decodable { let frames: [Frame] }
    private struct Lane {
        let id: String
        let norm: MLXArray
        let frames: [Frame]
        let tensors: [String: MLXArray]
        let expected: [[String: Any]]
    }
    private final class QueuedLane {
        let lane: Lane
        let limit: Int
        let gpuCache: [any KVCache]
        let aneCache: [any KVCache]
        var cursor = 0
        var gpu: [[Int]] = []
        var ane: [[Int]] = []

        init(_ lane: Lane, limit: Int, gpuCache: [any KVCache], aneCache: [any KVCache]) {
            self.lane = lane
            self.limit = limit
            self.gpuCache = gpuCache
            self.aneCache = aneCache
        }
    }

    @Test("independent MTP lanes preserve GPU reference before ANE substitution")
    func probe() throws {
        let env = ProcessInfo.processInfo.environment
        let fullFidelity = env["MLXFAST_GPU_BATCH_FULL_FIDELITY"] == "1"
        let aneDiagnostics = env["MLXFAST_ANE_DIAGNOSTICS"] == "1"
        let capture = env["MLXFAST_ANE_CAPTURE"] == "1"
        let matchedReplay = env["MLXFAST_ANE_MATCHED_REPLAY"] == "1"
        let hybridPilot = env["MLXFAST_HYBRID_TRAINING_PILOT"] == "1"
        let hybridTraining = env["MLXFAST_HYBRID_TRAINING"] == "1" || hybridPilot
        let poolProbe = env["MLXFAST_HYBRID_POOL_PROBE"] == "1"
        let queueProbe = env["MLXFAST_HYBRID_QUEUE_PROBE"] == "1"
        let gpuProbe = fullFidelity || env["MLXFAST_GPU_BATCH_PROBE"] == "1"
        guard gpuProbe || aneDiagnostics || capture || matchedReplay || hybridTraining || poolProbe || queueProbe || env["MLXFAST_ANE_BATCH_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
        let native = root.appendingPathComponent("native-training")
        let outputName = matchedReplay ? "matched-replay.json" : (capture ? "capture-result.json" : (fullFidelity ? "gpu-full-result.json" : (gpuProbe ? "gpu-result.json" : (aneDiagnostics ? "diagnostics.json" : "result.json"))))
        let output = root.appendingPathComponent("ane-batch/\(queueProbe ? "queue-probe.json" : (poolProbe ? "pool-probe.json" : outputName))")
        func json(_ url: URL) throws -> Any {
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        }
        if hybridTraining && !hybridPilot {
            try #require(FileManager.default.fileExists(atPath: native.appendingPathComponent("export/COMPLETE").path))
        }
        let headsURL = hybridTraining ? root.appendingPathComponent("campaign-prefix/heads.json") : native.appendingPathComponent("reference-pilot/heads.json")
        let heads = try #require(try json(headsURL) as? [[String: Any]])
        let manifests = try #require(try json(native.appendingPathComponent("export/manifest.json")) as? [[String: Any]])
        let targetURL = URL(fileURLWithPath: try #require(env["MLXFAST_QWEN_PREFILL_WEIGHTS"]))
        let headURL = URL(fileURLWithPath: try #require(env["MLXFAST_QWEN_PREFILL_HEAD"]))
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL, headDirectory: headURL
        ) { _ in
            let box = UnsafeSendableBox<Result<ModelContext, Error>>()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                do { box.value = .success(try await LLMModelFactory.shared.load(
                    from: targetURL, using: #huggingFaceTokenizerLoader())) }
                catch { box.value = .failure(error) }
                semaphore.signal()
            }
            semaphore.wait()
            return try #require(box.value).get()
        }
        let model = try #require(context.model as? MLXLLM.Qwen35TextModel)
        let mtp = try #require(model.mtp)
        try #require(mtp.layers.count == 1)
        let mlp = try #require(mtp.layers[0].mlp as? Qwen35FusedMLP)
        eval(model)
        var lanes: [Lane] = []
        let caseIndices = (poolProbe || queueProbe) ? [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 1, 17, 34, 49, 65, 81] : (hybridTraining ? [0] : (fullFidelity ? Array(stride(from: 0, to: 160, by: 16)) : [0, 32, 128]))
        func loadCase(_ index: Int) throws -> [Lane] {
            var loaded: [Lane] = []
            let record = try #require(manifests.first { $0["index"] as? Int == index })
            let directory = URL(fileURLWithPath: try #require(record["archive"] as? String))
            let archive = try JSONDecoder().decode(Archive.self, from: Data(contentsOf: directory.appendingPathComponent("frames.json")))
            let tensors = try MLX.loadArrays(url: directory.appendingPathComponent("frames.safetensors"))
            let reportPath = try #require(record["report"] as? String)
            let reference = try #require(try json(URL(fileURLWithPath: reportPath)) as? [String: [String: Any]])
            for head in heads {
                let id = try #require(head["id"] as? String)
                let path = try #require(head["norm_path"] as? String)
                let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
                loaded.append(Lane(id: "\(index)/\(id)", norm: try #require(arrays["norm.weight"]),
                    frames: archive.frames, tensors: tensors,
                    expected: try #require(reference[hybridTraining ? "-1" : id]?["rows"] as? [[String: Any]])))
            }
            return loaded
        }
        for index in caseIndices { lanes.append(contentsOf: try loadCase(index)) }
        var steps = 48
        let width = 32
        try #require((gpuProbe || poolProbe || queueProbe || lanes.count <= width) && lanes.allSatisfy { $0.frames.count > steps })
        var evidence: [String: Any] = ["lanes": lanes.count, "steps": steps,
            "batch_width": width, "phase": "gpu-control", "proposal_bits": qwen35HeadProposalQuantizationBits ?? 0]
        func save() throws {
            try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
                .write(to: output, options: .atomic)
        }
        var distributedGPUCount: Int?
        var program: MLXLLM.ANEFusedMLP?
        var dispatches = 0
        var gpuBatchSize = 1
        var gpuMLPBatches = 0
        var denseWeights: [MLXArray] = []
        var matchedWeights: [MLXArray] = []
        var capturedInputs: [MLXArray] = []
        var capturedOutputs: [MLXArray] = []
        let denseActivation = compile { (y: MLXArray) -> MLXArray in
            let half = y.dim(-1) / 2
            return silu(y[.ellipsis, ..<half]) * y[.ellipsis, half...]
        }
        func norm(_ lane: Lane) {
            mtp.norm.update(parameters: ModuleParameters.unflattened(["weight": lane.norm]))
        }
        func finish(_ parts: [(MLXArray, MLXArray)], mode: String) throws -> [MLXArray] {
            if mode == "paired-engines" {
                try #require(parts.count > 0)
                let half = distributedGPUCount ?? parts.count / 2
                try #require(half >= 0 && half <= parts.count)
                let aneCount = parts.count - half
                var gpuOutputs: [MLXArray] = []
                var aneOutputs: [MLXArray] = []
                for start in stride(from: 0, to: max(half, aneCount), by: width) {
                    let gpuParts = Array(parts[min(start, half)..<min(start + width, half)])
                    let aneParts = Array(parts[(half + min(start, aneCount))..<(half + min(start + width, aneCount))])
                    if gpuParts.isEmpty {
                        aneOutputs.append(contentsOf: try finish(aneParts, mode: "ane"))
                        continue
                    }
                    if aneParts.isEmpty {
                        gpuOutputs.append(contentsOf: try finish(gpuParts, mode: "matched-gpu"))
                        continue
                    }
                    let aneInput = concatenated(aneParts.map { $0.1.reshaped([1, 5120]) }, axis: 0)
                    let padded = concatenated([aneInput, MLXArray.zeros([width - aneParts.count, 5120], dtype: aneInput.dtype)], axis: 0)
                    let loaded = try #require(program)
                    let prepared = try loaded.makeInput(padded.asType(.float16))
                    let (_, gpuResult) = try ConcurrentEngines.run(
                        ane: { try loaded.predict(prepared) },
                        gpu: { try finish(gpuParts, mode: "matched-gpu") })
                    let copied = (loaded.readOutput(prepared) + MLXArray(Float16(0))).asType(.bfloat16)
                    eval(copied)
                    gpuOutputs.append(contentsOf: gpuResult)
                    aneOutputs.append(contentsOf: aneParts.indices.map { copied[$0..<$0 + 1].reshaped([1, 1, 5120]) })
                }
                return gpuOutputs + aneOutputs
            }
            if (mode == "matched-gpu" || mode == "ane") && parts.count > width {
                var outputs: [MLXArray] = []
                for start in stride(from: 0, to: parts.count, by: width) {
                    outputs.append(contentsOf: try finish(Array(parts[start..<min(start + width, parts.count)]), mode: mode))
                }
                return outputs
            }
            if mode == "split-gpu" {
                let outputs = parts.map { mlp($0.1) }
                if capture {
                    capturedInputs.append(contentsOf: parts.map { $0.1.reshaped([1, 5120]) })
                    capturedOutputs.append(contentsOf: outputs.map { $0.reshaped([1, 5120]) })
                    eval(outputs)
                }
                return outputs
            }
            if mode.hasPrefix("dense-") {
                try #require(denseWeights.count == 2)
                let dtype: DType = mode == "dense-bf16" ? .bfloat16 : .float16
                return parts.map { part in
                    let projected = matmul(part.1.asType(dtype), denseWeights[0].T)
                    return matmul(denseActivation(projected), denseWeights[1].T).asType(.bfloat16)
                }
            }
            if mode == "batched-gpu" {
                var outputs: [MLXArray] = []
                for start in stride(from: 0, to: parts.count, by: gpuBatchSize) {
                    let end = min(start + gpuBatchSize, parts.count)
                    let batch = concatenated(parts[start..<end].map { $0.1 }, axis: 1)
                    let result = mlp(batch)
                    // MLX eval materializes tensors; it does not evaluate source code.
                    eval(result)
                    gpuMLPBatches += 1
                    for row in 0..<(end - start) {
                        outputs.append(result[0..., row..<(row + 1), 0...])
                    }
                }
                return outputs
            }
            let input = concatenated(parts.map { $0.1.reshaped([1, 5120]) }, axis: 0)
            let padded = concatenated([input, MLXArray.zeros([width - parts.count, 5120], dtype: input.dtype)], axis: 0)
            if mode == "matched-gpu" {
                try #require(matchedWeights.count == 3)
                let x = padded.asType(.float16)
                func materialized(_ value: MLXArray) -> MLXArray {
                    eval(value)
                    return value
                }
                let gate = materialized(matmul(x, matchedWeights[0].T))
                let up = materialized(matmul(x, matchedWeights[1].T))
                let negative = materialized(gate * MLXArray(Float16(-1)))
                let exponential = materialized(exp(negative))
                let denominator = materialized(exponential + MLXArray(Float16(1)))
                let activated = materialized(materialized(gate / denominator) * up)
                let output = materialized(matmul(activated, matchedWeights[2].T).asType(.bfloat16))
                return parts.indices.map { output[$0..<$0 + 1].reshaped([1, 1, 5120]) }
            }
            let loadedProgram = try #require(program)
            let result = try loadedProgram(padded.asType(.float16))
            // Materialize each dispatch before its reusable IOSurface is restaged.
            let copied = (result + MLXArray(Float16(0))).asType(.bfloat16)
            eval(copied)
            dispatches += 1
            return parts.indices.map { copied[$0..<$0 + 1].reshaped([1, 1, 5120]) }
        }
        func run(_ mode: String, starts: [Int]? = nil, savedCaches: [[any KVCache]]? = nil, count: Int? = nil) throws -> [[[Int]]] {
            let caches = savedCaches ?? lanes.map { _ in model.makeMTPCache() }
            try #require(caches.count == lanes.count && (starts == nil || starts?.count == lanes.count))
            var predictions = lanes.map { _ in [[Int]]() }
            for step in 0..<(count ?? steps) {
                var hidden: [MLXArray] = []
                var first: [Int] = []
                var parts: [(MLXArray, MLXArray)] = []
                for (i, lane) in lanes.enumerated() {
                    norm(lane)
                    let frameIndex = (starts?[i] ?? 0) + step
                    let frame = lane.frames[frameIndex]
                    let x = try #require(lane.tensors[String(frameIndex)])
                    let ids = MLXArray(frame.inputTokens.map(Int32.init)).reshaped([1, frame.inputTokens.count])
                    if mode == "canonical" {
                        let h = model.mtpHeadLastHiddenWithKVOnlyHistory(hidden: x, nextTokenIds: ids, cache: caches[i])
                            ?? model.mtpHeadHiddenForward(hidden: x, nextTokenIds: ids, cache: caches[i])
                        hidden.append(h[0..., (h.dim(1)-1)..<h.dim(1), 0...])
                        first.append(model.draftTokenID(hidden[i]).item(Int.self))
                    } else {
                        parts.append(try #require(mtp.prepareIndependentReplayMLP(hidden: x,
                            nextTokenIds: ids, embedTokens: model.model.embedTokens, cache: caches[i])))
                    }
                }
                if mode != "canonical" {
                    let outputs = try finish(parts, mode: mode)
                    for (i, lane) in lanes.enumerated() {
                        norm(lane)
                        hidden.append(mtp.norm(parts[i].0 + outputs[i]))
                        first.append(model.draftTokenID(hidden[i]).item(Int.self))
                    }
                }
                let offsets = caches.map { $0.map { $0.offset } }
                parts = []
                var second: [Int] = []
                for (i, lane) in lanes.enumerated() {
                    norm(lane)
                    let ids = MLXArray([Int32(first[i])]).reshaped([1, 1])
                    if mode == "canonical" {
                        let h = model.mtpHeadHiddenForward(hidden: hidden[i], nextTokenIds: ids, cache: caches[i])
                        second.append(model.draftTokenID(h).item(Int.self))
                    } else {
                        parts.append(try #require(mtp.prepareIndependentReplayMLP(hidden: hidden[i],
                            nextTokenIds: ids, embedTokens: model.model.embedTokens, cache: caches[i])))
                    }
                }
                if mode != "canonical" {
                    let outputs = try finish(parts, mode: mode)
                    for (i, lane) in lanes.enumerated() {
                        norm(lane)
                        second.append(model.draftTokenID(mtp.norm(parts[i].0 + outputs[i])).item(Int.self))
                    }
                }
                for i in lanes.indices {
                    eval(caches[i].flatMap { $0.state })
                    for (entry, old) in zip(caches[i], offsets[i]) {
                        try #require(entry.offset == old + 1 && entry.isTrimmable)
                        try #require(entry.trim(1) == 1 && entry.offset == old)
                    }
                    predictions[i].append([first[i], second[i]])
                    if mode == "canonical" {
                        let frameIndex = (starts?[i] ?? 0) + step
                        try #require(first[i] == lanes[i].expected[frameIndex]["draft_token"] as? Int)
                        try #require(second[i] == lanes[i].expected[frameIndex]["second_draft_token"] as? Int)
                    }
                }
            }
            return predictions
        }
        func runQueue(next: () throws -> (Lane, Int)?, completed: (Lane, [[Int]], [[Int]]) throws -> Void) throws -> [String: Int] {
            var active: [QueuedLane] = []
            var exhausted = false
            var seen: Set<String> = []
            var finished = 0
            var ticks = 0
            var occupied = 0
            var peak = 0
            while true {
                while active.count < width && !exhausted {
                    guard let (lane, limit) = try next() else { exhausted = true; break }
                    try #require(limit > 0 && limit < lane.frames.count && seen.insert(lane.id).inserted)
                    active.append(QueuedLane(lane, limit: limit, gpuCache: model.makeMTPCache(), aneCache: model.makeMTPCache()))
                }
                if active.isEmpty { break }
                let laneCount = active.count
                peak = max(peak, laneCount)
                occupied += laneCount
                ticks += 1
                lanes = active.map { $0.lane } + active.map { $0.lane }
                let starts = active.map { $0.cursor } + active.map { $0.cursor }
                let caches = active.map { $0.gpuCache } + active.map { $0.aneCache }
                let predictions = try run("paired-engines", starts: starts, savedCaches: caches, count: 1)
                for (index, job) in active.enumerated() {
                    job.gpu.append(predictions[index][0])
                    job.ane.append(predictions[laneCount + index][0])
                    job.cursor += 1
                    if job.cursor == job.limit {
                        try completed(job.lane, job.gpu, job.ane)
                        finished += 1
                    }
                }
                active.removeAll { $0.cursor == $0.limit }
            }
            return ["finished_lanes": finished, "peak_slots": peak, "ticks": ticks, "occupied_slot_ticks": occupied]
        }
        func runDistributedQueue(next: (Int) throws -> (Lane, Int)?, completed: (Int, Lane, [[Int]]) throws -> Void) throws -> [String: Int] {
            var active: [[QueuedLane]] = [[], []]
            var exhausted = [false, false]
            var seen: Set<String> = []
            var finished = 0
            var ticks = 0
            var peak = 0
            defer { distributedGPUCount = nil }
            while true {
                for engine in 0..<2 {
                    while active[engine].count < width && !exhausted[engine] {
                        guard let (lane, limit) = try next(engine) else { exhausted[engine] = true; break }
                        try #require(limit > 0 && limit < lane.frames.count && seen.insert(lane.id).inserted)
                        active[engine].append(QueuedLane(lane, limit: limit, gpuCache: model.makeMTPCache(), aneCache: []))
                    }
                }
                let jobs = active[0] + active[1]
                if jobs.isEmpty { break }
                peak = max(peak, jobs.count)
                ticks += 1
                lanes = jobs.map { $0.lane }
                distributedGPUCount = active[0].count
                let predictions = try run("paired-engines", starts: jobs.map { $0.cursor }, savedCaches: jobs.map { $0.gpuCache }, count: 1)
                let gpuCount = active[0].count
                for (index, job) in jobs.enumerated() {
                    job.gpu.append(predictions[index][0])
                    job.cursor += 1
                    if job.cursor == job.limit {
                        try completed(index < gpuCount ? 0 : 1, job.lane, job.gpu)
                        finished += 1
                    }
                }
                for engine in 0..<2 { active[engine].removeAll { $0.cursor == $0.limit } }
            }
            return ["finished_lanes": finished, "peak_distinct_slots": peak, "ticks": ticks]
        }
        if queueProbe {
            let allLanes = lanes
            let limits = [5, 11, 17, 23, 29, 35, 41, 48]
            matchedWeights = [effective(mlp.gateProj).asType(.float16), effective(mlp.upProj).asType(.float16),
                effective(mlp.downProj).asType(.float16)]
            program = try MLXLLM.ANEFusedMLP(hidden: 5120, innerFraction: mlp.gateProj.weight.dim(0),
                sequenceLength: width, gate: matchedWeights[0], up: matchedWeights[1], down: matchedWeights[2],
                activation: .expDiv, weightForm: .fp16)
            var expectedGPU: [String: [[Int]]] = [:]
            var expectedANE: [String: [[Int]]] = [:]
            for casePosition in caseIndices.indices {
                lanes = Array(allLanes[(casePosition * heads.count)..<((casePosition + 1) * heads.count)])
                steps = limits[casePosition % limits.count]
                _ = try run("canonical")
                let gpu = try run("matched-gpu")
                let ane = try run("ane")
                for i in lanes.indices { expectedGPU[lanes[i].id] = gpu[i]; expectedANE[lanes[i].id] = ane[i] }
            }
            var nextIndex = 0
            var completedIDs: Set<String> = []
            let started = Date()
            let stats = try runQueue(next: {
                guard nextIndex < allLanes.count else { return nil }
                let index = nextIndex
                nextIndex += 1
                return (allLanes[index], limits[(index / heads.count) % limits.count])
            }, completed: { lane, gpu, ane in
                try #require(completedIDs.insert(lane.id).inserted)
                try #require(gpu == expectedGPU[lane.id] && ane == expectedANE[lane.id])
            })
            try #require(completedIDs == Set(allLanes.map { $0.id }) && stats["peak_slots"] == 32)
            evidence["stats"] = stats
            evidence["queue_seconds"] = Date().timeIntervalSince(started)
            evidence["exact_vs_serial"] = true
            evidence["phase"] = "complete"
            try save()
            return
        }
        if poolProbe {
            let allLanes = lanes
            matchedWeights = [effective(mlp.gateProj).asType(.float16), effective(mlp.upProj).asType(.float16),
                effective(mlp.downProj).asType(.float16)]
            program = try MLXLLM.ANEFusedMLP(hidden: 5120, innerFraction: mlp.gateProj.weight.dim(0),
                sequenceLength: width, gate: matchedWeights[0], up: matchedWeights[1], down: matchedWeights[2],
                activation: .expDiv, weightForm: .fp16)
            var trials: [[String: Any]] = []
            for caseCount in [1, 2, 4, 7, 8, 16] {
                let selected = Array(allLanes.prefix(caseCount * heads.count))
                lanes = selected
                _ = try run("canonical")
                let gpuControl = try run("matched-gpu")
                let aneControl = try run("ane")
                lanes = selected + selected
                try #require(try run("paired-engines") == gpuControl + aneControl)
                for repeatIndex in 0..<3 {
                    for mode in (repeatIndex % 2 == 0 ? ["serial", "paired"] : ["paired", "serial"]) {
                        let started = Date()
                        let values: [[[Int]]]
                        if mode == "serial" {
                            lanes = selected
                            values = try run("matched-gpu") + run("ane")
                        } else {
                            lanes = selected + selected
                            values = try run("paired-engines")
                        }
                        let seconds = Date().timeIntervalSince(started)
                        try #require(values == gpuControl + aneControl)
                        trials.append(["cases": caseCount, "lanes_per_engine": selected.count,
                            "repeat": repeatIndex, "mode": mode, "seconds": seconds,
                            "draft_rows": selected.count * steps * 4, "exact_vs_serial": true])
                        evidence["trials"] = trials
                        evidence["phase"] = "pool-throughput"
                        try save()
                    }
                }
            }
            evidence["phase"] = "complete"
            try save()
            return
        }
        if hybridTraining {
            try #require(heads.count == 28 && manifests.count == 160)
            let fingerprint = try #require(env["MLXFAST_HYBRID_FINGERPRINT"])
            let caseCount = hybridPilot ? 4 : 160
            let maximumSteps = hybridPilot ? 48 : Int.max
            let directory = root.appendingPathComponent(hybridPilot ? "hybrid-training-pilot" : "hybrid-training")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            matchedWeights = [effective(mlp.gateProj).asType(.float16), effective(mlp.upProj).asType(.float16),
                effective(mlp.downProj).asType(.float16)]
            program = try MLXLLM.ANEFusedMLP(hidden: 5120, innerFraction: mlp.gateProj.weight.dim(0),
                sequenceLength: width, gate: matchedWeights[0], up: matchedWeights[1], down: matchedWeights[2],
                activation: .expDiv, weightForm: .fp16)
            var nextCase = [0, 1]
            var pending: [[Lane]] = [[], []]
            var caseInputs: [Int: (ids: [String], baseline: [[Int]], steps: Int, started: Date)] = [:]
            var casePredictions: [Int: [String: [[Int]]]] = [:]
            let queueStarted = Date()
            let stats = try runDistributedQueue(next: { engine in
                while pending[engine].isEmpty && nextCase[engine] < caseCount {
                    let index = nextCase[engine]
                    nextCase[engine] += 2
                    let loaded = try loadCase(index)
                    let frameCount = loaded[0].frames.count
                    let limit = min(hybridPilot ? [48, 23, 35, 41][index] : maximumSteps, frameCount - 1)
                    try #require(limit > 0 && loaded.allSatisfy { $0.frames.count == frameCount && $0.expected.count == frameCount })
                    let baseline = try (0..<limit).map { step in
                        [try #require(loaded[0].expected[step]["draft_token"] as? Int),
                         try #require(loaded[0].expected[step]["second_draft_token"] as? Int)]
                    }
                    let path = directory.appendingPathComponent(String(format: "case-%04d.json", index))
                    if FileManager.default.fileExists(atPath: path.path) {
                        let previous = try #require(try json(path) as? [String: Any])
                        try #require(previous["fingerprint"] as? String == fingerprint)
                        try #require(previous["original_baseline"] as? [[Int]] == baseline)
                        try #require(previous["lane_ids"] as? [String] == loaded.map { $0.id })
                        try #require(previous["index"] as? Int == index && previous["steps"] as? Int == limit)
                        try #require(previous["engine"] as? String == (engine == 0 ? "matched_gpu" : "ane"))
                        let values = try #require(previous["predictions"] as? [[[Int]]])
                        try #require(values.count == 28 && values.allSatisfy { $0.count == limit && $0.allSatisfy { $0.count == 2 } })
                        continue
                    }
                    if index == 0 {
                        lanes = [try #require(loaded.first { $0.id == "0/-1" })]
                        try #require(try run("canonical", count: limit) == [baseline])
                    }
                    caseInputs[index] = (loaded.map { $0.id }, baseline, limit, Date())
                    pending[engine] = Array(loaded.reversed())
                }
                guard let lane = pending[engine].popLast() else { return nil }
                let index = try #require(Int(lane.id.split(separator: "/")[0]))
                return (lane, try #require(caseInputs[index]).steps)
            }, completed: { engine, lane, predictions in
                let index = try #require(Int(lane.id.split(separator: "/")[0]))
                var collected = casePredictions[index] ?? [:]
                try #require(collected[lane.id] == nil)
                collected[lane.id] = predictions
                casePredictions[index] = collected
                guard collected.count == 28 else { return }
                let source = try #require(caseInputs[index])
                let predictions = try source.ids.map { try #require(collected[$0]) }
                let engineName = engine == 0 ? "matched_gpu" : "ane"
                if hybridPilot {
                    lanes = try loadCase(index)
                    try #require(try run(engineName == "ane" ? "ane" : "matched-gpu", count: source.steps) == predictions)
                }
                let result: [String: Any] = ["index": index, "fingerprint": fingerprint,
                    "lane_ids": source.ids, "steps": source.steps, "original_baseline": source.baseline,
                    "engine": engineName, "predictions": predictions, "case_wall_seconds": Date().timeIntervalSince(source.started)]
                let path = directory.appendingPathComponent(String(format: "case-%04d.json", index))
                try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                    .write(to: path, options: .atomic)
                caseInputs.removeValue(forKey: index)
                casePredictions.removeValue(forKey: index)
                print("Hybrid training case \(index + 1)/\(caseCount) saved")
            })
            try #require(nextCase.allSatisfy { $0 >= caseCount } && pending.allSatisfy { $0.isEmpty } && caseInputs.isEmpty && casePredictions.isEmpty)
            try JSONSerialization.data(withJSONObject: ["stats": stats, "seconds": Date().timeIntervalSince(queueStarted)], options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("queue-stats.json"), options: .atomic)
            try Data(fingerprint.utf8).write(to: directory.appendingPathComponent("COMPLETE"), options: .atomic)
            return
        }
        if fullFidelity {
            evidence["batch_width"] = 4
            evidence.removeValue(forKey: "steps")
            let allLanes = lanes
            try #require(heads.count == 4 && allLanes.count == 40)
            var cases: [[String: Any]] = []
            for start in stride(from: 0, to: allLanes.count, by: 4) {
                lanes = Array(allLanes[start..<(start + 4)])
                steps = lanes[0].frames.count - 1
                try #require(steps > 0 && lanes.allSatisfy { $0.frames.count == steps + 1 })
                let canonical = try run("canonical")
                gpuBatchSize = 4
                let batched = try run("batched-gpu")
                cases.append(["lane_ids": lanes.map { $0.id }, "steps": steps,
                    "exact": batched == canonical, "canonical": canonical, "batched": batched])
                evidence["cases"] = cases
                evidence["phase"] = "full-gpu-fidelity"
                try save()
            }
            evidence["phase"] = "complete"
            try save()
            return
        }
        let warmStart = Date()
        _ = try run("canonical")
        evidence["gpu_warmup_seconds"] = Date().timeIntervalSince(warmStart)
        let canonicalStart = Date()
        let canonical = try run("canonical")
        evidence["canonical_seconds"] = Date().timeIntervalSince(canonicalStart)
        let splitStart = Date()
        let split = try run("split-gpu")
        evidence["split_gpu_seconds"] = Date().timeIntervalSince(splitStart)
        evidence["split_gpu_exact"] = split == canonical
        try save()
        try #require(split == canonical)
        if gpuProbe {
            evidence["phase"] = "gpu-batches"
            evidence["lane_ids"] = lanes.map { $0.id }
            evidence["canonical_predictions"] = canonical
            for size in [4, 8, 12] {
                gpuBatchSize = size
                _ = try run("batched-gpu")
            }
            let modes = ["canonical", "split-gpu", "gpu-4", "gpu-8", "gpu-12"]
            var trials: [[String: Any]] = []
            for repeatIndex in 0..<3 {
                for offset in modes.indices {
                    let mode = modes[(offset + repeatIndex) % modes.count]
                    let batched = mode.hasPrefix("gpu-")
                    if batched { gpuBatchSize = try #require(Int(mode.dropFirst(4))) }
                    gpuMLPBatches = 0
                    let start = Date()
                    let predictions = try run(batched ? "batched-gpu" : mode)
                    let seconds = Date().timeIntervalSince(start)
                    var firstMatches = 0
                    var secondMatches = 0
                    for lane in lanes.indices {
                        for step in 0..<steps {
                            firstMatches += predictions[lane][step][0] == canonical[lane][step][0] ? 1 : 0
                            secondMatches += predictions[lane][step][1] == canonical[lane][step][1] ? 1 : 0
                        }
                    }
                    trials.append(["mode": mode, "repeat": repeatIndex,
                        "seconds": seconds, "exact": predictions == canonical,
                        "first_matches": firstMatches, "second_matches": secondMatches,
                        "batched_mlp_calls": gpuMLPBatches, "predictions": predictions])
                    evidence["gpu_trials"] = trials
                    try save()
                }
            }
            evidence["phase"] = "complete"
            try save()
            return
        }
        func effective(_ projection: Linear) -> MLXArray {
            if let packed = projection as? QuantizedLinear {
                return MLX.dequantized(packed.weight, scales: packed.scales,
                    biases: packed.biases, groupSize: packed.groupSize,
                    bits: packed.bits, mode: packed.mode)
            }
            let weight = projection.weight
            guard let bits = qwen35HeadProposalQuantizationBits else { return weight }
            let (w, s, b) = MLX.quantized(weight, groupSize: 64, bits: bits, mode: .affine)
            return MLX.dequantized(w, scales: s, biases: b, groupSize: 64, bits: bits, mode: .affine)
        }
        if capture {
            var arrays = ["input": concatenated(capturedInputs, axis: 0),
                "reference": concatenated(capturedOutputs, axis: 0),
                "gate": effective(mlp.gateProj), "up": effective(mlp.upProj),
                "down": effective(mlp.downProj)]
            var metadata: [String: Any] = ["lane_ids": lanes.map { $0.id }, "steps": steps,
                "rows": capturedInputs.count, "order": "step, draft position, lane",
                "canonical_and_split_exact": split == canonical]
            for (name, projection) in [("gate", mlp.gateProj), ("up", mlp.upProj), ("down", mlp.downProj)] {
                let packed = try #require(projection as? QuantizedLinear)
                try #require(packed.mode == .affine)
                arrays["\(name)_packed"] = packed.weight
                arrays["\(name)_scales"] = packed.scales
                arrays["\(name)_biases"] = try #require(packed.biases)
                metadata[name] = ["bits": packed.bits, "group_size": packed.groupSize]
            }
            try MLX.save(arrays: arrays, url: root.appendingPathComponent("ane-batch/mlp-capture.safetensors"))
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("ane-batch/mlp-capture.json"), options: .atomic)
            return
        }
        if aneDiagnostics {
            var trials: [[String: Any]] = []
            func record(_ name: String, _ predictions: [[[Int]]]) throws {
                trials.append(["mode": name, "exact": predictions == canonical,
                    "predictions": predictions])
                evidence["trials"] = trials
                try save()
            }
            evidence["lane_ids"] = lanes.map { $0.id }
            evidence["canonical_predictions"] = canonical
            for mode in ["dense-bf16", "dense-fp16"] {
                let dtype: DType = mode == "dense-bf16" ? .bfloat16 : .float16
                denseWeights = [concatenated([effective(mlp.gateProj), effective(mlp.upProj)], axis: 0).asType(dtype),
                    effective(mlp.downProj).asType(dtype)]
                eval(denseWeights)
                try record(mode, run(mode))
            }
            denseWeights = []
            for activation in [ANEActivation.expDiv, .sigmoidMul, .tanhForm] {
                program = nil
                do {
                    program = try MLXLLM.ANEFusedMLP(hidden: 5120, innerFraction: mlp.gateProj.weight.dim(0),
                        sequenceLength: width, gate: effective(mlp.gateProj), up: effective(mlp.upProj),
                        down: effective(mlp.downProj), activation: activation, weightForm: .fp16)
                    try record("ane-\(activation.rawValue)", run("ane"))
                } catch {
                    trials.append(["mode": "ane-\(activation.rawValue)", "error": String(describing: error)])
                    evidence["trials"] = trials
                    try save()
                }
            }
            evidence["phase"] = "complete"
            try save()
            return
        }
        var matchedPredictions: [[[Int]]]? = nil
        if matchedReplay {
            matchedWeights = [effective(mlp.gateProj).asType(.float16), effective(mlp.upProj).asType(.float16),
                effective(mlp.downProj).asType(.float16)]
            matchedPredictions = try run("matched-gpu")
            evidence["matched_gpu_predictions"] = matchedPredictions
            evidence["lane_ids"] = lanes.map { $0.id }
            evidence["canonical_predictions"] = canonical
        }
        evidence["phase"] = "ane-compile"
        try save()
        let compileStart = Date()
        do {
            program = try MLXLLM.ANEFusedMLP(hidden: 5120, innerFraction: mlp.gateProj.weight.dim(0),
                sequenceLength: width, gate: effective(mlp.gateProj), up: effective(mlp.upProj),
                down: effective(mlp.downProj), activation: .expDiv, weightForm: .fp16)
            evidence["compile_seconds"] = Date().timeIntervalSince(compileStart)
            let start = Date()
            let ane = try run("ane")
            evidence["ane_seconds"] = Date().timeIntervalSince(start)
            evidence["ane_dispatches"] = dispatches
            evidence["ane_exact"] = ane == canonical
            if let matchedPredictions {
                evidence["ane_matches_matched_gpu"] = ane == matchedPredictions
            }
            evidence["predictions"] = zip(lanes, ane).enumerated().map { i, item in
                ["lane": item.0.id, "gpu": canonical[i], "ane": item.1] as [String: Any]
            }
            evidence["phase"] = "complete"
            try save()
        } catch {
            evidence["phase"] = "ane-failed"
            evidence["error"] = String(describing: error)
            evidence["ane_dispatches"] = dispatches
            try save()
            throw error
        }
    }
}
