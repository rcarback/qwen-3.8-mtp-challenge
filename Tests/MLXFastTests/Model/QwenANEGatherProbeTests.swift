import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEGatherProbeTests {
    @Test func runtimeRowLookup() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_GATHER_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let oneHot = env["MLXFAST_ANE_GATHER_ONE_HOT"] == "1"
        let checkpoint = env["MLXFAST_ANE_GATHER_CHECKPOINT"] == "1"
        let large = checkpoint || env["MLXFAST_ANE_GATHER_LARGE"] == "1"
        let tileStart = Int(env["MLXFAST_ANE_GATHER_TILE_START"] ?? "0")
        try #require(tileStart != nil)
        let start = try #require(tileStart)
        try #require(start >= 0 && start <= 248320 - 2048)
        try #require(!large || oneHot)
        let rowCount = large ? 2048 : 64
        let embeddingFeatures = large ? 5120 : 256
        let features = embeddingFeatures + (checkpoint ? 11 : 0)
        let table: [Float16]
        if checkpoint {
            let weightsRoot = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_TARGET_WEIGHTS"]))
            let data = try Data(contentsOf: weightsRoot.appendingPathComponent("model.safetensors.index.json"))
            let index = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let mapping = try #require(index["weight_map"] as? [String: String])
            let prefix = "language_model.model.embed_tokens"
            let shard = try #require(mapping[prefix + ".weight"])
            try #require(mapping[prefix + ".scales"] == shard && mapping[prefix + ".biases"] == shard)
            let arrays = try MLX.loadArrays(url: weightsRoot.appendingPathComponent(shard))
            let packed = try #require(arrays[prefix + ".weight"])
            let scales = try #require(arrays[prefix + ".scales"])
            let biases = try #require(arrays[prefix + ".biases"])
            try #require(packed.shape == [248320, 640] && scales.shape == [248320, 80] && biases.shape == scales.shape)
            let end = start + rowCount
            let weight = aneProbeAffineWeight(packed[start..<end], scales: scales[start..<end],
                biases: biases[start..<end], groupSize: 64, bits: 4)
            let embedding = weight.asArray(Float16.self)
            try #require(embedding.count == rowCount * embeddingFeatures && embedding.allSatisfy { $0.isFinite })
            var tagged: [Float16] = []
            tagged.reserveCapacity(rowCount * features)
            for row in 0..<rowCount {
                tagged.append(contentsOf: embedding[(row * embeddingFeatures)..<((row + 1) * embeddingFeatures)])
                for bit in 0..<11 { tagged.append(Float16((row >> bit) & 1)) }
            }
            table = tagged
        } else {
            table = (0..<(rowCount * features)).map { index in
            let row: Int = index / features
            let integer: Int = (index * 17 + row) % 509 - 254
            return Float16(Float(integer) / 16.0)
            }
        }
        let weights: [Float16] = oneHot ? (0..<(rowCount * features)).map { index in
            let row: Int = index % rowCount
            let channel: Int = index / rowCount
            return table[row * features + channel]
        } : table
        let blob = buildConvWeightBlob(weights.withUnsafeBytes { Data($0) })
        let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        var text = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(digest)"}})]
        {
          func main<ios18>(tensor<fp16, [1, 1, 1, 32]> x) {
            tensor<fp16, [64, 256]> table = const()[name=string("table"), val=tensor<fp16, [64, 256]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
            tensor<int32, [1]> indexShape = const()[name=string("indexShape"), val=tensor<int32, [1]>([32])];
            tensor<int32, [4]> outputShape = const()[name=string("outputShape"), val=tensor<int32, [4]>([32, 1, 256, 1])];
            tensor<int32, [4]> perm = const()[name=string("perm"), val=tensor<int32, [4]>([1, 2, 3, 0])];
            string indexType = const()[name=string("indexType"), val=string("int32")];
            int32 axis = const()[name=string("axis"), val=int32(0)];
            int32 batchDims = const()[name=string("batchDims"), val=int32(0)];
            bool validate = const()[name=string("validate"), val=bool(false)];
            tensor<fp16, [32]> flat = reshape(x=x, shape=indexShape)[name=string("flat")];
            tensor<int32, [32]> ids = cast(x=flat, dtype=indexType)[name=string("ids")];
            tensor<fp16, [32, 256]> selected = gather(x=table, indices=ids, axis=axis, batch_dims=batchDims, validate_indices=validate)[name=string("selected")];
            tensor<fp16, [32, 1, 256, 1]> rows = reshape(x=selected, shape=outputShape)[name=string("rows")];
            tensor<fp16, [1, 256, 1, 32]> y = transpose(x=rows, perm=perm)[name=string("y")];
          } -> (y);
        }
        """
        if oneHot {
            text = buildConvMILText(inputDim: rowCount, outputDim: features, sequenceLength: 32, programTag: digest)
            let signature = "func main<ios18>(tensor<fp16, [1, \(rowCount), 1, 32]> x) {"
            try #require(text.contains(signature))
            let indices = (0..<rowCount).map(String.init).joined(separator: ", ")
            let preparation = """
            func main<ios18>(tensor<fp16, [1, 1, 1, 32]> x) {
              tensor<fp16, [\(rowCount)]> indices = const()[name=string("indices"), val=tensor<fp16, [\(rowCount)]>([\(indices)])];
              tensor<int32, [4]> indexShape = const()[name=string("indexShape"), val=tensor<int32, [4]>([1, \(rowCount), 1, 1])];
              tensor<fp16, [1, \(rowCount), 1, 1]> rows = reshape(x=indices, shape=indexShape)[name=string("rows")];
              fp16 one = const()[name=string("one"), val=fp16(1.0)];
              fp16 zero = const()[name=string("zero"), val=fp16(0.0)];
              tensor<bool, [1, \(rowCount), 1, 32]> match = equal(x=x, y=rows)[name=string("match")];
              tensor<fp16, [1, \(rowCount), 1, 32]> selector = select(cond=match, a=one, b=zero)[name=string("selector")];
            """
            text = text.replacingOccurrences(of: signature, with: preparation)
                .replacingOccurrences(of: "weight=w, x=x)", with: "weight=w, x=selector)")
        }
        let stem = checkpoint ? "gather-checkpoint-tagged-\(start)" : large ? "gather-one-hot-large" : (oneHot ? "gather-one-hot" : "gather-probe")
        try text.write(to: root.appendingPathComponent("\(stem).mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "table_rows": rowCount, "features": features, "lanes": 32,
            "weight_blob_sha256": digest, "checkpoint": checkpoint, "tile_start": start,
            "embedding_features": embeddingFeatures, "exact_marker_channels": checkpoint ? 11 : 0,
            "embedding_relative_rms_limit": checkpoint ? 0.002 : 0,
            "scope": "ANE tile-local lookup; offline FP16 table preparation; host tile routing not integrated"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("\(stem).json"), options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob)
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            var patterns = [(0..<32).map { ($0 * 67) % rowCount }, (0..<32).map { $0 % 2 == 0 ? 0 : rowCount - 1 }, Array(repeating: 17, count: 32)]
            if checkpoint && start <= 248044 && start + rowCount > 248069 {
                patterns.append((0..<32).map { 248044 + $0 % 26 - start })
            }
            var trials: [[String: Any]] = []
            for indices in patterns {
                let expected = indices.flatMap { Array(table[($0 * features)..<(($0 + 1) * features)]) }
                let input = MLXArray(indices.map { Float16($0) }, [32, 1])
                result["phase"] = "dispatch"; try save()
                let began = Date()
                let output = try ANEDirectDispatch.runConv(model: model, x: input, inputDim: 1,
                    outputDim: features, sequenceLength: 32)
                let actual = output.asArray(Float16.self)
                let elapsed = Date().timeIntervalSince(began)
                try #require(actual.count == expected.count && actual.allSatisfy { $0.isFinite })
                let mismatches = zip(actual, expected).filter { $0 != $1 }.count
                let maxError = zip(actual, expected).map { abs(Float($0) - Float($1)) }.max() ?? 0
                let flushed = zip(actual, expected).filter { $0 == 0 && $1 != 0 && abs($1) < Float16.leastNormalMagnitude }.count
                var markerErrors = 0
                var squared = 0.0, energy = 0.0
                for row in 0..<32 {
                    for channel in 0..<features {
                        let offset = row * features + channel
                        if channel >= embeddingFeatures {
                            if actual[offset] != expected[offset] { markerErrors += 1 }
                        } else {
                            let value = Double(expected[offset])
                            let error = Double(actual[offset]) - value
                            squared += error * error; energy += value * value
                        }
                    }
                }
                let relativeRMS = sqrt(squared / max(energy, Double.leastNormalMagnitude))
                trials.append(["indices": indices, "mismatched_values": mismatches,
                    "marker_errors": markerErrors, "embedding_relative_rms_error": relativeRMS,
                    "max_abs_error": maxError, "flushed_subnormals": flushed,
                    "dispatch_and_read_seconds": elapsed])
                if mismatches > 0 {
                    try MLX.save(arrays: ["actual": MLXArray(actual, [32, features]),
                        "expected": MLXArray(expected, [32, features])],
                        url: root.appendingPathComponent("\(stem)-mismatch.safetensors"))
                }
                result["trials"] = trials; try save()
                if checkpoint {
                    try #require(markerErrors == 0)
                    try #require(relativeRMS <= 0.002)
                } else {
                    try #require(mismatches == 0)
                }
            }
            result["phase"] = "complete"; try save()
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
