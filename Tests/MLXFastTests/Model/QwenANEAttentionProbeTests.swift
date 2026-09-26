import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEAttentionProbeTests {
    @Test func maskedAttention() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_ATTENTION_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
        let checkpoint = env["MLXFAST_ANE_ATTENTION_CHECKPOINT"] == "1"
        let prepared = checkpoint ? try MLX.loadArrays(url: root.appendingPathComponent("ane-batch/checkpoint-prepared-qk.safetensors")) : [:]
        let projected = checkpoint ? try MLX.loadArrays(url: root.appendingPathComponent("ane-batch/checkpoint-qkv.safetensors")) : [:]
        let batch = 32, dimension = 256, history = checkpoint ? 32 : 128
        let queryHeads = 24, kvHeads = 4
        let groups = queryHeads / kvHeads
        let queryChannels = queryHeads * dimension
        let kvChannels = kvHeads * history * dimension
        let channels = queryChannels + 2 * kvChannels + history
        var lines: [String] = []
        func ints(_ name: String, _ values: [Int]) {
            lines.append("tensor<int32, [\(values.count)]> \(name) = const()[name=string(\"\(name)\"), val=tensor<int32, [\(values.count)]>([\(values.map(String.init).joined(separator: ", "))])];")
        }
        ints("perm", [3, 0, 1, 2])
        var offset = 0
        for (name, count, shape) in [("q", queryChannels, [batch, kvHeads, groups, dimension]),
                                     ("k", kvChannels, [batch, kvHeads, history, dimension]),
                                     ("v", kvChannels, [batch, kvHeads, history, dimension]),
                                     ("mask", history, [batch, 1, 1, history])] {
            ints("\(name)begin", [0, offset, 0, 0])
            ints("\(name)size", [1, count, 1, batch])
            ints("\(name)shape", shape)
            let dims = shape.map(String.init).joined(separator: ", ")
            lines.append("tensor<fp16, [1, \(count), 1, \(batch)]> \(name)raw = slice_by_size(x=x, begin=\(name)begin, size=\(name)size)[name=string(\"\(name)raw\")];")
            lines.append("tensor<fp16, [\(batch), 1, \(count), 1]> \(name)t = transpose(x=\(name)raw, perm=perm)[name=string(\"\(name)t\")];")
            lines.append("tensor<fp16, [\(dims)]> \(name) = reshape(x=\(name)t, shape=\(name)shape)[name=string(\"\(name)\")];")
            offset += count
        }
        ints("outputPerm", [1, 3, 2, 0])
        lines.append("bool yes = const()[name=string(\"yes\"), val=bool(true)];")
        lines.append("bool no = const()[name=string(\"no\"), val=bool(false)];")
        lines.append("fp16 scale = const()[name=string(\"scale\"), val=fp16(0.0625)];")
        lines.append("int32 axis = const()[name=string(\"axis\"), val=int32(-1)];")
        let scores = "tensor<fp16, [\(batch), \(kvHeads), \(groups), \(history)]>"
        lines.append("\(scores) dot = matmul(x=q, y=k, transpose_x=no, transpose_y=yes)[name=string(\"dot\")];")
        lines.append("\(scores) scaled = mul(x=dot, y=scale)[name=string(\"scaled\")];")
        lines.append("\(scores) masked = add(x=scaled, y=mask)[name=string(\"masked\")];")
        lines.append("\(scores) probabilities = softmax(x=masked, axis=axis)[name=string(\"probabilities\")];")
        lines.append("tensor<fp16, [\(batch), \(kvHeads), \(groups), \(dimension)]> attention = matmul(x=probabilities, y=v, transpose_x=no, transpose_y=no)[name=string(\"attention\")];")
        ints("flatShape", [batch, 1, 1, queryChannels])
        lines.append("tensor<fp16, [\(batch), 1, 1, \(queryChannels)]> flat = reshape(x=attention, shape=flatShape)[name=string(\"flat\")];")
        lines.append("tensor<fp16, [1, \(queryChannels), 1, \(batch)]> y = transpose(x=flat, perm=outputPerm)[name=string(\"y\")];")
        let text = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, \(channels), 1, \(batch)]> x) {
            \(lines.joined(separator: "\n"))
          } -> (y);
        }
        """
        let output = root.appendingPathComponent("ane-batch/attention-probe\(checkpoint ? "-checkpoint" : "").json")
        try text.write(to: root.appendingPathComponent("ane-batch/attention-probe\(checkpoint ? "-checkpoint" : "").mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "lanes": batch, "head_dimension": dimension, "history": history, "query_heads": queryHeads, "kv_heads": kvHeads, "checkpoint": checkpoint, "history_scope": checkpoint ? "Synthetic ordered projection rows, not a corpus replay" : "Synthetic masked attention"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: output, options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: Data())
            try model.compile()
            result["phase"] = "load"; try save()
            try model.load()
            defer { model.unload() }
            func preparedArray(_ name: String) throws -> [Float] {
                guard checkpoint else { return [] }
                let array = try #require(prepared[name])
                try #require(array.shape == [batch, 28, dimension])
                return array.asArray(Float.self)
            }
            func projectedArray(_ name: String) throws -> [Float] {
                guard checkpoint else { return [] }
                let array = try #require(projected[name])
                try #require(array.shape == [batch, 14336])
                return array.asArray(Float.self)
            }
            let aneQK = try preparedArray("ane_qk"), referenceQK = try preparedArray("reference_qk")
            let aneProjection = try projectedArray("ane_qkv"), referenceProjection = try projectedArray("gpu_qkv")
            func cache(_ source: [Float], keys: Bool) -> [Float] {
                var values: [Float] = []
                for head in 0..<kvHeads {
                    for row in 0..<history {
                        let start = keys ? row * 28 * dimension + (24 + head) * dimension : row * 14336 + 13312 + head * dimension
                        values += source[start..<(start + dimension)]
                    }
                }
                return values
            }
            let checkpointK = checkpoint ? cache(aneQK, keys: true) : []
            let checkpointV = checkpoint ? cache(aneProjection, keys: false) : []
            let referenceK = checkpoint ? cache(referenceQK, keys: true) : []
            let referenceV = checkpoint ? cache(referenceProjection, keys: false) : []
            var inputs: [Float16] = []
            var expected: [Float] = []
            for lane in 0..<batch {
                let q: [Float16] = checkpoint ? aneQK[(lane * 28 * dimension)..<(lane * 28 * dimension + queryChannels)].map(Float16.init) : (0..<queryChannels).map { Float16(Float(($0 + lane) % 17 - 8) / 16) }
                let k: [Float16] = checkpoint ? checkpointK.map(Float16.init) : (0..<kvChannels).map { index in
                    let integer = (index * 3 + lane) % 23 - 11
                    return Float16(Float(integer) / 16.0)
                }
                let v: [Float16] = checkpoint ? checkpointV.map(Float16.init) : (0..<kvChannels).map { index in
                    let integer = (index * 7 + lane) % 29 - 14
                    return Float16(Float(integer) / 16.0)
                }
                let oracleQ: [Float] = checkpoint ? Array(referenceQK[(lane * 28 * dimension)..<(lane * 28 * dimension + queryChannels)]) : q.map(Float.init)
                let oracleK: [Float] = checkpoint ? referenceK : k.map(Float.init)
                let oracleV: [Float] = checkpoint ? referenceV : v.map(Float.init)
                let valid = 1 + lane * (history - 1) / (batch - 1)
                inputs += q + k + v + (0..<history).map { Float16($0 < valid ? 0 : -10000) }
                for head in 0..<queryHeads {
                    let kvOffset = (head / groups) * history * dimension
                    var logits: [Float] = []
                    for t in 0..<valid {
                        var dot: Float = 0
                        for d in 0..<dimension {
                            dot += oracleQ[head * dimension + d] * oracleK[kvOffset + t * dimension + d]
                        }
                        logits.append(dot / 16.0)
                    }
                    let maximum = try #require(logits.max())
                    let expValues = logits.map { Foundation.exp($0 - maximum) }
                    let total = expValues.reduce(0, +)
                    for d in 0..<dimension {
                        var output: Float = 0
                        for t in 0..<valid {
                            output += expValues[t] / total * oracleV[kvOffset + t * dimension + d]
                        }
                        expected.append(output)
                    }
                }
            }
            result["phase"] = "dispatch"; try save()
            let started = Date()
            let value = try ANEDirectDispatch.runConv(model: model, x: MLXArray(inputs, [batch, channels]), inputDim: channels, outputDim: queryChannels, sequenceLength: batch)
            let actual = value.asArray(Float.self)
            try #require(actual.count == expected.count)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite > 0 { try save(); try #require(nonfinite == 0) }
            let errors = zip(actual, expected).map { abs($0 - $1) }
            let maximum = try #require(errors.max())
            result["max_abs_error"] = maximum
            result["mean_abs_error"] = errors.reduce(0, +) / Float(errors.count)
            result["dispatch_and_read_seconds"] = Date().timeIntervalSince(started)
            result["phase"] = "complete"; try save()
            if checkpoint {
                try MLX.save(arrays: ["ane_attention": MLXArray(actual, [batch, queryChannels]),
                                     "reference_attention": MLXArray(expected, [batch, queryChannels])],
                             url: root.appendingPathComponent("ane-batch/checkpoint-attention.safetensors"))
            }
            try #require(maximum < 0.01)
        } catch {
            result["error"] = String(describing: error)
            try save()
            throw error
        }
    }
}
