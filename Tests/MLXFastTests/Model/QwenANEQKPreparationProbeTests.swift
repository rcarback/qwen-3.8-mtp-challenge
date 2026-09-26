import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANEQKPreparationProbeTests {
    @Test func normalizationAndRoPE() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_QK_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
        let checkpoint = env["MLXFAST_ANE_QK_CHECKPOINT"] == "1"
        let capture = checkpoint ? try MLX.loadArrays(url: root.appendingPathComponent("ane-batch/checkpoint-qkv.safetensors")) : [:]
        let exhaustive = env["MLXFAST_ANE_QK_ALL_POSITIONS"] == "1"
        let binaryAngles = env["MLXFAST_ANE_QK_BINARY_ANGLES"] == "1"
        let binaryBits = Int(env["MLXFAST_ANE_QK_BINARY_BITS"] ?? "10") ?? 10
        try #require((1...10).contains(binaryBits))
        let fp32Angles = env["MLXFAST_ANE_QK_FP32_ANGLES"] == "1"
        let normOnly = env["MLXFAST_ANE_QK_NORM_ONLY"] == "1"
        let normStage = env["MLXFAST_ANE_QK_OUTPUT"] ?? "weighted"
        try #require(["vectors", "squared", "mean", "variance", "inverse", "normalized", "weighted"].contains(normStage))
        let batch = 32, heads = 28, dimension = 256, rotary = 64
        let positions: [Int]
        if let savedPositions = capture["positions"] {
            try #require(!exhaustive && savedPositions.shape == [batch] && savedPositions.dtype == .int32)
            positions = savedPositions.asArray(Int32.self).map(Int.init)
        } else {
            positions = (0..<batch).map { $0 * 31 }
        }
        if !exhaustive {
            try #require(positions.allSatisfy { $0 >= 0 && $0 <= 2048 })
            if binaryAngles { try #require(positions.allSatisfy { $0 < (1 << binaryBits) }) }
        }
        let features = heads * dimension
        let outputFeatures = normOnly && ["mean", "variance", "inverse"].contains(normStage) ? heads : features
        let channels = features + 1
        var lines: [String] = []
        func ints(_ name: String, _ values: [Int]) {
            lines.append("tensor<int32, [\(values.count)]> \(name) = const()[name=string(\"\(name)\"), val=tensor<int32, [\(values.count)]>([\(values.map(String.init).joined(separator: ", "))])];")
        }
        func slice(_ name: String, _ input: String, _ begin: [Int], _ shape: [Int]) {
            ints("\(name)_begin", begin); ints("\(name)_size", shape)
            lines.append("tensor<fp16, [\(shape.map(String.init).joined(separator: ", "))]> \(name) = slice_by_size(x=\(input), begin=\(name)_begin, size=\(name)_size)[name=string(\"\(name)\")];")
        }
        slice("raw", "x", [0, 0, 0, 0], [1, features, 1, batch])
        slice("position", "x", [0, features, 0, 0], [1, 1, 1, batch])
        ints("perm", [3, 0, 1, 2]); ints("shape", [batch, heads, 1, dimension])
        lines.append("tensor<fp16, [\(batch), 1, \(features), 1]> transposed = transpose(x=raw, perm=perm)[name=string(\"transposed\")];")
        lines.append("tensor<fp16, [\(batch), \(heads), 1, \(dimension)]> vectors = reshape(x=transposed, shape=shape)[name=string(\"vectors\")];")
        lines.append("tensor<fp16, [\(batch), 1, 1, 1]> positions = transpose(x=position, perm=perm)[name=string(\"positions\")];")
        ints("axes", [-1])
        lines.append("bool keep = const()[name=string(\"keep\"), val=bool(true)];")
        lines.append("fp16 eps = const()[name=string(\"eps\"), val=fp16(0.000001)];")
        lines.append("fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];")
        let full = "tensor<fp16, [\(batch), \(heads), 1, \(dimension)]>"
        let reduced = "tensor<fp16, [\(batch), \(heads), 1, 1]>"
        lines.append("\(full) squared = mul(x=vectors, y=vectors)[name=string(\"squared\")];")
        lines.append("\(reduced) mean = reduce_mean(x=squared, axes=axes, keep_dims=keep)[name=string(\"mean\")];")
        lines.append("\(reduced) variance = add(x=mean, y=eps)[name=string(\"variance\")];")
        lines.append("\(reduced) inverse = rsqrt(x=variance, epsilon=zero)[name=string(\"inverse\")];")
        lines.append("\(full) normalized = mul(x=vectors, y=inverse)[name=string(\"normalized\")];")
        let weights: [Float16]
        if checkpoint {
            let q = try #require(capture["q_norm"]).asArray(Float16.self)
            let k = try #require(capture["k_norm"]).asArray(Float16.self)
            try #require(q.count == dimension && k.count == dimension)
            weights = (0..<heads).flatMap { $0 < 24 ? q : k }
        } else {
            weights = (0..<features).map { Float16(0.75 + Float(($0 % dimension) % 11) / 32) }
        }
        lines.append("tensor<fp16, [\(features)]> weightFlat = const()[name=string(\"weightFlat\"), val=tensor<fp16, [\(features)]>([\(weights.map { String(Float($0)) }.joined(separator: ", "))])];")
        ints("weightShape", [1, heads, 1, dimension])
        lines.append("tensor<fp16, [1, \(heads), 1, \(dimension)]> weight = reshape(x=weightFlat, shape=weightShape)[name=string(\"weight\")];")
        lines.append("\(full) weighted = mul(x=normalized, y=weight)[name=string(\"weighted\")];")
        slice("left", "weighted", [0, 0, 0, 0], [batch, heads, 1, rotary / 2])
        slice("right", "weighted", [0, 0, 0, rotary / 2], [batch, heads, 1, rotary / 2])
        slice("tail", "weighted", [0, 0, 0, rotary], [batch, heads, 1, dimension - rotary])
        try #require(!(binaryAngles && fp32Angles))
        try #require(!exhaustive || (binaryAngles && binaryBits == 10 && !normOnly))
        let angleType = fp32Angles ? "fp32" : "fp16"
        let rotated = "tensor<fp16, [\(batch), \(heads), 1, \(rotary / 2)]>"
        if binaryAngles {
            let angle = "tensor<fp16, [\(batch), 1, 1, \(rotary / 2)]>"
            let scalarLanes = "tensor<fp16, [\(batch), 1, 1, 1]>"
            lines.append("fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];")
            lines.append("fp16 two = const()[name=string(\"two\"), val=fp16(2.0)];")
            ints("rotationShape", [1, 1, 1, rotary / 2])
            var previousCosine = "one", previousSine = "zero"
            var remainder = "positions"
            for bit in (0..<10).reversed() {
                let tag = "bit\(bit)"
                lines.append("fp16 \(tag)threshold = const()[name=string(\"\(tag)threshold\"), val=fp16(\(Double(1 << bit)))];")
                lines.append("\(scalarLanes) \(tag)difference = sub(x=\(remainder), y=\(tag)threshold)[name=string(\"\(tag)difference\")];")
                lines.append("\(scalarLanes) \(tag)shifted = add(x=\(tag)difference, y=one)[name=string(\"\(tag)shifted\")];")
                lines.append("\(scalarLanes) \(tag)value = clip(x=\(tag)shifted, alpha=zero, beta=one)[name=string(\"\(tag)value\")];")
                lines.append("\(scalarLanes) \(tag)used = mul(x=\(tag)value, y=\(tag)threshold)[name=string(\"\(tag)used\")];")
                lines.append("\(scalarLanes) \(tag)remainder = sub(x=\(remainder), y=\(tag)used)[name=string(\"\(tag)remainder\")];")
                remainder = tag + "remainder"
                if bit >= binaryBits { continue }
                lines.append("\(scalarLanes) \(tag)inactive = sub(x=one, y=\(tag)value)[name=string(\"\(tag)inactive\")];")
                let trigonometry: [(String, (Double) -> Double)] = [("c", Foundation.cos), ("s", Foundation.sin)]
                for (name, operation) in trigonometry {
                    let values = (0..<(rotary / 2)).map { i in
                        let frequency = pow(10_000_000.0, -2.0 * Double(i) / Double(rotary))
                        return Float16(operation(Double(1 << bit) * frequency))
                    }
                    lines.append("tensor<fp16, [\(rotary / 2)]> \(tag)\(name)Flat = const()[name=string(\"\(tag)\(name)Flat\"), val=tensor<fp16, [\(rotary / 2)]>([\(values.map { String(Float($0)) }.joined(separator: ", "))])];")
                    lines.append("tensor<fp16, [1, 1, 1, \(rotary / 2)]> \(tag)\(name)Const = reshape(x=\(tag)\(name)Flat, shape=rotationShape)[name=string(\"\(tag)\(name)Const\")];")
                    lines.append("\(angle) \(tag)\(name)Selected = mul(x=\(tag)value, y=\(tag)\(name)Const)[name=string(\"\(tag)\(name)Selected\")];")
                }
                lines.append("\(angle) \(tag)c = add(x=\(tag)cSelected, y=\(tag)inactive)[name=string(\"\(tag)c\")];")
                for (name, x, y) in [("cc", previousCosine, tag + "c"), ("ss", previousSine, tag + "sSelected"), ("sc", previousSine, tag + "c"), ("cs", previousCosine, tag + "sSelected")] {
                    lines.append("\(angle) \(tag)\(name) = mul(x=\(x), y=\(y))[name=string(\"\(tag)\(name)\")];")
                }
                lines.append("\(angle) \(tag)cosine = sub(x=\(tag)cc, y=\(tag)ss)[name=string(\"\(tag)cosine\")];")
                lines.append("\(angle) \(tag)sine = add(x=\(tag)sc, y=\(tag)cs)[name=string(\"\(tag)sine\")];")
                previousCosine = tag + "cosine"; previousSine = tag + "sine"
            }
            lines.append("\(angle) cosine = add(x=\(previousCosine), y=zero)[name=string(\"cosine\")];")
            lines.append("\(angle) sine = add(x=\(previousSine), y=zero)[name=string(\"sine\")];")
        } else {
        let frequencies = (0..<(rotary / 2)).map { index -> Float in
            let value = Float(pow(10_000_000.0, -2 * Double(index) / Double(rotary)))
            return fp32Angles ? value : Float(Float16(value))
        }
        lines.append("tensor<\(angleType), [\(rotary / 2)]> frequenciesFlat = const()[name=string(\"frequenciesFlat\"), val=tensor<\(angleType), [\(rotary / 2)]>([\(frequencies.map { String($0) }.joined(separator: ", "))])];")
        ints("frequencyShape", [1, 1, 1, rotary / 2])
        lines.append("tensor<\(angleType), [1, 1, 1, \(rotary / 2)]> frequencies = reshape(x=frequenciesFlat, shape=frequencyShape)[name=string(\"frequencies\")];")
        let angle = "tensor<\(angleType), [\(batch), 1, 1, \(rotary / 2)]>"
        if fp32Angles {
            lines.append("string dtype32 = const()[name=string(\"dtype32\"), val=string(\"fp32\")];")
            lines.append("string dtype16 = const()[name=string(\"dtype16\"), val=string(\"fp16\")];")
            lines.append("tensor<fp32, [\(batch), 1, 1, 1]> widePositions = cast(x=positions, dtype=dtype32)[name=string(\"widePositions\")];")
        }
        lines.append("\(angle) angles = mul(x=\(fp32Angles ? "widePositions" : "positions"), y=frequencies)[name=string(\"angles\")];")
        for (name, operation) in [("cosine", "cos"), ("sine", "sin")] {
            let output = fp32Angles ? name + "Wide" : name
            lines.append("\(angle) \(output) = \(operation)(x=angles)[name=string(\"\(output)\")];")
            if fp32Angles {
                lines.append("tensor<fp16, [\(batch), 1, 1, \(rotary / 2)]> \(name) = cast(x=\(output), dtype=dtype16)[name=string(\"\(name)\")];")
            }
        }
        }
        for (name, x, y) in [("lc", "left", "cosine"), ("rs", "right", "sine"), ("ls", "left", "sine"), ("rc", "right", "cosine")] {
            lines.append("\(rotated) \(name) = mul(x=\(x), y=\(y))[name=string(\"\(name)\")];")
        }
        lines.append("\(rotated) first = sub(x=lc, y=rs)[name=string(\"first\")];")
        lines.append("\(rotated) second = add(x=ls, y=rc)[name=string(\"second\")];")
        lines.append("int32 axis = const()[name=string(\"axis\"), val=int32(-1)];")
        lines.append("bool interleave = const()[name=string(\"interleave\"), val=bool(false)];")
        lines.append("\(full) result = concat(values=(first, second, tail), axis=axis, interleave=interleave)[name=string(\"result\")];")
        ints("flatShape", [batch, 1, outputFeatures, 1]); ints("back", [1, 2, 3, 0])
        lines.append("tensor<fp16, [\(batch), 1, \(outputFeatures), 1]> flat = reshape(x=result, shape=flatShape)[name=string(\"flat\")];")
        lines.append("tensor<fp16, [1, \(outputFeatures), 1, \(batch)]> y = transpose(x=flat, perm=back)[name=string(\"y\")];")
        if normOnly {
            let end = try #require(lines.firstIndex { $0.contains("> \(normStage) =") })
            lines = Array(lines[...end]) + lines.suffix(4).map { $0.replacingOccurrences(of: "x=result", with: "x=\(normStage)") }
        }
        let text = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, \(channels), 1, \(batch)]> x) {
            \(lines.joined(separator: "\n"))
          } -> (y);
        }
        """
        let output = root.appendingPathComponent("ane-batch/qk-preparation-\(checkpoint ? "checkpoint-" : "")\(normOnly ? normStage : (binaryAngles ? "binary-angles-\(binaryBits)\(exhaustive ? "-all-positions" : "")" : (fp32Angles ? "fp32-angles" : "full"))).json")
        try text.write(to: root.appendingPathComponent("ane-batch/qk-preparation-\(checkpoint ? "checkpoint-" : "")\(normOnly ? normStage : (binaryAngles ? "binary-angles-\(binaryBits)\(exhaustive ? "-all-positions" : "")" : (fp32Angles ? "fp32-angles" : "full"))).mil"), atomically: true, encoding: .utf8)
        var result: [String: Any] = ["phase": "compile", "lanes": batch, "heads": heads, "rotary_dimensions": rotary, "angle_dtype": angleType, "binary_angles": binaryAngles, "binary_bits": binaryBits, "checkpoint": checkpoint]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: output, options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: Data())
            try model.compile(); result["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            func projectedRows(_ name: String) throws -> [[Float16]] {
                guard checkpoint else { return [] }
                let tensor = try #require(capture[name])
                try #require(tensor.shape == [batch, 14336])
                let flat = tensor.asArray(Float16.self)
                return (0..<batch).map { lane in
                    let start = lane * 14336
                    var row: [Float16] = []
                    for head in 0..<24 {
                        row += flat[(start + head * 512)..<(start + head * 512 + dimension)]
                    }
                    row += flat[(start + 12288)..<(start + 13312)]
                    return row
                }
            }
            let projected = try projectedRows("ane_qkv")
            let referenceProjected = try projectedRows("gpu_qkv")
            var records: [[String: Any]] = []
            var peakError: Float = 0
            var errorSum: Double = 0
            var valueCount = 0
            for block in 0..<(exhaustive ? 32 : 1) {
            var inputs: [Float16] = [], expected: [Float] = []
            for lane in 0..<batch {
                let position = exhaustive ? block * batch + lane : positions[lane]
                let values = checkpoint ? projected[lane] : (0..<features).map { Float16(Float(($0 * 3 + lane) % 31 - 15) / 16) }
                let referenceValues = checkpoint ? referenceProjected[lane] : values
                inputs += values + [Float16(position)]
                for head in 0..<heads {
                    let row = Array(referenceValues[(head * dimension)..<((head + 1) * dimension)]).map(Float.init)
                    let inverse = 1 / sqrt(row.reduce(0) { $0 + $1 * $1 } / Float(dimension) + 0.000001)
                    if normOnly && ["mean", "variance", "inverse"].contains(normStage) {
                        let mean = row.reduce(0) { $0 + $1 * $1 } / Float(dimension)
                        expected.append(normStage == "mean" ? mean : (normStage == "variance" ? mean + 0.000001 : inverse))
                        continue
                    }
                    var normalized = row.enumerated().map { item in
                        if normOnly && normStage == "vectors" { return item.element }
                        if normOnly && normStage == "squared" { return item.element * item.element }
                        let weight: Float = normOnly && normStage == "normalized" ? 1 : Float(weights[head * dimension + item.offset])
                        return item.element * inverse * weight
                    }
                    for i in 0..<(normOnly ? 0 : rotary / 2) {
                        let referencePosition = binaryAngles ? position % (1 << binaryBits) : position
                        let angle = Double(referencePosition) * pow(10_000_000.0, -2 * Double(i) / Double(rotary))
                        let a = normalized[i], b = normalized[i + rotary / 2]
                        normalized[i] = a * Float(cos(angle)) - b * Float(sin(angle))
                        normalized[i + rotary / 2] = a * Float(sin(angle)) + b * Float(cos(angle))
                    }
                    expected += normalized
                }
            }
            result["phase"] = "dispatch"; try save()
            let started = Date()
            let value = try ANEDirectDispatch.runConv(model: model, x: MLXArray(inputs, [batch, channels]), inputDim: channels, outputDim: outputFeatures, sequenceLength: batch)
            let actual = value.asArray(Float.self)
            try #require(actual.count == expected.count)
            let nonfinite = actual.filter { !$0.isFinite }.count
            result["nonfinite_values"] = nonfinite
            if nonfinite > 0 {
                result["phase"] = "nonfinite-output"
                try save()
                try #require(nonfinite == 0)
            }
            let errors = zip(actual, expected).map { abs($0 - $1) }
            let maximum = try #require(errors.max())
            let sum = errors.reduce(Double(0)) { $0 + Double($1) }
            peakError = max(peakError, maximum)
            errorSum += sum
            valueCount += errors.count
            records.append(["block": block, "first_position": exhaustive ? block * batch : positions[0],
                            "last_position": exhaustive ? (block + 1) * batch - 1 : positions[batch - 1],
                            "max_abs_error": maximum, "mean_abs_error": sum / Double(errors.count),
                            "dispatch_and_read_seconds": Date().timeIntervalSince(started)])
            result["blocks"] = records
            result["max_abs_error"] = peakError
            result["mean_abs_error"] = errorSum / Double(valueCount)
            result["positions_checked"] = records.count * batch
            try save()
            if checkpoint && !exhaustive && !normOnly {
                try MLX.save(arrays: ["ane_qk": MLXArray(actual, [batch, heads, dimension]),
                                     "reference_qk": MLXArray(expected, [batch, heads, dimension]),
                                     "positions": MLXArray(positions.map(Int32.init))],
                             url: root.appendingPathComponent("ane-batch/checkpoint-prepared-qk.safetensors"))
            }
            }
            result["phase"] = "complete"; try save()
            try #require(peakError < 0.01)
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
