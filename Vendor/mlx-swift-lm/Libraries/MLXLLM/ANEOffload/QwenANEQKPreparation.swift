import Foundation

public enum QwenANEQKInputError: Error {
    case invalidShape
    case invalidPosition
    case nonfiniteInput
}

/// Packs Q without its interleaved gate, then K and an exact FP16 position per lane.
public func packQwenANEQKRows(qkv: [Float16], positions: [Int]) throws -> [Float16] {
    guard qkv.count == 32 * 14336, positions.count == 32 else { throw QwenANEQKInputError.invalidShape }
    guard positions.allSatisfy({ (0..<1024).contains($0) }) else { throw QwenANEQKInputError.invalidPosition }
    guard qkv.allSatisfy({ $0.isFinite }) else { throw QwenANEQKInputError.nonfiniteInput }
    var packed: [Float16] = []
    packed.reserveCapacity(32 * 7169)
    for lane in 0..<32 {
        let start = lane * 14336
        for head in 0..<24 {
            let offset = start + head * 512
            packed.append(contentsOf: qkv[offset..<(offset + 256)])
        }
        packed.append(contentsOf: qkv[(start + 12288)..<(start + 13312)])
        packed.append(Float16(positions[lane]))
    }
    return packed
}

/// Head RMSNorm and partial RoPE using ten binary position angles. Input comes from the checked packer.
public func buildQwenANEQKMILText(weights: [Float16]) -> String {
    precondition(weights.count == 7168 && weights.allSatisfy { $0.isFinite })
    let batch = 32, heads = 28, dimension = 256, rotary = 64
    let features = heads * dimension, outputFeatures = heads * dimension, channels = heads * dimension + 1
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
        lines.append("tensor<fp16, [\(features)]> weightFlat = const()[name=string(\"weightFlat\"), val=tensor<fp16, [\(features)]>([\(weights.map { String(Float($0)) }.joined(separator: ", "))])];")
        ints("weightShape", [1, heads, 1, dimension])
        lines.append("tensor<fp16, [1, \(heads), 1, \(dimension)]> weight = reshape(x=weightFlat, shape=weightShape)[name=string(\"weight\")];")
        lines.append("\(full) weighted = mul(x=normalized, y=weight)[name=string(\"weighted\")];")
        slice("left", "weighted", [0, 0, 0, 0], [batch, heads, 1, rotary / 2])
        slice("right", "weighted", [0, 0, 0, rotary / 2], [batch, heads, 1, rotary / 2])
        slice("tail", "weighted", [0, 0, 0, rotary], [batch, heads, 1, dimension - rotary])
        let rotated = "tensor<fp16, [\(batch), \(heads), 1, \(rotary / 2)]>"
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
        let text = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, \(channels), 1, \(batch)]> x) {
            \(lines.joined(separator: "\n"))
          } -> (y);
        }
        """
    return text
}
