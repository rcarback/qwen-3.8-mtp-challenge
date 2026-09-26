import Foundation

public enum QwenANECacheUpdateError: Error {
    case invalidShape(batch: Int, capacity: Int)
}

/// Replaces one KV slot per lane. Packed inputs contain cache [2, 4, T, 256],
/// new KV [2, 4, 1, 256], then a slot and 255 padding values. Slot -1 preserves an inactive lane.
/// Slots are tile-local FP16 integers; callers validate them before dispatch.
public func buildQwenANECacheUpdateMILText(batch: Int, capacity: Int) throws -> String {
    try buildQwenANECacheMILText(batch: batch, capacity: capacity, attention: false)
}

/// Reads an already updated cache plus queries and valid lengths; returns attention only.
public func buildQwenANECachedAttentionMILText(batch: Int, capacity: Int) throws -> String {
    try buildQwenANECacheMILText(batch: batch, capacity: capacity, attention: true)
}

private func buildQwenANECacheMILText(batch: Int, capacity: Int, attention: Bool) throws -> String {
    guard (1...32).contains(batch), (1...1024).contains(capacity) else {
        throw QwenANECacheUpdateError.invalidShape(batch: batch, capacity: capacity)
    }
    let cacheChannels = 2048 * capacity
    let channels = cacheChannels + (attention ? 8704 : 2304)
    var lines: [String] = []
    func ints(_ name: String, _ values: [Int]) {
        let value = values.map(String.init).joined(separator: ", ")
        lines.append("tensor<int32, [\(values.count)]> \(name) = const()[name=string(\"\(name)\"), val=tensor<int32, [\(values.count)]>([\(value)])];")
    }
    ints("packedShape", [1, channels, 1, batch])
    lines.append("tensor<fp16, [1, \(channels), 1, \(batch)]> packed = reshape(x=x, shape=packedShape)[name=string(\"packed\")];")
    ints("perm", [3, 0, 1, 2])
    var parts: [(String, Int, [Int], Int)] = [
        ("cache", cacheChannels, [batch, 8, capacity, 256], 0),
        ("newKV", 2048, [batch, 8, 1, 256], cacheChannels),
        ("slot", 1, [batch, 1, 1, 1], cacheChannels + 2048),
    ]
    if attention {
        parts.append(("query", 6144, [batch, 4, 6, 256], cacheChannels + 2304))
        parts.append(("length", 1, [batch, 1, 1, 1], cacheChannels + 8448))
    }
    for (name, count, shape, offset) in parts {
        ints("\(name)begin", [0, offset, 0, 0])
        ints("\(name)size", [1, count, 1, batch])
        ints("\(name)shape", shape)
        let dims = shape.map(String.init).joined(separator: ", ")
        lines.append("tensor<fp16, [1, \(count), 1, \(batch)]> \(name)raw = slice_by_size(x=packed, begin=\(name)begin, size=\(name)size)[name=string(\"\(name)raw\")];")
        lines.append("tensor<fp16, [\(batch), 1, \(count), 1]> \(name)t = transpose(x=\(name)raw, perm=perm)[name=string(\"\(name)t\")];")
        lines.append("tensor<fp16, [\(dims)]> \(name) = reshape(x=\(name)t, shape=\(name)shape)[name=string(\"\(name)\")];")
    }
    let slots = (0..<capacity).map(String.init).joined(separator: ", ")
    lines.append("tensor<fp16, [\(capacity)]> indices = const()[name=string(\"indices\"), val=tensor<fp16, [\(capacity)]>([\(slots)])];")
    ints("indexShape", [1, 1, capacity, 1])
    lines.append("tensor<fp16, [1, 1, \(capacity), 1]> positions = reshape(x=indices, shape=indexShape)[name=string(\"positions\")];")
    lines.append("tensor<bool, [\(batch), 1, \(capacity), 1]> match = equal(x=slot, y=positions)[name=string(\"match\")];")
    if attention {
        lines.append("tensor<fp16, [\(batch), 8, \(capacity), 256]> updated = identity(x=cache)[name=string(\"updated\")];")
    } else {
    lines.append("tensor<fp16, [\(batch), 8, \(capacity), 256]> updated = select(cond=match, a=newKV, b=cache)[name=string(\"updated\")];")
    }
    ints("flatShape", [batch, 1, 8 * capacity, 256])
    ints("outputPerm", [1, 2, 3, 0])
    lines.append("tensor<fp16, [\(batch), 1, \(8 * capacity), 256]> flat = reshape(x=updated, shape=flatShape)[name=string(\"flat\")];")
    var output = "flat"
    if attention {
        ints("keyBegin", [0, 0, 0, 0]); ints("valueBegin", [0, 4, 0, 0])
        ints("kvSize", [batch, 4, capacity, 256])
        lines.append("tensor<fp16, [\(batch), 4, \(capacity), 256]> keys = slice_by_size(x=updated, begin=keyBegin, size=kvSize)[name=string(\"keys\")];")
        lines.append("tensor<fp16, [\(batch), 4, \(capacity), 256]> values = slice_by_size(x=updated, begin=valueBegin, size=kvSize)[name=string(\"values\")];")
        ints("maskShape", [1, 1, 1, capacity])
        lines.append("tensor<fp16, [1, 1, 1, \(capacity)]> maskPositions = reshape(x=indices, shape=maskShape)[name=string(\"maskPositions\")];")
        lines.append("tensor<bool, [\(batch), 1, 1, \(capacity)]> visible = less(x=maskPositions, y=length)[name=string(\"visible\")];")
        lines.append("fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];")
        lines.append("fp16 negative = const()[name=string(\"negative\"), val=fp16(-10000.0)];")
        lines.append("tensor<fp16, [\(batch), 1, 1, \(capacity)]> mask = select(cond=visible, a=zero, b=negative)[name=string(\"mask\")];")
        lines.append("bool yes = const()[name=string(\"yes\"), val=bool(true)];")
        lines.append("bool no = const()[name=string(\"no\"), val=bool(false)];")
        lines.append("fp16 scale = const()[name=string(\"scale\"), val=fp16(0.0625)];")
        lines.append("int32 lastAxis = const()[name=string(\"lastAxis\"), val=int32(-1)];")
        let scores = "tensor<fp16, [\(batch), 4, 6, \(capacity)]>"
        lines.append("\(scores) dot = matmul(x=query, y=keys, transpose_x=no, transpose_y=yes)[name=string(\"dot\")];")
        lines.append("\(scores) scaled = mul(x=dot, y=scale)[name=string(\"scaled\")];")
        lines.append("\(scores) masked = add(x=scaled, y=mask)[name=string(\"masked\")];")
        lines.append("\(scores) probabilities = softmax(x=masked, axis=lastAxis)[name=string(\"probabilities\")];")
        lines.append("tensor<fp16, [\(batch), 4, 256, 6]> attendedT = matmul(x=values, y=probabilities, transpose_x=yes, transpose_y=yes)[name=string(\"attendedT\")];")
        ints("attendedPerm", [0, 1, 3, 2])
        lines.append("tensor<fp16, [\(batch), 4, 6, 256]> attended = transpose(x=attendedT, perm=attendedPerm)[name=string(\"attended\")];")
        ints("attentionShape", [batch, 1, 24, 256])
        lines.append("tensor<fp16, [\(batch), 1, 24, 256]> attentionFlat = reshape(x=attended, shape=attentionShape)[name=string(\"attentionFlat\")];")
        output = "attentionFlat"
    }
    lines.append("tensor<fp16, [1, \(attention ? 24 : 8 * capacity), 256, \(batch)]> y = transpose(x=\(output), perm=outputPerm)[name=string(\"y\")];")
    return """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "qwen-cache-spatial-\(batch)-\(capacity)-\(attention)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(channels / 256), 256, \(batch)]> x) {
        \(lines.joined(separator: "\n    "))
      } -> (y);
    }
    """
}
