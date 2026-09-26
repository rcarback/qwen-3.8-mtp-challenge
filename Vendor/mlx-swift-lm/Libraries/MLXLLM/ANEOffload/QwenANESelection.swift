public func buildQwenANESelectionMILText() -> String {
    let header = """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
    {
      func main<ios18>(tensor<fp16, [1, 2048, 1, 32]> x) {
        int32 axis
    """
    let ids = (0..<2048).map(String.init).joined(separator: ", ")
    var lines = [
        "tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([1])];",
        "bool keep = const()[name=string(\"keep\"), val=bool(true)];",
        "bool interleave = const()[name=string(\"interleave\"), val=bool(false)];",
        "fp16 absentID = const()[name=string(\"absentID\"), val=fp16(2048.0)];",
        "fp16 absentScore = const()[name=string(\"absentScore\"), val=fp16(-65504.0)];",
        "tensor<int32, [4]> idShape = const()[name=string(\"idShape\"), val=tensor<int32, [4]>([1, 2048, 1, 1])];",
        "tensor<fp16, [2048]> flatIDs = const()[name=string(\"flatIDs\"), val=tensor<fp16, [2048]>([\(ids)])];",
        "tensor<fp16, [1, 2048, 1, 1]> ids = reshape(x=flatIDs, shape=idShape)[name=string(\"ids\")];"
    ]
    let scalar = "tensor<fp16, [1, 1, 1, 32]>"
    let vector = "tensor<fp16, [1, 2048, 1, 32]>"
    let mask = "tensor<bool, [1, 2048, 1, 32]>"
    var remaining = "x"
    for index in 0..<32 {
        let tag = "s\(index)"
        lines += [
            "\(scalar) \(tag)value = reduce_max(x=\(remaining), axes=axes, keep_dims=keep)[name=string(\"\(tag)value\")];",
            "\(mask) \(tag)ties = equal(x=\(remaining), y=\(tag)value)[name=string(\"\(tag)ties\")];",
            "\(vector) \(tag)candidates = select(cond=\(tag)ties, a=ids, b=absentID)[name=string(\"\(tag)candidates\")];",
            "\(scalar) \(tag)id = reduce_min(x=\(tag)candidates, axes=axes, keep_dims=keep)[name=string(\"\(tag)id\")];",
            "\(mask) \(tag)chosen = equal(x=ids, y=\(tag)id)[name=string(\"\(tag)chosen\")];",
            "\(vector) \(tag)remaining = select(cond=\(tag)chosen, a=absentScore, b=\(remaining))[name=string(\"\(tag)remaining\")];"
        ]
        remaining = tag + "remaining"
    }
    let outputs = (0..<32).map { "s\($0)value" } + (0..<32).map { "s\($0)id" }
    lines.append("tensor<fp16, [1, 64, 1, 32]> y = concat(values=(\(outputs.joined(separator: ", "))), axis=axis, interleave=interleave)[name=string(\"y\")];")
    return header + " = const()[name=string(\"axis\"), val=int32(1)];\n"
        + lines.joined(separator: "\n") + "\n  } -> (y);\n}"
}

public enum QwenANESelectionError: Error {
    case invalidInput
    case invalidOutput
}

public final class QwenANEShortlistSelector {
    private let loop: ANEDirectDispatch.FP16Loop

    public init() throws {
        loop = try ANEDirectDispatch.FP16Loop(milText: buildQwenANESelectionMILText(),
            weightBlob: .init(), inputDim: 2048, outputDim: 64, batch: 32)
    }

    public func callAsFunction(_ logits: [Float16]) throws -> (ids: [Int32], scores: [Float16]) {
        guard logits.count == 32 * 98336 && logits.allSatisfy({ $0.isFinite && $0 > -65504 }) else {
            throw QwenANESelectionError.invalidInput
        }
        var slots = Array(repeating: [(score: Float16, id: Int32)](), count: 32)
        for start in stride(from: 0, to: 98330, by: 2048) {
            let count = min(2048, 98330 - start)
            var input = [Float16](repeating: -65504, count: 32 * 2048)
            for lane in 0..<32 {
                input.replaceSubrange((lane * 2048)..<(lane * 2048 + count),
                    with: logits[(lane * 98336 + start)..<(lane * 98336 + start + count)])
            }
            let selected = try loop(input)
            for lane in 0..<32 {
                for rank in 0..<32 {
                    let id = selected[lane * 64 + 32 + rank]
                    guard id >= 0 && id < 2048 && id.rounded() == id else {
                        throw QwenANESelectionError.invalidOutput
                    }
                    slots[lane].append((selected[lane * 64 + rank], Int32(start) + Int32(id)))
                }
            }
        }
        var input = [Float16](repeating: -65504, count: 32 * 2048)
        for lane in 0..<32 {
            for index in slots[lane].indices { input[lane * 2048 + index] = slots[lane][index].score }
        }
        let selected = try loop(input)
        var ids: [Int32] = [], scores: [Float16] = []
        for lane in 0..<32 {
            for rank in 0..<32 {
                let index = selected[lane * 64 + 32 + rank]
                guard index >= 0 && index < Float16(slots[lane].count) && index.rounded() == index else {
                    throw QwenANESelectionError.invalidOutput
                }
                let chosen = slots[lane][Int(index)]
                guard chosen.id < 98330 else { throw QwenANESelectionError.invalidOutput }
                ids.append(chosen.id); scores.append(selected[lane * 64 + rank])
            }
        }
        return (ids, scores)
    }
}
