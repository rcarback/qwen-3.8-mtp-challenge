import Foundation

public enum QwenANEFinalSelectorError: Error {
    case invalidInput
    case invalidOutput
}

public final class QwenANEFinalSelector {
    private let loop: ANEDirectDispatch.FP16Loop

    public init() throws {
        loop = try ANEDirectDispatch.FP16Loop(milText: Self.mil(), weightBlob: Data(),
            inputDim: 96, outputDim: 2, batch: 32)
    }

    public func callAsFunction(scores: [Float16], tokenIDs: [Int32]) throws -> [Int32] {
        guard scores.count == 1024 && tokenIDs.count == 1024,
            scores.allSatisfy({ $0.isFinite }), tokenIDs.allSatisfy({ $0 >= 0 && $0 < 248320 })
        else { throw QwenANEFinalSelectorError.invalidInput }
        var input: [Float16] = []
        for lane in 0..<32 {
            let range = (lane * 32)..<((lane + 1) * 32)
            input.append(contentsOf: scores[range])
            input.append(contentsOf: tokenIDs[range].map { Float16($0 / 2048) })
            input.append(contentsOf: tokenIDs[range].map { Float16($0 % 2048) })
        }
        let output = try loop(input)
        var tokens: [Int32] = []
        for lane in 0..<32 {
            let high = output[lane * 2], low = output[lane * 2 + 1]
            guard high >= 0 && high < 122 && high.rounded() == high,
                low >= 0 && low < 2048 && low.rounded() == low else {
                throw QwenANEFinalSelectorError.invalidOutput
            }
            let token = Int32(high) * 2048 + Int32(low)
            guard token < 248320 && tokenIDs[(lane * 32)..<((lane + 1) * 32)].contains(token) else {
                throw QwenANEFinalSelectorError.invalidOutput
            }
            tokens.append(token)
        }
        return tokens
    }

    private static func mil() -> String {
        var slices: [String] = []
        for (name, offset) in [("scores", 0), ("high", 32), ("low", 64)] {
            slices.append("""
            tensor<int32, [4]> \(name)Begin = const()[name=string("\(name)Begin"), val=tensor<int32, [4]>([0, \(offset), 0, 0])];
            tensor<fp16, [1, 32, 1, 32]> \(name) = slice_by_size(x=x, begin=\(name)Begin, size=size)[name=string("\(name)")];
            """)
        }
        return """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, 96, 1, 32]> x) {
            tensor<int32, [4]> size = const()[name=string("size"), val=tensor<int32, [4]>([1, 32, 1, 32])];
            tensor<int32, [1]> axes = const()[name=string("axes"), val=tensor<int32, [1]>([1])];
            int32 axis = const()[name=string("axis"), val=int32(1)];
            bool keep = const()[name=string("keep"), val=bool(true)];
            bool interleave = const()[name=string("interleave"), val=bool(false)];
            fp16 absent = const()[name=string("absent"), val=fp16(2048.0)];
            \(slices.joined(separator: "\n"))
            tensor<fp16, [1, 1, 1, 32]> best = reduce_max(x=scores, axes=axes, keep_dims=keep)[name=string("best")];
            tensor<bool, [1, 32, 1, 32]> ties = equal(x=scores, y=best)[name=string("ties")];
            tensor<fp16, [1, 32, 1, 32]> highCandidates = select(cond=ties, a=high, b=absent)[name=string("highCandidates")];
            tensor<fp16, [1, 1, 1, 32]> bestHigh = reduce_min(x=highCandidates, axes=axes, keep_dims=keep)[name=string("bestHigh")];
            tensor<bool, [1, 32, 1, 32]> highTies = equal(x=highCandidates, y=bestHigh)[name=string("highTies")];
            tensor<fp16, [1, 32, 1, 32]> lowCandidates = select(cond=highTies, a=low, b=absent)[name=string("lowCandidates")];
            tensor<fp16, [1, 1, 1, 32]> bestLow = reduce_min(x=lowCandidates, axes=axes, keep_dims=keep)[name=string("bestLow")];
            tensor<fp16, [1, 2, 1, 32]> y = concat(values=(bestHigh, bestLow), axis=axis, interleave=interleave)[name=string("y")];
          } -> (y);
        }
        """
    }
}
