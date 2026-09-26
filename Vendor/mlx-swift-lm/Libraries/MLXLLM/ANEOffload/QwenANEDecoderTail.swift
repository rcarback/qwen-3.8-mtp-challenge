import Foundation

/// Fuses both residual additions, post-attention RMSNorm, SwiGLU, and final RMSNorm.
public func buildQwenANEDecoderTailMILText(offsets: [UInt64], postNorm: [Float16],
                                         finalNorm: [Float16], programTag: String) -> String {
    precondition(offsets.count == 3 && postNorm.count == 5120 && finalNorm.count == 5120)
    precondition(postNorm.allSatisfy { $0.isFinite } && finalNorm.allSatisfy { $0.isFinite })
        var text = buildSwiGLUDownMILText(inputDim: 5120, hiddenDim: 17408, outputDim: 5120,
            sequenceLength: 32, gateOffset: offsets[0], upOffset: offsets[1], downOffset: offsets[2],
            activation: .expDiv, programTag: programTag)
        func normMIL(_ name: String, input: String, weights: [Float16]) -> String {
            let values = weights.map { String(Float($0)) }.joined(separator: ", ")
            return """
              tensor<fp16, [5120]> \(name)Weight = const()[name=string("\(name)Weight"), val=tensor<fp16, [5120]>([\(values)])];
              tensor<fp16, [32, 1, 1, 5120]> \(name)Rows = transpose(x=\(input), perm=normPerm)[name=string("\(name)Rows")];
              tensor<fp16, [32, 1, 1, 5120]> \(name)Squares = mul(x=\(name)Rows, y=\(name)Rows)[name=string("\(name)Squares")];
              tensor<fp16, [32, 1, 1, 1]> \(name)Mean = reduce_mean(x=\(name)Squares, axes=normAxes, keep_dims=normKeep)[name=string("\(name)Mean")];
              tensor<fp16, [32, 1, 1, 1]> \(name)Variance = add(x=\(name)Mean, y=normEpsilon)[name=string("\(name)Variance")];
              tensor<fp16, [32, 1, 1, 1]> \(name)Inverse = rsqrt(x=\(name)Variance, epsilon=normZero)[name=string("\(name)Inverse")];
              tensor<fp16, [32, 1, 1, 5120]> \(name)Unit = mul(x=\(name)Rows, y=\(name)Inverse)[name=string("\(name)Unit")];
              tensor<fp16, [32, 1, 1, 5120]> \(name)Weighted = mul(x=\(name)Unit, y=\(name)Weight)[name=string("\(name)Weighted")];
              tensor<fp16, [1, 5120, 1, 32]> \(name) = transpose(x=\(name)Weighted, perm=normBack)[name=string("\(name)")];
            """
        }
        let signature = "func main<ios18>(tensor<fp16, [1, 5120, 1, 32]> x) {"
        precondition(text.contains(signature) && text.contains("} -> (y);"))
        let preparation = """
        func main<ios18>(tensor<fp16, [1, 10240, 1, 32]> x) {
          tensor<int32, [4]> residualBegin = const()[name=string("residualBegin"), val=tensor<int32, [4]>([0, 0, 0, 0])];
          tensor<int32, [4]> attentionBegin = const()[name=string("attentionBegin"), val=tensor<int32, [4]>([0, 5120, 0, 0])];
          tensor<int32, [4]> partSize = const()[name=string("partSize"), val=tensor<int32, [4]>([1, 5120, 1, 32])];
          tensor<fp16, [1, 5120, 1, 32]> residual = slice_by_size(x=x, begin=residualBegin, size=partSize)[name=string("residual")];
          tensor<fp16, [1, 5120, 1, 32]> attention = slice_by_size(x=x, begin=attentionBegin, size=partSize)[name=string("attention")];
          tensor<fp16, [1, 5120, 1, 32]> afterAttention = add(x=residual, y=attention)[name=string("afterAttention")];
          tensor<int32, [4]> normPerm = const()[name=string("normPerm"), val=tensor<int32, [4]>([3, 2, 0, 1])];
          tensor<int32, [4]> normBack = const()[name=string("normBack"), val=tensor<int32, [4]>([2, 3, 1, 0])];
          tensor<int32, [1]> normAxes = const()[name=string("normAxes"), val=tensor<int32, [1]>([-1])];
          bool normKeep = const()[name=string("normKeep"), val=bool(true)];
          fp16 normEpsilon = const()[name=string("normEpsilon"), val=fp16(0.000001)];
          fp16 normZero = const()[name=string("normZero"), val=fp16(0.0)];
        \(normMIL("postNorm", input: "afterAttention", weights: postNorm))
        """
        let finish = """
          tensor<fp16, [1, 5120, 1, 32]> afterMLP = add(x=afterAttention, y=y)[name=string("afterMLP")];
        \(normMIL("finalNorm", input: "afterMLP", weights: finalNorm))
          } -> (finalNorm);
        """
        text = text.replacingOccurrences(of: signature, with: preparation)
            .replacingOccurrences(of: "weight=gw, x=x)", with: "weight=gw, x=postNorm)")
            .replacingOccurrences(of: "weight=uw, x=x)", with: "weight=uw, x=postNorm)")
            .replacingOccurrences(of: "} -> (y);", with: finish)
    return text
}
