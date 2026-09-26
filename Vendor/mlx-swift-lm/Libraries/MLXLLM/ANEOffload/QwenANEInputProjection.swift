import Foundation

/// Independent embedding/hidden RMSNorm followed by FC for 32 rows.
public func buildQwenANEInputMILText(normWeights: [Float16], programTag: String) -> String {
    precondition(normWeights.count == 10240 && normWeights.allSatisfy { $0.isFinite })
    let values = normWeights.map { String(Float($0)) }.joined(separator: ", ")
        var text = buildConvMILText(inputDim: 10240, outputDim: 5120, sequenceLength: 32, programTag: programTag)
        let signature = "func main<ios18>(tensor<fp16, [1, 10240, 1, 32]> x) {"
        precondition(text.contains(signature))
        let preparation = """
        \(signature)
          tensor<int32, [4]> normPerm = const()[name=string("normPerm"), val=tensor<int32, [4]>([3, 2, 0, 1])];
          tensor<int32, [4]> normBack = const()[name=string("normBack"), val=tensor<int32, [4]>([2, 3, 1, 0])];
          tensor<int32, [4]> pairShape = const()[name=string("pairShape"), val=tensor<int32, [4]>([32, 2, 1, 5120])];
          tensor<int32, [4]> flatShape = const()[name=string("flatShape"), val=tensor<int32, [4]>([32, 1, 1, 10240])];
          tensor<int32, [4]> weightShape = const()[name=string("weightShape"), val=tensor<int32, [4]>([1, 2, 1, 5120])];
          tensor<int32, [1]> normAxes = const()[name=string("normAxes"), val=tensor<int32, [1]>([-1])];
          bool normKeep = const()[name=string("normKeep"), val=bool(true)];
          fp16 normEpsilon = const()[name=string("normEpsilon"), val=fp16(0.000001)];
          fp16 normZero = const()[name=string("normZero"), val=fp16(0.0)];
          tensor<fp16, [10240]> normWeightFlat = const()[name=string("normWeightFlat"), val=tensor<fp16, [10240]>([\(values)])];
          tensor<fp16, [1, 2, 1, 5120]> normWeight = reshape(x=normWeightFlat, shape=weightShape)[name=string("normWeight")];
          tensor<fp16, [32, 1, 1, 10240]> rows = transpose(x=x, perm=normPerm)[name=string("rows")];
          tensor<fp16, [32, 2, 1, 5120]> pairs = reshape(x=rows, shape=pairShape)[name=string("pairs")];
          tensor<fp16, [32, 2, 1, 5120]> squares = mul(x=pairs, y=pairs)[name=string("squares")];
          tensor<fp16, [32, 2, 1, 1]> means = reduce_mean(x=squares, axes=normAxes, keep_dims=normKeep)[name=string("means")];
          tensor<fp16, [32, 2, 1, 1]> variances = add(x=means, y=normEpsilon)[name=string("variances")];
          tensor<fp16, [32, 2, 1, 1]> inverse = rsqrt(x=variances, epsilon=normZero)[name=string("inverse")];
          tensor<fp16, [32, 2, 1, 5120]> unit = mul(x=pairs, y=inverse)[name=string("unit")];
          tensor<fp16, [32, 2, 1, 5120]> weighted = mul(x=unit, y=normWeight)[name=string("weighted")];
          tensor<fp16, [32, 1, 1, 10240]> flattened = reshape(x=weighted, shape=flatShape)[name=string("flattened")];
          tensor<fp16, [1, 10240, 1, 32]> normalized = transpose(x=flattened, perm=normBack)[name=string("normalized")];
        """
        text = text.replacingOccurrences(of: signature, with: preparation)
            .replacingOccurrences(of: "weight=w, x=x)", with: "weight=w, x=normalized)")
    return text
}

/// QKV projection, optionally preceded by input RMSNorm.
public func buildQwenANEProjectionMILText(norm: [Float16]?, programTag: String) -> String {
        var text = buildConvMILText(inputDim: 5120, outputDim: 14336, sequenceLength: 32, programTag: programTag)
        if let norm {
            precondition(norm.count == 5120 && norm.allSatisfy { $0.isFinite })
            let signature = "func main<ios18>(tensor<fp16, [1, 5120, 1, 32]> x) {"
            precondition(text.contains(signature))
            let values = norm.map { String(Float($0)) }.joined(separator: ", ")
            let preparation = """
            \(signature)
              tensor<int32, [4]> normPerm = const()[name=string("normPerm"), val=tensor<int32, [4]>([3, 2, 0, 1])];
              tensor<int32, [4]> normBack = const()[name=string("normBack"), val=tensor<int32, [4]>([2, 3, 1, 0])];
              tensor<int32, [1]> normAxes = const()[name=string("normAxes"), val=tensor<int32, [1]>([-1])];
              bool normKeep = const()[name=string("normKeep"), val=bool(true)];
              fp16 normEpsilon = const()[name=string("normEpsilon"), val=fp16(0.000001)];
              fp16 normZero = const()[name=string("normZero"), val=fp16(0.0)];
              tensor<fp16, [5120]> normWeight = const()[name=string("normWeight"), val=tensor<fp16, [5120]>([\(values)])];
              tensor<fp16, [32, 1, 1, 5120]> normRows = transpose(x=x, perm=normPerm)[name=string("normRows")];
              tensor<fp16, [32, 1, 1, 5120]> normSquares = mul(x=normRows, y=normRows)[name=string("normSquares")];
              tensor<fp16, [32, 1, 1, 1]> normMean = reduce_mean(x=normSquares, axes=normAxes, keep_dims=normKeep)[name=string("normMean")];
              tensor<fp16, [32, 1, 1, 1]> normVariance = add(x=normMean, y=normEpsilon)[name=string("normVariance")];
              tensor<fp16, [32, 1, 1, 1]> normInverse = rsqrt(x=normVariance, epsilon=normZero)[name=string("normInverse")];
              tensor<fp16, [32, 1, 1, 5120]> normUnit = mul(x=normRows, y=normInverse)[name=string("normUnit")];
              tensor<fp16, [32, 1, 1, 5120]> normWeighted = mul(x=normUnit, y=normWeight)[name=string("normWeighted")];
              tensor<fp16, [1, 5120, 1, 32]> normOutput = transpose(x=normWeighted, perm=normBack)[name=string("normOutput")];
            """
            text = text.replacingOccurrences(of: signature, with: preparation)
                .replacingOccurrences(of: "weight=w, x=x)", with: "weight=w, x=normOutput)")
        }
    return text
}
