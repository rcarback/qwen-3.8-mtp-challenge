import Foundation

/// Applies the Qwen attention gate and output projection to packed [attention | gate] rows.
public func buildQwenANEGatedOutputMILText(programTag: String) -> String {
        var text = buildConvMILText(inputDim: 6144, outputDim: 5120, sequenceLength: 32, programTag: programTag)
        let signature = "func main<ios18>(tensor<fp16, [1, 6144, 1, 32]> x) {"
        precondition(text.contains(signature))
        let preparation = """
        func main<ios18>(tensor<fp16, [1, 12288, 1, 32]> x) {
          tensor<int32, [4]> aBegin = const()[name=string("aBegin"), val=tensor<int32, [4]>([0, 0, 0, 0])];
          tensor<int32, [4]> gBegin = const()[name=string("gBegin"), val=tensor<int32, [4]>([0, 6144, 0, 0])];
          tensor<int32, [4]> partSize = const()[name=string("partSize"), val=tensor<int32, [4]>([1, 6144, 1, 32])];
          tensor<fp16, [1, 6144, 1, 32]> a = slice_by_size(x=x, begin=aBegin, size=partSize)[name=string("a")];
          tensor<fp16, [1, 6144, 1, 32]> g = slice_by_size(x=x, begin=gBegin, size=partSize)[name=string("g")];
          fp16 negativeOne = const()[name=string("negativeOne"), val=fp16(-1.0)];
          fp16 one = const()[name=string("one"), val=fp16(1.0)];
          tensor<fp16, [1, 6144, 1, 32]> negativeGate = mul(x=g, y=negativeOne)[name=string("negativeGate")];
          tensor<fp16, [1, 6144, 1, 32]> exponential = exp(x=negativeGate)[name=string("exponential")];
          tensor<fp16, [1, 6144, 1, 32]> denominator = add(x=exponential, y=one)[name=string("denominator")];
          tensor<fp16, [1, 6144, 1, 32]> gated = real_div(x=a, y=denominator)[name=string("gated")];
        """
        text = text.replacingOccurrences(of: signature, with: preparation)
            .replacingOccurrences(of: "weight=w, x=x)", with: "weight=w, x=gated)")
    return text
}
