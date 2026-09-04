// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// G independent SwiGLU expert FFNs as ONE fixed-shape ANE program.
//
// Shape choice. The ANE conv reads `tensor<fp16,[1, C_in, 1, S]>` and a 1x1
// conv mixes every input channel for every spatial position, so a plain conv
// cannot give different spatial rows different weights. Putting the group
// index on the CHANNEL axis and using a GROUPED conv (`groups=G`) does: the
// activation is `[1, G*hidden, 1, C]`, expert g owns channels
// `[g*hidden, (g+1)*hidden)`, and one conv with `groups=G` applies expert g's
// weight block to expert g's channel block. Three grouped convs plus the
// activation give the whole grouped SwiGLU in one program, one dispatch.
//
// The alternative spellings considered were a block-diagonal dense weight
// (G times the weight bytes, so it does not fit the program budget) and G
// separate slice/conv/concat chains in one program (same arithmetic, 5G more
// MIL ops). The grouped conv is what is implemented.
import CryptoKit
import Foundation
import MLX

/// The grouped SwiGLU-down program text. Same structure as
/// `buildSwiGLUDownMILText` with `groups=G` on all three convs and the
/// channel counts multiplied by `G`. `hiddenDim` is one expert's
/// intermediate width, `inputDim` one expert's hidden width.
public func buildGroupedSwiGLUDownMILText(
    inputDim: Int, hiddenDim: Int, groups: Int, sequenceLength: Int,
    gateOffset: UInt64, upOffset: UInt64, downOffset: UInt64,
    activation: ANEActivation = ANESplitConfig.activation,
    programTag: String = UUID().uuidString
) -> String {
    let gIn = groups * inputDim
    let gHid = groups * hiddenDim
    return """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}, {"mlxfast-program-tag", "\(programTag)"}})]
    {
      func main<ios18>(tensor<fp16, [1, \(gIn), 1, \(sequenceLength)]> x) {
        tensor<fp16, [\(gHid), \(inputDim), 1, 1]> gw = const()[name=string("gw"), val=tensor<fp16, [\(gHid), \(inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(gateOffset))))];
        tensor<fp16, [\(gHid), \(inputDim), 1, 1]> uw = const()[name=string("uw"), val=tensor<fp16, [\(gHid), \(inputDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(upOffset))))];
        tensor<fp16, [\(gIn), \(hiddenDim), 1, 1]> dw = const()[name=string("dw"), val=tensor<fp16, [\(gIn), \(hiddenDim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(\(downOffset))))];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(\(groups))];
        tensor<fp16, [1, \(gHid), 1, \(sequenceLength)]> gate = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=gw, x=x)[name=string("gate")];
        tensor<fp16, [1, \(gHid), 1, \(sequenceLength)]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=uw, x=x)[name=string("up")];
        \(activation.milLines(hiddenDim: gHid, sequenceLength: sequenceLength))
        tensor<fp16, [1, \(gHid), 1, \(sequenceLength)]> act = mul(x=silu_out, y=up)[name=string("swiglu")];
        tensor<fp16, [1, \(gIn), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=dw, x=act)[name=string("down")];
      } -> (y);
    }
    """
}

/// One compiled, loaded ANE program holding `groupSize` experts' SwiGLU FFNs.
///
/// The activation it consumes and produces is `[capacity, groupSize * hidden]`
/// fp16: row `c`, columns `[g*hidden, (g+1)*hidden)` is expert `g`'s `c`-th
/// capacity slot. Rows a caller has no token for must be zero-filled; the
/// program computes them regardless (fixed shape) and the caller drops them.
public final class ANEGroupedExpertMLP {
    public let hidden: Int
    public let inter: Int
    public let groupSize: Int
    /// Rows per expert. This is the program's compiled spatial length.
    public let capacity: Int
    /// fp16 weight bytes this program holds resident.
    public let programBytes: Int

    private let model: ANEInMemoryModel

    /// `gate`/`up` are `groupSize` arrays of `[inter, hidden]`; `down` of
    /// `[hidden, inter]`. Any float dtype; cast to fp16 inside.
    public init(gate: [MLXArray], up: [MLXArray], down: [MLXArray], capacity: Int) throws {
        precondition(!gate.isEmpty, "ANEGroupedExpertMLP needs at least one expert")
        precondition(gate.count == up.count && up.count == down.count,
                     "ANEGroupedExpertMLP: gate/up/down expert counts differ")
        groupSize = gate.count
        inter = gate[0].dim(0)
        hidden = gate[0].dim(1)
        self.capacity = capacity
        for e in 0 ..< groupSize {
            precondition(
                gate[e].shape == [inter, hidden] && up[e].shape == [inter, hidden]
                    && down[e].shape == [hidden, inter],
                "ANEGroupedExpertMLP: expert \(e) shapes \(gate[e].shape)/\(up[e].shape)/\(down[e].shape) "
                    + "do not match [\(inter), \(hidden)] / [\(hidden), \(inter)]")
        }
        programBytes = 3 * groupSize * inter * hidden * 2

        // Concatenate along the OUTPUT-channel axis: a grouped conv reads its
        // weight as `groups` consecutive blocks of `C_out/groups` rows.
        let gw = concatenated(gate.map { $0.asType(.float16) }, axis: 0)  // [G*inter, hidden]
        let uw = concatenated(up.map { $0.asType(.float16) }, axis: 0)  // [G*inter, hidden]
        let dw = concatenated(down.map { $0.asType(.float16) }, axis: 0)  // [G*hidden, inter]
        let (blob, offsets) = buildMultiWeightBlob(chunks: [f16Bytes(gw), f16Bytes(uw), f16Bytes(dw)])
        // Same weight-blob-hash program identity as `ANEFusedMLP`: the ANE
        // runtime derives the descriptor identity from the MIL text alone, so
        // without this tag every layer's group would collide on one staging
        // directory and only the first loaded program would hold its own
        // weights.
        let programTag = "sha256:" + SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
        let milText = buildGroupedSwiGLUDownMILText(
            inputDim: hidden, hiddenDim: inter, groups: groupSize, sequenceLength: capacity,
            gateOffset: offsets[0], upOffset: offsets[1], downOffset: offsets[2],
            programTag: programTag)
        let m = try ANEInMemoryModel(milText: milText, weightBlob: blob, weightFileName: "weight.bin")
        try m.compile()
        try m.load()
        model = m
    }

    deinit { model.unload() }

    /// CALLER THREAD ONLY (MLX). `packed` is `[capacity, groupSize * hidden]`.
    /// Contains BARRIER 1 (`ANEDirectDispatch.prepare` evals).
    public func makeInput(_ packed: MLXArray) throws -> ANEDirectDispatch.Prepared {
        precondition(
            packed.ndim == 2 && packed.dim(0) == capacity && packed.dim(1) == groupSize * hidden,
            "ANEGroupedExpertMLP.makeInput expected [\(capacity), \(groupSize * hidden)], got \(packed.shape)")
        return try ANEDirectDispatch.prepare(
            model: model, x: packed, inputDim: groupSize * hidden, outputDim: groupSize * hidden,
            sequenceLength: capacity)
    }

    /// BACKGROUND SAFE. No MLX.
    public func predict(_ p: ANEDirectDispatch.Prepared) throws { try ANEDirectDispatch.evaluate(p) }

    /// CALLER THREAD ONLY (MLX). Returns `[capacity, groupSize * hidden]` fp16.
    public func readOutput(_ p: ANEDirectDispatch.Prepared) -> MLXArray {
        Qwen4ExpANEFused.zeroCopyReadback ? ANEDirectDispatch.readZeroCopy(p) : ANEDirectDispatch.read(p)
    }

    /// CALLER THREAD ONLY. Stage, run, read. Tests only; the model call site
    /// uses the three phases so the GPU leg can overlap.
    public func callAsFunction(_ packed: MLXArray) throws -> MLXArray {
        let p = try makeInput(packed)
        try predict(p)
        return readOutput(p)
    }
}
