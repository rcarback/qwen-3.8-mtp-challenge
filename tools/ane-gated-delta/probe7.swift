import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes7() async {
    var probes: [OpProbe] = []
    let L = 64
    // matmul with column vector y [4,L,L]@[4,L,1]
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,1]))
        probes.append(OpProbe(name: "matmul_colvec", inputs: [("a",[4,L,L]),("b",[4,L,1])], outputs: [("y",[4,L,1])], ops: o))
    }
    // row vector x: [4,1,L]@[4,L,L] ty=true
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,1,L]))
        probes.append(OpProbe(name: "matmul_rowvec_ty", inputs: [("a",[4,1,L]),("b",[4,L,L])], outputs: [("y",[4,1,L])], ops: o))
    }
    // elementwise chain WITH matmul anchor: sub->mul->exp->mul->matmul
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","b")], outName: "d1", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m1", inputs: [("x","d1"),("y","b")], outName: "d2", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","d2")], outName: "d3", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","d3"),("y","b")], outName: "d4", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","d4"),("y","c"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,128]))
        probes.append(OpProbe(name: "ewchain_matmul_anchor", inputs: [("a",[4,L,L]),("b",[4,L,L]),("c",[4,L,128])], outputs: [("y",[4,L,128])], ops: o))
    }
    // squaring via transpose+ty: MT=transpose(M); M2=matmul(M,MT,ty=true)
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, constIntsOp(name: "pm", values: [0,2,1]))
        o += lenF(3, op("transpose", name: "t", inputs: [("x","a"),("perm","pm")], outName: "mt", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","mt"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "square_via_transpose_ty", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // slice row L-1 of [4,L,L] -> [4,1,L]
    do {
        var o = lenF(3, constIntsVecOp(name: "bg", values: [0,L-1,0]))
        o += lenF(3, constIntsVecOp(name: "en", values: [4,L,L]))
        o += lenF(3, constIntsVecOp(name: "st", values: [1,1,1]))
        o += lenF(3, op("slice_by_index", name: "sl", inputs: [("x","a"),("begin","bg"),("end","en"),("stride","st")], outName: "y", outType: .fp16, outShape: [4,1,L]))
        probes.append(OpProbe(name: "slice_lastrow", inputs: [("a",[4,L,L])], outputs: [("y",[4,1,L])], ops: o))
    }
    for p in probes {
        let spec = buildSpec(inputs: p.inputs, outputs: p.outputs, ops: p.ops)
        guard let asset = try? MLModelAsset(specification: spec) else { print("\(p.name): REJECTED"); continue }
        if let r = try? await planPlacement(asset, verbose: false) {
            let bad = r.lines.filter { !$0.hasSuffix("ANE") }.map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " | ")
            print("\(p.name): ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)  \(bad.isEmpty ? "" : "[\(bad)]")")
        } else { print("\(p.name): PLAN FAILED") }
    }
}
