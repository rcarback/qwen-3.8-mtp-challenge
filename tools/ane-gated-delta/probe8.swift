import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes8() async {
    var probes: [OpProbe] = []
    let L = 64
    // rank-4 mul->exp
    do {
        var o = lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","b")], outName: "d", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","d")], outName: "y", outType: .fp16, outShape: [1,4,L,L]))
        probes.append(OpProbe(name: "r4_mul_exp", inputs: [("a",[1,4,L,L]),("b",[1,4,L,L])], outputs: [("y",[1,4,L,L])], ops: o))
    }
    // rank-4 full elementwise chain + matmul
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","b")], outName: "d1", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("mul", name: "m1", inputs: [("x","d1"),("y","b")], outName: "d2", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","d2")], outName: "d3", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","d3"),("y","b")], outName: "d4", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","d4"),("y","c"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [1,4,L,128]))
        probes.append(OpProbe(name: "r4_ewchain_matmul", inputs: [("a",[1,4,L,L]),("b",[1,4,L,L]),("c",[1,4,L,128])], outputs: [("y",[1,4,L,128])], ops: o))
    }
    // rank-4 squaring via transpose+ty
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, constIntsOp(name: "pm", values: [0,1,3,2]))
        o += lenF(3, op("transpose", name: "t", inputs: [("x","a"),("perm","pm")], outName: "mt", outType: .fp16, outShape: [1,4,L,L]))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","mt"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [1,4,L,L]))
        probes.append(OpProbe(name: "r4_square_transpose_ty", inputs: [("a",[1,4,L,L])], outputs: [("y",[1,4,L,L])], ops: o))
    }
    // rank-4 same-operand matmul no transpose
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [1,4,L,L]))
        probes.append(OpProbe(name: "r4_mm_same_notrans", inputs: [("a",[1,4,L,L])], outputs: [("y",[1,4,L,L])], ops: o))
    }
    // rank-4 T-chain (3 iters) with eye input
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("add", name: "t0", inputs: [("x","ey"),("y","a")], outName: "T0", outType: .fp16, outShape: [1,4,L,L]))
        var tC = "T0", mC = "a"
        for i in 1...3 {
            o += lenF(3, op("matmul", name: "ms\(i)", inputs: [("x",mC),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "M\(i)", outType: .fp16, outShape: [1,4,L,L]))
            o += lenF(3, op("matmul", name: "tm\(i)", inputs: [("x",tC),("y","M\(i)"),("transpose_x","ptx"),("transpose_y","pty")], outName: "TM\(i)", outType: .fp16, outShape: [1,4,L,L]))
            o += lenF(3, op("add", name: "ta\(i)", inputs: [("x",tC),("y","TM\(i)")], outName: "T\(i)", outType: .fp16, outShape: [1,4,L,L]))
            tC = "T\(i)"; mC = "M\(i)"
        }
        o += lenF(3, op("mul", name: "fin", inputs: [("x",tC),("y","b")], outName: "y", outType: .fp16, outShape: [1,4,L,L]))
        probes.append(OpProbe(name: "r4_tchain_mixed", inputs: [("a",[1,4,L,L]),("b",[1,4,L,L]),("ey",[1,1,L,L])], outputs: [("y",[1,4,L,L])], ops: o))
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
