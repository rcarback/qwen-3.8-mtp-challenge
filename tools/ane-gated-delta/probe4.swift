import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes4() async {
    var probes: [OpProbe] = []
    let L = 64
    // distinct inputs, no transpose
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "mm_distinct_notrans", inputs: [("a",[4,L,L]),("b",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // same input, ty=true
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "mm_same_ty", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // same input via identity copy, no transpose
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("identity", name: "id", inputs: [("x","a")], outName: "ac", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","ac"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "mm_same_via_identity", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // rectangular no-transpose distinct: [4,64,64]@[4,64,128]
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,128]))
        probes.append(OpProbe(name: "mm_rect_notrans", inputs: [("a",[4,L,L]),("b",[4,L,128])], outputs: [("y",[4,L,128])], ops: o))
    }
    // mask as INPUT: mul(input, maskinput) then exp then mul again (decay part redo)
    do {
        var o = lenF(3, constIntsOp(name: "pm", values: [0,2,1]))
        o += lenF(3, op("transpose", name: "t", inputs: [("x","a"),("perm","pm")], outName: "ct", outType: .fp16, outShape: [4,1,L]))
        o += lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","ct")], outName: "df", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m1", inputs: [("x","df"),("y","tl")], outName: "dm", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","dm")], outName: "ex", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","ex"),("y","tl")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "decay_maskinput", inputs: [("a",[4,L,1]),("tl",[1,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // tchain with eye as input, distinct-operand squarings via identity
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("add", name: "t0", inputs: [("x","ey"),("y","a")], outName: "T0", outType: .fp16, outShape: [4,L,L]))
        var tC = "T0", mC = "a"
        for i in 1...6 {
            o += lenF(3, op("identity", name: "idc\(i)", inputs: [("x",mC)], outName: "MC\(i)", outType: .fp16, outShape: [4,L,L]))
            o += lenF(3, op("matmul", name: "ms\(i)", inputs: [("x",mC),("y","MC\(i)"),("transpose_x","ptx"),("transpose_y","pty")], outName: "M\(i)", outType: .fp16, outShape: [4,L,L]))
            o += lenF(3, op("matmul", name: "tm\(i)", inputs: [("x",tC),("y","M\(i)"),("transpose_x","ptx"),("transpose_y","pty")], outName: "TM\(i)", outType: .fp16, outShape: [4,L,L]))
            o += lenF(3, op("add", name: "ta\(i)", inputs: [("x",tC),("y","TM\(i)")], outName: "T\(i)", outType: .fp16, outShape: [4,L,L]))
            tC = "T\(i)"; mC = "M\(i)"
        }
        o += lenF(3, op("identity", name: "id", inputs: [("x",tC)], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "tchain_eyeinput", inputs: [("a",[4,L,L]),("ey",[1,L,L])], outputs: [("y",[4,L,L])], ops: o))
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
