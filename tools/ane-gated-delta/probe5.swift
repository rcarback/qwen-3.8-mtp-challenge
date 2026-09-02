import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes5() async {
    var probes: [OpProbe] = []
    let L = 64
    func ewProbe(_ nm: String, _ kind: String, _ sa: [Int], _ sb: [Int], _ so: [Int]) {
        let o = lenF(3, op(kind, name: "e", inputs: [("x","a"),("y","b")], outName: "y", outType: .fp16, outShape: so))
        probes.append(OpProbe(name: nm, inputs: [("a",sa),("b",sb)], outputs: [("y",so)], ops: o))
    }
    ewProbe("mul_b4_same", "mul", [4,L,L], [4,L,L], [4,L,L])
    ewProbe("mul_b4_bcast1", "mul", [4,L,L], [1,L,L], [4,L,L])
    ewProbe("add_b4_bcast1", "add", [4,L,L], [1,L,L], [4,L,L])
    ewProbe("sub_b4_inner", "sub", [4,L,1], [4,1,L], [4,L,L])
    // transpose [4,L,1] -> [4,1,L]
    do {
        var o = lenF(3, constIntsOp(name: "pm", values: [0,2,1]))
        o += lenF(3, op("transpose", name: "t", inputs: [("x","a"),("perm","pm")], outName: "y", outType: .fp16, outShape: [4,1,L]))
        probes.append(OpProbe(name: "transpose_lastdim1", inputs: [("a",[4,L,1])], outputs: [("y",[4,1,L])], ops: o))
    }
    // exp at b=4
    do {
        let o = lenF(3, op("exp", name: "e", inputs: [("x","a")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "exp_b4", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // decay part with FULL-batch mask input, no transpose op (cT fed as input)
    do {
        var o = lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","ct")], outName: "df", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m1", inputs: [("x","df"),("y","tl")], outName: "dm", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","dm")], outName: "ex", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","ex"),("y","tl")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "decay_fullmask_notranspose", inputs: [("a",[4,L,1]),("ct",[4,1,L]),("tl",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
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
