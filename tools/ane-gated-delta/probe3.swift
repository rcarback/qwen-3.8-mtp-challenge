import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes3() async {
    // grid: mul(input, const-full) and matmul(square) over batch/size combos
    var probes: [OpProbe] = []
    for b in [1,2,3,4,6,8,16] {
        let L = 64
        var o = lenF(3, constOp(name: "w", dt: .fp16, shape: [b,L,L], payload: Data.f16(b*L*L, 0.5)))
        o += lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","w")], outName: "y", outType: .fp16, outShape: [b,L,L]))
        probes.append(OpProbe(name: "mulconst_b\(b)", inputs: [("a",[b,L,L])], outputs: [("y",[b,L,L])], ops: o))
    }
    for b in [4,8] { for d in [32,64,128] {
        let L = 64
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [b,L,L]))
        probes.append(OpProbe(name: "matmul_b\(b)_d\(d)", inputs: [("a",[b,L,d]),("b",[b,L,d])], outputs: [("y",[b,L,L])], ops: o))
    } }
    // square matmul without transpose at b=8
    do {
        let L = 64
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [8,L,L]))
        probes.append(OpProbe(name: "matmul_sq_b8_notrans", inputs: [("a",[8,L,L])], outputs: [("y",[8,L,L])], ops: o))
    }
    do {
        let L = 64
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "matmul_sq_b4_notrans", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    for p in probes {
        let spec = buildSpec(inputs: p.inputs, outputs: p.outputs, ops: p.ops)
        guard let asset = try? MLModelAsset(specification: spec) else { print("\(p.name): REJECTED"); continue }
        if let r = try? await planPlacement(asset, verbose: false) {
            print("\(p.name): ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)")
        } else { print("\(p.name): PLAN FAILED") }
    }
}
