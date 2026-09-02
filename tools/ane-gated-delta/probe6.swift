import CoreML
import Foundation

@available(macOS 15.0, *)
func runOpProbes6() async {
    var probes: [OpProbe] = []
    let L = 64
    func chain(_ nm: String, _ steps: [(String, String, String?)], inputs: [(String,[Int])]) {
        // steps: (opkind, outname, second-operand or nil for unary); input to each is previous out (or "a")
        var o = Data(); var cur = "a"
        for (i, s) in steps.enumerated() {
            let outN = i == steps.count-1 ? "y" : s.1
            if let y2 = s.2 {
                o += lenF(3, op(s.0, name: "o\(i)", inputs: [("x",cur),("y",y2)], outName: outN, outType: .fp16, outShape: [4,L,L]))
            } else {
                o += lenF(3, op(s.0, name: "o\(i)", inputs: [("x",cur)], outName: outN, outType: .fp16, outShape: [4,L,L]))
            }
            cur = outN
        }
        probes.append(OpProbe(name: nm, inputs: inputs, outputs: [("y",[4,L,L])], ops: o))
    }
    let ab: [(String,[Int])] = [("a",[4,L,L]),("b",[4,L,L])]
    chain("sub_mul", [("sub","d","b"),("mul","m","b")], inputs: ab)
    chain("mul_exp", [("mul","m","b"),("exp","e",nil)], inputs: ab)
    chain("exp_mul", [("exp","e",nil),("mul","m","b")], inputs: ab)
    chain("sub_mul_exp", [("sub","d","b"),("mul","m","b"),("exp","e",nil)], inputs: ab)
    chain("mul_exp_mul", [("mul","m","b"),("exp","e",nil),("mul","m2","b")], inputs: ab)
    chain("sub_mul_exp_mul", [("sub","d","b"),("mul","m","b"),("exp","e",nil),("mul","m2","b")], inputs: ab)
    // two muls sharing operand b, independent
    do {
        var o = lenF(3, op("mul", name: "m1", inputs: [("x","a"),("y","b")], outName: "u", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","u"),("y","b")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "mul_mul_sharedb", inputs: ab, outputs: [("y",[4,L,L])], ops: o))
    }
    for p in probes {
        let spec = buildSpec(inputs: p.inputs, outputs: p.outputs, ops: p.ops)
        guard let asset = try? MLModelAsset(specification: spec) else { print("\(p.name): REJECTED"); continue }
        if let r = try? await planPlacement(asset, verbose: false) {
            print("\(p.name): ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)")
        } else { print("\(p.name): PLAN FAILED") }
    }
}
