// Probe individual MIL ops for compile acceptance + ANE placement.
// Each probe is its own tiny spec so a rejection is isolated.
import CoreML
import Foundation

struct OpProbe {
    let name: String
    let inputs: [(String,[Int])]
    let outputs: [(String,[Int])]
    let ops: Data
}

func mmFlags(_ p: String, tx: Bool, ty: Bool) -> Data {
    lenF(3, constBoolOp(name: "\(p)tx", value: tx)) + lenF(3, constBoolOp(name: "\(p)ty", value: ty))
}

func makeOpProbes() -> [OpProbe] {
    var probes: [OpProbe] = []
    let H = 8, L = 64, D = 32

    // 1. rank-3 batched matmul  x[H,L,D] @ y[H,L,D]^T -> [H,L,L]
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [H,L,L]))
        probes.append(OpProbe(name: "matmul_rank3_batched", inputs: [("a",[H,L,D]),("b",[H,L,D])], outputs: [("y",[H,L,L])], ops: o))
    }
    // 2. rank-4 batched matmul  [1,H,L,D] @ [1,H,L,D]^T
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [1,H,L,L]))
        probes.append(OpProbe(name: "matmul_rank4_batched", inputs: [("a",[1,H,L,D]),("b",[1,H,L,D])], outputs: [("y",[1,H,L,L])], ops: o))
    }
    // 3. const-x matmul with batch broadcast: const[L,L] @ y[H,L,D]
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, constOp(name: "w", dt: .fp16, shape: [L,L], payload: Data.f16(L*L, 0.01)))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","w"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [H,L,D]))
        probes.append(OpProbe(name: "matmul_constX_broadcast", inputs: [("a",[H,L,D])], outputs: [("y",[H,L,D])], ops: o))
    }
    // 4. cumsum along axis 1 of [H,L,D]
    do {
        var o = lenF(3, constIntsOp(name: "ax", values: [1]))
        o += lenF(3, constBoolOp(name: "ex", value: false))
        o += lenF(3, constBoolOp(name: "rv", value: false))
        o += lenF(3, op("cumsum", name: "cs", inputs: [("x","a"),("axis","ax"),("exclusive","ex"),("reverse","rv")], outName: "y", outType: .fp16, outShape: [H,L,D]))
        probes.append(OpProbe(name: "cumsum_axis1", inputs: [("a",[H,L,D])], outputs: [("y",[H,L,D])], ops: o))
    }
    // 5. exp elementwise
    do {
        let o = lenF(3, op("exp", name: "e", inputs: [("x","a")], outName: "y", outType: .fp16, outShape: [H,L,L]))
        probes.append(OpProbe(name: "exp", inputs: [("a",[H,L,L])], outputs: [("y",[H,L,L])], ops: o))
    }
    // 6. broadcast sub [H,L,1] - [H,1,L] -> [H,L,L]
    do {
        let o = lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","b")], outName: "y", outType: .fp16, outShape: [H,L,L]))
        probes.append(OpProbe(name: "sub_broadcast", inputs: [("a",[H,L,1]),("b",[H,1,L])], outputs: [("y",[H,L,L])], ops: o))
    }
    // 7. slice_by_index: take chunk 2 of axis0 -> [1,L,D]
    do {
        var o = lenF(3, constIntsVecOp(name: "bg", values: [2,0,0]))
        o += lenF(3, constIntsVecOp(name: "en", values: [3,L,D]))
        o += lenF(3, constIntsVecOp(name: "st", values: [1,1,1]))
        o += lenF(3, op("slice_by_index", name: "sl", inputs: [("x","a"),("begin","bg"),("end","en"),("stride","st")], outName: "y", outType: .fp16, outShape: [1,L,D]))
        probes.append(OpProbe(name: "slice_by_index", inputs: [("a",[H,L,D])], outputs: [("y",[1,L,D])], ops: o))
    }
    // 8. depthwise causal conv1d kernel 4 over (1,C,1,S)
    do {
        let C = 256, S = 64
        var o = lenF(3, constIntsOp(name: "st", values: [1,1])) + lenF(3, constIntsOp(name: "dl", values: [1,1]))
        o += lenF(3, constIntsOp(name: "pd", values: [0,0,3,0])) + lenF(3, constIntsOp(name: "gp", values: [C]))
        o += lenF(3, constStringOp(name: "pt", value: "custom"))
        o += lenF(3, constOp(name: "w", dt: .fp16, shape: [C,1,1,4], payload: Data.f16(C*4, 0.25)))
        o += lenF(3, op("conv", name: "cv", inputs: [("x","a"),("weight","w"),("strides","st"),("pad_type","pt"),("pad","pd"),("dilations","dl"),("groups","gp")], outName: "y", outType: .fp16, outShape: [1,C,1,S]))
        probes.append(OpProbe(name: "conv_depthwise_causal_k4", inputs: [("a",[1,C,1,S])], outputs: [("y",[1,C,1,S])], ops: o))
    }
    // 9. mul by scalar const (negation)
    do {
        var o = lenF(3, constScalarOp(name: "ng", dt: .fp16, payload: Data.f16(1, -1)))
        o += lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","ng")], outName: "y", outType: .fp16, outShape: [H,L,L]))
        probes.append(OpProbe(name: "mul_scalar_const", inputs: [("a",[H,L,L])], outputs: [("y",[H,L,L])], ops: o))
    }
    // 10. sigmoid (for beta) + silu (for z gate)
    do {
        var o = lenF(3, op("sigmoid", name: "sg", inputs: [("x","a")], outName: "s1", outType: .fp16, outShape: [H,L,D]))
        o += lenF(3, op("silu", name: "sl", inputs: [("x","s1")], outName: "y", outType: .fp16, outShape: [H,L,D]))
        probes.append(OpProbe(name: "sigmoid_silu", inputs: [("a",[H,L,D])], outputs: [("y",[H,L,D])], ops: o))
    }
    // 11. GVA broadcast batched matmul: [16,1,L,D] @ [16,3,L,D]^T -> [16,3,L,L]
    do {
        var o = mmFlags("p", tx: false, ty: true)
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","b"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [16,3,L,L]))
        probes.append(OpProbe(name: "matmul_gva_broadcast", inputs: [("a",[16,1,L,D]),("b",[16,3,L,D])], outputs: [("y",[16,3,L,L])], ops: o))
    }
    // 12. concat two [H,L,D] on axis 1
    do {
        var o = lenF(3, constIntsOp(name: "ax", values: [1]))
        o += lenF(3, constBoolOp(name: "il", value: false))
        var d = strF(1, "concat")
        d += inputBindingMulti(2, param: "values", varNames: ["a","b"])
        d += inputBinding(2, param: "axis", varName: "ax")
        d += inputBinding(2, param: "interleave", varName: "il")
        d += lenF(3, namedValue("y", .fp16, [H,2*L,D]))
        d += mapEntry(5, key: "name", value: stringValue("cc"))
        o += lenF(3, d)
        probes.append(OpProbe(name: "concat_axis1", inputs: [("a",[H,L,D]),("b",[H,L,D])], outputs: [("y",[H,2*L,D])], ops: o))
    }
    // 13. decay-matrix part: c[4,64,1] -> transpose,sub,mask,exp,mask
    do {
        var tril = [Float](repeating: 0, count: L*L)
        for i in 0..<L { for j in 0...i { tril[i*L+j] = 1 } }
        var o = lenF(3, constOp(name: "tl", dt: .fp16, shape: [L,L], payload: Data.f16Arr(tril)))
        o += lenF(3, constIntsOp(name: "pm", values: [0,2,1]))
        o += lenF(3, op("transpose", name: "t", inputs: [("x","a"),("perm","pm")], outName: "ct", outType: .fp16, outShape: [4,1,L]))
        o += lenF(3, op("sub", name: "s", inputs: [("x","a"),("y","ct")], outName: "df", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m1", inputs: [("x","df"),("y","tl")], outName: "dm", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("exp", name: "e", inputs: [("x","dm")], outName: "ex", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("mul", name: "m2", inputs: [("x","ex"),("y","tl")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "decay_part", inputs: [("a",[4,L,1])], outputs: [("y",[4,L,L])], ops: o))
    }
    // 14. T-inverse squaring chain from input M[4,64,64]
    do {
        var eye = [Float](repeating: 0, count: L*L)
        for i in 0..<L { eye[i*L+i] = 1 }
        var o = lenF(3, constOp(name: "ey", dt: .fp16, shape: [L,L], payload: Data.f16Arr(eye)))
        o += mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("add", name: "t0", inputs: [("x","ey"),("y","a")], outName: "T0", outType: .fp16, outShape: [4,L,L]))
        var tC = "T0", mC = "a"
        for i in 1...6 {
            o += lenF(3, op("matmul", name: "ms\(i)", inputs: [("x",mC),("y",mC),("transpose_x","ptx"),("transpose_y","pty")], outName: "M\(i)", outType: .fp16, outShape: [4,L,L]))
            o += lenF(3, op("matmul", name: "tm\(i)", inputs: [("x",tC),("y","M\(i)"),("transpose_x","ptx"),("transpose_y","pty")], outName: "TM\(i)", outType: .fp16, outShape: [4,L,L]))
            o += lenF(3, op("add", name: "ta\(i)", inputs: [("x",tC),("y","TM\(i)")], outName: "T\(i)", outType: .fp16, outShape: [4,L,L]))
            tC = "T\(i)"; mC = "M\(i)"
        }
        o += lenF(3, op("identity", name: "id", inputs: [("x",tC)], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "tchain", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // 15. const as matmul Y: [4,1,64] @ const[64,64]
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, constOp(name: "w", dt: .fp16, shape: [L,L], payload: Data.f16(L*L, 0.01)))
        o += lenF(3, op("matmul", name: "mm", inputs: [("x","a"),("y","w"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,1,L]))
        probes.append(OpProbe(name: "matmul_constY", inputs: [("a",[4,1,L])], outputs: [("y",[4,1,L])], ops: o))
    }
    // 16. cumsum via depthwise causal ones-conv over (1,H,1,L)
    do {
        var o = lenF(3, constIntsOp(name: "st", values: [1,1])) + lenF(3, constIntsOp(name: "dl", values: [1,1]))
        o += lenF(3, constIntsOp(name: "pd", values: [0,0,L-1,0])) + lenF(3, constIntsOp(name: "gp", values: [H]))
        o += lenF(3, constStringOp(name: "pt", value: "custom"))
        o += lenF(3, constOp(name: "w", dt: .fp16, shape: [H,1,1,L], payload: Data.f16(H*L, 1)))
        o += lenF(3, op("conv", name: "cv", inputs: [("x","a"),("weight","w"),("strides","st"),("pad_type","pt"),("pad","pd"),("dilations","dl"),("groups","gp")], outName: "y", outType: .fp16, outShape: [1,H,1,L]))
        probes.append(OpProbe(name: "cumsum_via_conv", inputs: [("a",[1,H,1,L])], outputs: [("y",[1,H,1,L])], ops: o))
    }
    // 17. mul broadcast with last-dim-1: [4,64,128] * [4,64,1]
    do {
        let o = lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","b")], outName: "y", outType: .fp16, outShape: [4,L,2*L]))
        probes.append(OpProbe(name: "mul_lastdim1", inputs: [("a",[4,L,2*L]),("b",[4,L,1])], outputs: [("y",[4,L,2*L])], ops: o))
    }
    return probes
}

@available(macOS 15.0, *)
func runOpProbes() async {
    for p in makeOpProbes() {
        let spec = buildSpec(inputs: p.inputs, outputs: p.outputs, ops: p.ops)
        let asset: MLModelAsset
        do { asset = try MLModelAsset(specification: spec) }
        catch {
            print("\(p.name): REJECTED  \(String("\(error)".replacingOccurrences(of: "\n", with: " ").prefix(180)))")
            continue
        }
        do {
            let r = try await planPlacement(asset, verbose: false)
            let detail = r.lines.map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " | ")
            print("\(p.name): ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)   [\(detail)]")
        } catch { print("\(p.name): PLAN FAILED \(error)") }
    }
}
