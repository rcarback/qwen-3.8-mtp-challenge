import CoreML
import Foundation

func makeOpProbes2() -> [OpProbe] {
    var probes: [OpProbe] = []
    let L = 64
    // A. mul input[4,L,L] * const[L,L]  (rank-2 broadcast)
    for (nm, cshape) in [("mulconst_r2", [L,L]), ("mulconst_r3b", [1,L,L]), ("mulconst_full", [4,L,L])] {
        var o = lenF(3, constOp(name: "w", dt: .fp16, shape: cshape, payload: Data.f16(cshape.reduce(1,*), 0.5)))
        o += lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","w")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: nm, inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // B. add input[4,L,L] + const[1,L,L]
    do {
        var o = lenF(3, constOp(name: "w", dt: .fp16, shape: [1,L,L], payload: Data.f16(L*L, 0.5)))
        o += lenF(3, op("add", name: "m", inputs: [("x","a"),("y","w")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "addconst_r3b", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // C. bare matmul chain, no consts at all: M2=M*M, M4=M2*M2 (input [4,L,L])
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "m1", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "m2", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("matmul", name: "m2o", inputs: [("x","m2"),("y","m2"),("transpose_x","ptx"),("transpose_y","pty")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "matmul_chain_noconst", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // D. matmul + add(input,input)
    do {
        var o = mmFlags("p", tx: false, ty: false)
        o += lenF(3, op("matmul", name: "m1", inputs: [("x","a"),("y","a"),("transpose_x","ptx"),("transpose_y","pty")], outName: "m2", outType: .fp16, outShape: [4,L,L]))
        o += lenF(3, op("add", name: "ad", inputs: [("x","m2"),("y","a")], outName: "y", outType: .fp16, outShape: [4,L,L]))
        probes.append(OpProbe(name: "matmul_add_inputs", inputs: [("a",[4,L,L])], outputs: [("y",[4,L,L])], ops: o))
    }
    // E. depthwise ones conv k=64 C=256
    do {
        let C = 256
        var o = lenF(3, constIntsOp(name: "st", values: [1,1])) + lenF(3, constIntsOp(name: "dl", values: [1,1]))
        o += lenF(3, constIntsOp(name: "pd", values: [0,0,L-1,0])) + lenF(3, constIntsOp(name: "gp", values: [C]))
        o += lenF(3, constStringOp(name: "pt", value: "custom"))
        o += lenF(3, constOp(name: "w", dt: .fp16, shape: [C,1,1,L], payload: Data.f16(C*L, 1)))
        o += lenF(3, op("conv", name: "cv", inputs: [("x","a"),("weight","w"),("strides","st"),("pad_type","pt"),("pad","pd"),("dilations","dl"),("groups","gp")], outName: "y", outType: .fp16, outShape: [1,C,1,L]))
        probes.append(OpProbe(name: "cumsum_conv_c256_k64", inputs: [("a",[1,C,1,L])], outputs: [("y",[1,C,1,L])], ops: o))
    }
    // F. rank-4 variants of the poison candidates
    do {
        var o = lenF(3, constOp(name: "w", dt: .fp16, shape: [1,1,L,L], payload: Data.f16(L*L, 0.5)))
        o += lenF(3, op("mul", name: "m", inputs: [("x","a"),("y","w")], outName: "y", outType: .fp16, outShape: [1,4,L,L]))
        probes.append(OpProbe(name: "mulconst_rank4", inputs: [("a",[1,4,L,L])], outputs: [("y",[1,4,L,L])], ops: o))
    }
    return probes
}

@available(macOS 15.0, *)
func runOpProbes2() async {
    for p in makeOpProbes2() {
        let spec = buildSpec(inputs: p.inputs, outputs: p.outputs, ops: p.ops)
        guard let asset = try? MLModelAsset(specification: spec) else { print("\(p.name): REJECTED"); continue }
        do {
            let r = try await planPlacement(asset, verbose: false)
            print("\(p.name): ANE=\(r.ane) CPU=\(r.cpu) GPU=\(r.gpu)")
        } catch { print("\(p.name): PLAN FAILED") }
    }
}
