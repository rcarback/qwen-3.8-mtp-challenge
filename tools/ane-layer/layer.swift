// A representative Qwen full-attention layer as a MIL op chain, in (B,C,1,S):
//   rmsnorm(x) -> qkv conv -> [reshape to heads] SDPA [reshape back] ->
//   o conv -> +residual -> rmsnorm -> gate/up conv -> silu(gate)*up ->
//   down conv -> +residual
// Chained L times so activations pipeline. Placement + throughput measured.
import CoreML
import Foundation

extension Data {
    static func f16(_ n: Int, _ v: Float) -> Data {
        var d = Data(count: n*2)
        d.withUnsafeMutableBytes { r in let p = r.bindMemory(to: Float16.self); for i in 0..<n { p[i]=Float16(v) } }
        return d
    }
}

struct LayerEnc {
    // conv hyperparams shared across the whole program
    static func hyper() -> Data {
        lenF(3, constIntsOp(name:"st",values:[1,1])) + lenF(3, constIntsOp(name:"dl",values:[1,1]))
        + lenF(3, constIntsOp(name:"pd",values:[0,0,0,0])) + lenF(3, constIntsOp(name:"gp",values:[1]))
        + lenF(3, constStringOp(name:"pt",value:"valid"))
    }
    static func conv(_ nm: String, _ inV: String, _ w: String, _ out: String, _ Cout: Int, _ S: Int) -> Data {
        lenF(3, op("conv", name:nm, inputs:[("x",inV),("weight",w),("strides","st"),
            ("pad_type","pt"),("pad","pd"),("dilations","dl"),("groups","gp")],
            outName:out, outType:.fp16, outShape:[1,Cout,1,S]))
    }
    // RMSNorm over channel axis (axis 1) of [1,C,1,S]: x * rsqrt(mean(x^2)+eps)*g.
    // mean over axes [1] with keep_dims. Approximated with a fixed gamma=1.
    static func rmsnorm(_ pfx: String, _ inV: String, _ out: String, _ C: Int, _ S: Int) -> Data {
        var d = Data()
        d += lenF(3, op("mul", name:"\(pfx)sq", inputs:[("x",inV),("y",inV)], outName:"\(pfx)sq", outType:.fp16, outShape:[1,C,1,S]))
        d += lenF(3, constIntsVecOp(name:"\(pfx)ax", values:[1]))
        d += lenF(3, constBoolOp(name:"\(pfx)kd", value:true))
        d += lenF(3, op("reduce_mean", name:"\(pfx)mn", inputs:[("x","\(pfx)sq"),("axes","\(pfx)ax"),("keep_dims","\(pfx)kd")], outName:"\(pfx)mn", outType:.fp16, outShape:[1,1,1,S]))
        d += lenF(3, constScalarOp(name:"\(pfx)ep", dt:.fp16, payload:Data.f16(1,1e-6)))
        d += lenF(3, op("rsqrt", name:"\(pfx)rs", inputs:[("x","\(pfx)mn"),("epsilon","\(pfx)ep")], outName:"\(pfx)rs", outType:.fp16, outShape:[1,1,1,S]))
        d += lenF(3, op("mul", name:out, inputs:[("x",inV),("y","\(pfx)rs")], outName:out, outType:.fp16, outShape:[1,C,1,S]))
        return d
    }
    // Attention: for a PLACEMENT/PIPELINE probe we route the qkv output through
    // SDPA using a reshape to [1,H,S,D] and back. C = H*D on the q side.
    static func layer(_ l: Int, _ inV: String, _ C: Int, _ inter: Int, _ H: Int, _ D: Int, _ S: Int) -> Data {
        let p = "l\(l)_"
        var d = Data()
        d += rmsnorm("\(p)n1", inV, "\(p)nx", C, S)
        // qkv: keep C channels (self-attn q=k=v projection for the probe)
        d += lenF(3, constOp(name:"\(p)wq", dt:.fp16, shape:[C,C,1,1], payload:Data.f16(C*C,0.002)))
        d += conv("\(p)q", "\(p)nx", "\(p)wq", "\(p)qk", C, S)
        // reshape [1,C,1,S] -> [1,H,S,D] : C=H*D
        d += lenF(3, constIntsOp(name:"\(p)hs", values:[1,H,D,S]))
        d += lenF(3, op("reshape", name:"\(p)r1", inputs:[("x","\(p)qk"),("shape","\(p)hs")], outName:"\(p)hd", outType:.fp16, outShape:[1,H,D,S]))
        d += lenF(3, constIntsOp(name:"\(p)pm", values:[0,1,3,2]))
        d += lenF(3, op("transpose", name:"\(p)t1", inputs:[("x","\(p)hd"),("perm","\(p)pm")], outName:"\(p)qh", outType:.fp16, outShape:[1,H,S,D]))
        d += lenF(3, op("scaled_dot_product_attention", name:"\(p)sdpa", inputs:[("query","\(p)qh"),("key","\(p)qh"),("value","\(p)qh")], outName:"\(p)ao", outType:.fp16, outShape:[1,H,S,D]))
        // back to [1,C,1,S]
        d += lenF(3, op("transpose", name:"\(p)t2", inputs:[("x","\(p)ao"),("perm","\(p)pm")], outName:"\(p)at", outType:.fp16, outShape:[1,H,D,S]))
        d += lenF(3, constIntsOp(name:"\(p)cs", values:[1,C,1,S]))
        d += lenF(3, op("reshape", name:"\(p)r2", inputs:[("x","\(p)at"),("shape","\(p)cs")], outName:"\(p)ac", outType:.fp16, outShape:[1,C,1,S]))
        d += lenF(3, constOp(name:"\(p)wo", dt:.fp16, shape:[C,C,1,1], payload:Data.f16(C*C,0.002)))
        d += conv("\(p)o", "\(p)ac", "\(p)wo", "\(p)oo", C, S)
        d += lenF(3, op("add", name:"\(p)res1", inputs:[("x",inV),("y","\(p)oo")], outName:"\(p)h1", outType:.fp16, outShape:[1,C,1,S]))
        // MLP
        d += rmsnorm("\(p)n2", "\(p)h1", "\(p)mx", C, S)
        d += lenF(3, constOp(name:"\(p)wg", dt:.fp16, shape:[inter,C,1,1], payload:Data.f16(inter*C,0.002)))
        d += conv("\(p)g", "\(p)mx", "\(p)wg", "\(p)gg", inter, S)
        d += lenF(3, constOp(name:"\(p)wu", dt:.fp16, shape:[inter,C,1,1], payload:Data.f16(inter*C,0.002)))
        d += conv("\(p)u", "\(p)mx", "\(p)wu", "\(p)uu", inter, S)
        d += lenF(3, op("silu", name:"\(p)si", inputs:[("x","\(p)gg")], outName:"\(p)sg", outType:.fp16, outShape:[1,inter,1,S]))
        d += lenF(3, op("mul", name:"\(p)gm", inputs:[("x","\(p)sg"),("y","\(p)uu")], outName:"\(p)gu", outType:.fp16, outShape:[1,inter,1,S]))
        d += lenF(3, constOp(name:"\(p)wd", dt:.fp16, shape:[C,inter,1,1], payload:Data.f16(C*inter,0.002)))
        d += conv("\(p)d", "\(p)gu", "\(p)wd", "\(p)dd", C, S)
        d += lenF(3, op("add", name:"\(p)res2", inputs:[("x","\(p)h1"),("y","\(p)dd")], outName:"l\(l)_out", outType:.fp16, outShape:[1,C,1,S]))
        return d
    }
}
