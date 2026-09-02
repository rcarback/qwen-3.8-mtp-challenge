import CoreML
import Foundation

func env(_ k: String, _ d: Int) -> Int { Int(ProcessInfo.processInfo.environment[k] ?? "") ?? d }

func buildStack(L: Int, C: Int, inter: Int, H: Int, D: Int, S: Int) -> Data {
    var ops = LayerEnc.hyper()
    var cur = "x"
    for l in 0 ..< L { ops += LayerEnc.layer(l, cur, C, inter, H, D, S); cur = "l\(l)_out" }
    // rename last output to "y" via identity
    ops += lenF(3, op("identity", name:"outid", inputs:[("x",cur)], outName:"y", outType:.fp16, outShape:[1,C,1,S]))
    let block = strF(2, "y") + ops
    let fn = lenF(1, namedValue("x", .fp16, [1,C,1,S])) + strF(2, "CoreML8")
        + mapEntry(3, key:"CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key:"main", value: fn)
    func fd(_ n: String) -> Data {
        let arr = lenF(1, [1,C,1,S].reduce(Data()){ $0+varint(UInt64($1)) }) + varF(2,65552)
        return strF(1,n) + lenF(3, lenF(5, arr))
    }
    let desc = lenF(1, fd("x")) + lenF(10, fd("y"))
    return varF(1,9) + lenF(2,desc) + lenF(502,program)
}

@available(macOS 15.0, *)
func run() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let L = env("LY_L", 1), C = env("LY_C", 5120), inter = env("LY_INTER", 17408)
    let H = env("LY_H", 20), D = C / env("LY_H", 20), S = env("LY_S", 128)
    let spec = buildStack(L: L, C: C, inter: inter, H: H, D: D, S: S)
    print("spec \(spec.count) bytes  L=\(L) C=\(C) inter=\(inter) H=\(H) D=\(D) S=\(S)")
    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) } catch { print("REJECTED \(String("\(error)".replacingOccurrences(of:"\n",with:" ").prefix(200)))"); exit(1) }
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    var ane=0, cpu=0, other=0
    do {
        let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
        if case .program(let p) = plan.modelStructure, let f = p.functions["main"] {
            for o in f.block.operations where !o.operatorName.hasSuffix("const") {
                let d = String(describing: plan.deviceUsage(for: o)?.preferred)
                if d.contains("NeuralEngine") { ane+=1 } else if d.contains("CPU") { cpu+=1 } else { other+=1 }
            }
        }
    } catch { print("plan failed \(error)") }
    print("placement ANE=\(ane) CPU=\(cpu) other=\(other)")
    if cpu > 0 { print("NOTE: CPU fallback present") }
    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) } catch { print("LOAD FAILED \(error)"); exit(1) }
    guard let x = try? MLMultiArray(shape: [1,C,1,S].map{NSNumber(value:$0)}, dataType:.float16) else { exit(1) }
    x.withUnsafeMutableBytes { r,_ in let p=r.bindMemory(to:Float16.self); for i in 0..<C*S { p[i]=Float16(0.01) } }
    let input = try! MLDictionaryFeatureProvider(dictionary:["x":MLFeatureValue(multiArray:x)])
    _ = try? await model.prediction(from: input)
    var best = Double.infinity
    for _ in 0..<env("LY_REPS",5) { let t0=Date(); _=try? await model.prediction(from:input); best=min(best,Date().timeIntervalSince(t0)) }
    // FLOPs per layer: qkv + o (2*C*C each) + gate+up (2*C*inter each) + down (2*inter*C) + attn (~2*2*H*S*D*S)
    let projF = Double(S) * (2.0*Double(C)*Double(C)*2 + 2.0*Double(C)*Double(inter)*3)
    let attnF = 2.0 * 2.0 * Double(H)*Double(S)*Double(D)*Double(S)
    let fl = Double(L) * (projF + attnF)
    print(String(format:"LAYERPOINT L=%d C=%d S=%d  ms=%.3f  TFLOPS=%.2f", L, C, S, 1000*best, fl/best/1e12))
}
if #available(macOS 15.0, *) { await run() } else { exit(3) }
