// Shared harness: build a CoreML spec around a MIL program with arbitrary
// fp16 inputs/outputs, load it, print per-op ANE/CPU/GPU placement, run it.
import CoreML
import Foundation

extension Data {
    static func f16(_ n: Int, _ v: Float) -> Data {
        var d = Data(count: n*2)
        d.withUnsafeMutableBytes { r in let p = r.bindMemory(to: Float16.self); for i in 0..<n { p[i]=Float16(v) } }
        return d
    }
    static func f16Arr(_ a: [Float]) -> Data {
        var d = Data(count: a.count*2)
        d.withUnsafeMutableBytes { r in let p = r.bindMemory(to: Float16.self); for i in 0..<a.count { p[i]=Float16(a[i]) } }
        return d
    }
}

func fd(_ n: String, _ shape: [Int]) -> Data {
    let arr = lenF(1, shape.reduce(Data()){ $0+varint(UInt64($1)) }) + varF(2, 65552) // fp16
    return strF(1, n) + lenF(3, lenF(5, arr))
}

func buildSpec(inputs: [(String,[Int])], outputs: [(String,[Int])], ops: Data) -> Data {
    var block = Data()
    for o in outputs { block += strF(2, o.0) }
    block += ops
    var fnInputs = Data()
    for i in inputs { fnInputs += lenF(1, namedValue(i.0, .fp16, i.1)) }
    let fn = fnInputs + strF(2, "CoreML8") + mapEntry(3, key: "CoreML8", value: block)
    let program = varF(1, 1) + mapEntry(2, key: "main", value: fn)
    var desc = Data()
    for i in inputs { desc += lenF(1, fd(i.0, i.1)) }
    for o in outputs { desc += lenF(10, fd(o.0, o.1)) }
    return varF(1, 9) + lenF(2, desc) + lenF(502, program)
}

struct PlacementResult { var ane = 0; var cpu = 0; var gpu = 0; var lines: [String] = [] }

@available(macOS 15.0, *)
func planPlacement(_ asset: MLModelAsset, verbose: Bool) async throws -> PlacementResult {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
    var r = PlacementResult()
    if case .program(let p) = plan.modelStructure, let f = p.functions["main"] {
        for o in f.block.operations where o.operatorName != "const" {
            let d = String(describing: plan.deviceUsage(for: o)?.preferred)
            let outName = o.outputs.first?.name ?? "?"
            let dev: String
            if d.contains("NeuralEngine") { r.ane += 1; dev = "ANE" }
            else if d.contains("CPU") { r.cpu += 1; dev = "CPU" }
            else { r.gpu += 1; dev = "GPU/other" }
            r.lines.append("  \(o.operatorName) -> \(outName): \(dev)")
            if verbose { print(r.lines.last!) }
        }
    }
    return r
}

func randF16Array(shape: [Int], range: ClosedRange<Float>, seed: inout UInt64) -> MLMultiArray {
    let n = shape.reduce(1, *)
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    a.withUnsafeMutableBytes { r, _ in
        let p = r.bindMemory(to: Float16.self)
        for i in 0..<n {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(seed >> 40) / Float(1 << 24)
            p[i] = Float16(range.lowerBound + u * (range.upperBound - range.lowerBound))
        }
    }
    return a
}

func floats(_ a: MLMultiArray) -> [Float] {
    let n = a.count
    var out = [Float](repeating: 0, count: n)
    a.withUnsafeBytes { r in
        let p = r.bindMemory(to: Float16.self)
        for i in 0..<n { out[i] = Float(p[i]) }
    }
    return out
}

/// Variadic argument binding (e.g. concat's `values`).
func inputBindingMulti(_ field: Int, param: String, varNames: [String]) -> Data {
    var args = Data()
    for v in varNames { args += lenF(1, strF(1, v)) }
    return mapEntry(field, key: param, value: args)
}

/// Deterministic per-tag pseudo-random fp16 weights (djb2-seeded LCG, uniform +-scale).
func weightFloats(_ tag: String, _ n: Int, scale: Float) -> [Float] {
    var h: UInt64 = 5381
    for b in tag.utf8 { h = h &* 33 &+ UInt64(b) }
    var out = [Float](repeating: 0, count: n)
    for i in 0..<n {
        h = h &* 6364136223846793005 &+ 1442695040888963407
        let u = Float(h >> 40) / Float(1 << 24)
        out[i] = (2*u - 1) * scale
    }
    return out
}
func weightData(_ tag: String, _ n: Int, scale: Float) -> Data {
    Data.f16Arr(weightFloats(tag, n, scale: scale))
}
