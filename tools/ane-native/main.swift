// Minimal pure-Swift Core ML model authoring + ANE execution.
//
// Proves the whole chain with no Python and no files:
//   hand-encoded .mlmodel protobuf bytes
//     -> MLModelAsset(specification:)        (macOS 13+, in memory)
//     -> MLComputePlan.load(asset:config:)   (which device got each op)
//     -> MLModel.load + prediction
//
// Field numbers are read from apple/coremltools mlmodel/format/*.proto,
// which is proto3, so repeated scalars are PACKED.
//   Model:                specificationVersion=1, description=2, neuralNetwork=500
//   ModelDescription:     input=1, output=10
//   FeatureDescription:   name=1, type=3
//   FeatureType:          multiArrayType=5
//   ArrayFeatureType:     shape=1 (packed), dataType=2   FLOAT16=65552
//   NeuralNetwork:        layers=1
//   NeuralNetworkLayer:   name=1, input=2, output=3, innerProduct=140
//   InnerProduct:         inputChannels=1, outputChannels=2, hasBias=10, weights=20
//   WeightParams:         float16Value=2
import CoreML
import Foundation

// MARK: - minimal protobuf writer

func varint(_ v: UInt64) -> Data {
    var v = v, out = Data()
    repeat {
        var b = UInt8(v & 0x7F)
        v >>= 7
        if v != 0 { b |= 0x80 }
        out.append(b)
    } while v != 0
    return out
}
func tag(_ field: Int, _ wire: UInt8) -> Data { varint(UInt64(field) << 3 | UInt64(wire)) }
/// length-delimited (wire type 2): sub-messages, strings, bytes, packed scalars
func lenF(_ field: Int, _ payload: Data) -> Data {
    tag(field, 2) + varint(UInt64(payload.count)) + payload
}
func varF(_ field: Int, _ v: UInt64) -> Data { tag(field, 0) + varint(v) }
func strF(_ field: Int, _ s: String) -> Data { lenF(field, Data(s.utf8)) }
func packedVarints(_ vs: [Int]) -> Data { vs.reduce(Data()) { $0 + varint(UInt64($1)) } }


/// proto3 repeated float is packed: wire type 2, payload = raw little-endian f32.
func packedFloats(_ vs: [Float]) -> Data {
    var d = Data()
    for v in vs { withUnsafeBytes(of: v.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
    return d
}
// MARK: - model construction

let FLOAT16: UInt64 = 65552
let FLOAT32: UInt64 = 65568

func arrayFeature(name: String, shape: [Int]) -> Data {
    let arrayType = lenF(1, packedVarints(shape)) + varF(2, FLOAT32)
    let featureType = lenF(5, arrayType)
    return strF(1, name) + lenF(3, featureType)
}

func buildModel(rows: Int, inCh: Int, outCh: Int, weights: Data, int8: Bool) -> Data {
    let desc = lenF(1, arrayFeature(name: "x", shape: [rows, inCh]))
        + lenF(10, arrayFeature(name: "y", shape: [rows, outCh]))
    let mode = ProcessInfo.processInfo.environment["ANE_DTYPE"] ?? "fp16"
    var weightParams: Data
    var ipExtra = Data()
    if mode == "int8dyn" {
        // int8DynamicQuantize: int8RawValue(31) + quantization(40).
        // Contract from the proto: hasBias false, numberOfBits 8,
        // LinearQuantizationParams with exactly one scale and no bias.
        let linear = lenF(1, packedFloats([0.02]))
        let quant = varF(1, 8) + lenF(101, linear)
        weightParams = lenF(31, weights) + lenF(40, quant)
        ipExtra = varF(22, 1)
    } else if mode == "int8w" {
        // Weight-only int8: uint8 codes in rawValue(30) plus per-tensor
        // affine scale+bias. No int8DynamicQuantize, so activations stay
        // fp16 and the layer remains an ordinary inner_product. This is
        // the path E1/E2/E3 measured at ~1.6x fp16 on the ANE.
        let linear = lenF(1, packedFloats([0.0004])) + lenF(2, packedFloats([-0.05]))
        let quant = varF(1, 8) + lenF(101, linear)
        weightParams = lenF(30, weights) + lenF(40, quant)
    } else {
        weightParams = lenF(2, weights)   // float16Value
    }
    let ip = varF(1, UInt64(inCh)) + varF(2, UInt64(outCh))
        + varF(10, 0) + lenF(20, weightParams) + ipExtra
    let layer = strF(1, "ip") + strF(2, "x") + strF(3, "y") + lenF(140, ip)
    let nn = lenF(1, layer) + varF(5, 1)
    return varF(1, 7) + lenF(2, desc) + lenF(500, nn)
}

// MARK: - run

@available(macOS 15.0, *)
func main() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let inCh = Int(ProcessInfo.processInfo.environment["ANE_K"] ?? "512") ?? 512
    let outCh = Int(ProcessInfo.processInfo.environment["ANE_N"] ?? "512") ?? 512

    // Weight matrix [C_out, C_in] as raw float16 bytes.
    let dt = ProcessInfo.processInfo.environment["ANE_DTYPE"] ?? "fp16"
    let useInt8 = (dt == "int8dyn" || dt == "int8w")
    var w = Data(count: inCh * outCh * (useInt8 ? 1 : 2))
    w.withUnsafeMutableBytes { raw in
        if useInt8 {
            let q = raw.bindMemory(to: Int8.self)
            if dt == "int8w" {
                let uq = raw.bindMemory(to: UInt8.self)
                for i in 0 ..< inCh * outCh { uq[i] = UInt8.random(in: 0 ... 255) }
            } else {
                for i in 0 ..< inCh * outCh { q[i] = Int8.random(in: -127 ... 127) }
            }
        } else {
        let p = raw.bindMemory(to: Float16.self)
        for i in 0 ..< inCh * outCh { p[i] = Float16(Float.random(in: -0.05 ... 0.05)) }
        }
    }
    let rows = Int(ProcessInfo.processInfo.environment["ANE_M"] ?? "1024") ?? 1024
    let spec = buildModel(rows: rows, inCh: inCh, outCh: outCh, weights: w, int8: useInt8)
    print("spec bytes: \(spec.count)")

    let asset: MLModelAsset
    do { asset = try MLModelAsset(specification: spec) }
    catch { print("ASSET FAILED: \(error)"); exit(1) }
    print("asset ok")

    let cfg = MLModelConfiguration()
    switch ProcessInfo.processInfo.environment["ANE_UNITS"] ?? "ane" {
    case "gpu": cfg.computeUnits = .cpuAndGPU
    case "cpu": cfg.computeUnits = .cpuOnly
    case "all": cfg.computeUnits = .all
    default: cfg.computeUnits = .cpuAndNeuralEngine
    }

    // Which device actually got the op? This is the placement proof.
    do {
        let plan = try await MLComputePlan.load(asset: asset, configuration: cfg)
        switch plan.modelStructure {
        case .neuralNetwork(let nn):
            for layer in nn.layers {
                let u = plan.deviceUsage(for: layer)
                print("layer \(layer.name) [\(layer.type)] -> preferred \(String(describing: u?.preferred))")
            }
        case .program(let program):
            if let fn = program.functions["main"] {
                for op in fn.block.operations {
                    let u = plan.deviceUsage(for: op)
                    print("op \(op.operatorName) -> preferred \(String(describing: u?.preferred))")
                }
            }
        default:
            print("compute plan: unsupported structure")
        }
    } catch { print("compute plan unavailable: \(error)") }

    let model: MLModel
    do { model = try await MLModel.load(asset: asset, configuration: cfg) }
    catch { print("LOAD FAILED: \(error)"); exit(1) }
    print("model loaded")

    print("step: alloc input")
    guard let x = try? MLMultiArray(shape: [NSNumber(value: rows), NSNumber(value: inCh)], dataType: .float32)
    else { print("input alloc failed"); exit(1) }
    x.withUnsafeMutableBytes { raw, _ in
        let p = raw.bindMemory(to: Float.self)
        for i in 0 ..< rows * inCh { p[i] = 0.01 }
    }
    print("step: provider")
    let input = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    do {
        print("step: predict")
        let out = try await model.prediction(from: input)
        print("step: read output")
        if let y = out.featureValue(for: "y")?.multiArrayValue {
            var best = Double.infinity
            let reps = Int(ProcessInfo.processInfo.environment["ANE_REPS"] ?? "5") ?? 5
            for _ in 0 ..< reps {
                let t0 = Date()
                _ = try await model.prediction(from: input)
                best = min(best, Date().timeIntervalSince(t0))
            }
            let fl = 2.0 * Double(rows) * Double(inCh) * Double(outCh)
            print(String(format: "NATIVEPOINT\t%@\t%d\t%d\t%.4f\t%.4f", (ProcessInfo.processInfo.environment["ANE_UNITS"] ?? "ane") + "/" + (ProcessInfo.processInfo.environment["ANE_DTYPE"] ?? "fp16"), inCh, outCh, 1000*best, fl/best/1e9))
            print("PREDICTED y shape \(y.shape) first \(y[0])")
        }
    } catch { print("PREDICT FAILED: \(error)"); exit(1) }
}

if #available(macOS 15.0, *) { await main() } else { exit(3) }
