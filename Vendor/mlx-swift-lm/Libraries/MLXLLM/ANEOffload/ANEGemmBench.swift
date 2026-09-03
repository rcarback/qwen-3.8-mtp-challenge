// Shape sweep comparing one fp16 ANE 1x1-conv program against the equivalent
// MLX GPU matmul. No model, no routing: this measures the engines only.
import Foundation
import MLX

public struct ANEGemmSample: Codable {
    public let m: Int
    public let k: Int
    public let n: Int
    public let aneSeconds: Double
    public let gpuSeconds: Double
    /// GPU seconds over ANE seconds. Above 1.0 means the ANE is faster.
    public let rate: Double
}

public enum ANEGemmBench {
    /// `m` tokens, `k` input features, `n` output features. Shapes the ANE
    /// cannot build are omitted from the result rather than trapping, because
    /// the point of a sweep is to discover which shapes are viable.
    public static func sweep(shapes: [(m: Int, k: Int, n: Int)], iterations: Int) -> [ANEGemmSample] {
        var out = [ANEGemmSample]()
        for shape in shapes {
            guard shape.m > 0, shape.k > 0, shape.n > 0 else { continue }
            let w = MLXRandom.normal([shape.n, shape.k]).asType(.bfloat16)
            let x = MLXRandom.normal([shape.m, shape.k]).asType(.bfloat16)
            guard let projection = try? Qwen4ExpANEProjection(weight: w, sequenceLength: shape.m) else {
                continue
            }

            // Warm both engines once so neither pays first-dispatch cost.
            _ = try? projection(x)
            let wf = w.asType(.float32)
            eval(matmul(x.asType(.float32), wf.transposed()))

            var aneSeconds = Double.greatestFiniteMagnitude
            var gpuSeconds = Double.greatestFiniteMagnitude
            for _ in 0 ..< max(1, iterations) {
                let a0 = CFAbsoluteTimeGetCurrent()
                guard let prepared = try? projection.makeInput(x),
                    (try? projection.predict(prepared)) != nil
                else { break }
                _ = projection.readOutput(prepared)
                aneSeconds = min(aneSeconds, CFAbsoluteTimeGetCurrent() - a0)

                let g0 = CFAbsoluteTimeGetCurrent()
                let y = matmul(x.asType(.float32), wf.transposed())
                eval(y)
                gpuSeconds = min(gpuSeconds, CFAbsoluteTimeGetCurrent() - g0)
            }
            guard aneSeconds < .greatestFiniteMagnitude, gpuSeconds < .greatestFiniteMagnitude else {
                continue
            }
            out.append(
                ANEGemmSample(
                    m: shape.m, k: shape.k, n: shape.n,
                    aneSeconds: aneSeconds, gpuSeconds: gpuSeconds,
                    rate: gpuSeconds / aneSeconds))
        }
        return out
    }
}
