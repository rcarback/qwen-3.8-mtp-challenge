import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastCore

/// Where does the quantized matmul stop being free in M?
///
/// `quantized.cpp:1418` routes every `M >= vector_limit` case to a tiled
/// kernel. `qmm_splitk` tiles M at 32 (`quantized.cpp:791`) and `qmm` at 32
/// (`quantized.cpp:719`), while `qmm_nax` tiles at 64 (`quantized.cpp:495`)
/// and is selected only on architecture generation 17 or newer
/// (`device.cpp:926`). Every M inside one tile launches the same threadgroup
/// count, so cost is flat across a tile and steps at the boundary.
///
/// This prices the step directly, at the four shapes a decode round actually
/// dispatches. The width where the step lands is the width the n-gram plan may
/// spend for free.
@Suite(.serialized)
struct WideVerifyMatmulTileTests {
    /// The model's own affine 4-bit group-64 projection shapes.
    /// hidden 5120, MLP intermediate 17408, fused gate/up 34816,
    /// gated-delta fused in-projection 16480.
    private static let shapes: [(label: String, K: Int, N: Int)] = [
        ("mlp.gate_up fused", 5120, 34816),
        ("mlp.gate", 5120, 17408),
        ("mlp.down", 17408, 5120),
        ("gdn.in_proj fused", 5120, 16480),
    ]

    @Test("quantized matmul row-tile step")
    func rowTileStep() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let widths = [1, 2, 4, 8, 9, 12, 16, 24, 31, 32, 33, 40, 48, 64, 65, 72]

        for shape in Self.shapes {
            let K = shape.K, N = shape.N
            let w = MLXRandom.randInt(0 ..< Int32.max, [N, K * 4 / 32])
                .asType(.uint32)
            let scales = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
            let biases = MLXRandom.normal([N, K / 64]).asType(.bfloat16)
            eval(w, scales, biases)

            print("\n\(shape.label)  [M, \(K)] x [\(K), \(N)]  affine4 gs64")
            print("      M   seconds   ms/row   vs M=1")
            var measured: [Int: Double] = [:]
            var base = 0.0
            for M in widths {
                let x = MLXRandom.normal([1, M, K]).asType(.bfloat16)
                eval(x)
                func once() -> MLXArray {
                    quantizedMM(
                        x, w, scales: scales, biases: biases, transpose: true,
                        groupSize: 64, bits: 4, mode: .affine)
                }
                for _ in 0 ..< 3 { eval(once()) }
                var best = Double.greatestFiniteMagnitude
                for _ in 0 ..< 5 {
                    let start = Date()
                    eval(once())
                    best = Swift.min(best, Date().timeIntervalSince(start))
                }
                measured[M] = best
                if M == 1 { base = best }
                print(String(
                    format: "  %5d  %8.4f  %7.4f  %6.2fx",
                    M, best, 1000 * best / Double(M), best / base))
            }

            // The plateau the spec measured end to end: 16 and 32 identical.
            let m16 = try #require(measured[16])
            let m32 = try #require(measured[32])
            #expect(
                m32 < 1.5 * m16,
                """
                \(shape.label): width 32 cost \(m32) against width 16 \(m16); \
                the row-tile hypothesis predicts these are within a small \
                constant of each other
                """)
        }
        print("")
    }
}
