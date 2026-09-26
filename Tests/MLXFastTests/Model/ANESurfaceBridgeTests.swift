import Foundation
import IOSurface
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"))
struct ANESurfaceBridgeTests {
    @Test("GPU input packing matches the original cast and transpose, including padding")
    func packingMatchesReference() throws {
        let properties: NSDictionary = [
            kIOSurfaceWidth: 65536, kIOSurfaceHeight: 1,
            kIOSurfaceBytesPerElement: 1, kIOSurfaceBytesPerRow: 65536,
            kIOSurfaceAllocSize: 65536,
        ]
        let surface = try #require(IOSurfaceCreate(properties as CFDictionary))
        let packing = try ANESurfacePacking(surface: surface)
        let hidden = 64, stride = 128
        for dtype in [DType.float16, .bfloat16, .float32] {
            for rows in [128, 1, 31, 32, 33, 100] {
                // Transpose creates a noncontiguous producer view.
                let x = ((MLXArray(0 ..< rows * hidden).asType(.float32) / 100) - 3)
                    .reshaped([hidden, rows]).transposed(1, 0).asType(dtype)
                let transfer = try packing.submit(x, width: hidden, rows: rows, stride: stride)
                try transfer.wait()
                let snapshot = Data(bytes: IOSurfaceGetBaseAddress(surface), count: hidden * stride * 2)
                let actual = MLXArray(snapshot, [hidden, stride], type: Float16.self)
                let expected = concatenated([
                    x.asType(.float16).transposed(1, 0),
                    MLXArray.zeros([hidden, stride - rows], dtype: .float16),
                ], axis: 1)
                // MLX eval completes tensor work before checking the bytes.
                eval(actual, expected)
                #expect(MLX.abs(actual - expected).max().item(Float.self) == 0,
                    "dtype=\(dtype), rows=\(rows)")
            }
        }
    }

    @Test("Packing preserves floating-point boundary values")
    func packingBoundaryValues() throws {
        let properties: NSDictionary = [
            kIOSurfaceWidth: 65536, kIOSurfaceHeight: 1,
            kIOSurfaceBytesPerElement: 1, kIOSurfaceBytesPerRow: 65536,
            kIOSurfaceAllocSize: 65536,
        ]
        let surface = try #require(IOSurfaceCreate(properties as CFDictionary))
        let packing = try ANESurfacePacking(surface: surface)
        let values: [Float] = [0, -0.0, 0x1p-24, -0x1p-24, 0x1p-14,
            65504, .infinity, -.infinity, .nan]
        for dtype in [DType.float16, .bfloat16, .float32] {
            let x = MLXArray(values).asType(dtype).reshaped([1, values.count])
            try packing.submit(x, width: values.count, rows: 1, stride: 1).wait()
            let actual = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: Float16.self)
            let expected = x.asType(.float16).asArray(Float16.self)
            for index in values.indices {
                if expected[index].isNaN {
                    #expect(actual[index].isNaN)
                } else {
                    #expect(actual[index].bitPattern == expected[index].bitPattern,
                        "dtype=\(dtype), index=\(index)")
                }
            }
        }
    }

    @Test("Fused join preserves cast-before-add rounding")
    func joinMatchesReference() {
        let pattern: [Float] = [1.00390625, -1.00390625, 0.125, 1.0078125]
        for (hidden, stride) in [(4, 32), (35, 128), (64, 128)] {
            let values = (0 ..< hidden).map { pattern[$0 % pattern.count] }
            let offsets = MLXArray(0 ..< hidden * stride).asType(.float32)
                .reshaped([hidden, stride]) / 8192
            let ane = (broadcast(MLXArray(values).reshaped([hidden, 1]),
                to: [hidden, stride]) + offsets).asType(.float16).contiguous()
            for dtype in [DType.float16, .bfloat16, .float32] {
                for rows in [1, 17, 32, 33, 100].filter({ $0 <= stride }) {
                    let transposed = ane[0..., 0 ..< rows].transposed(1, 0)
                    let gpu = (-transposed).asType(dtype).contiguous()
                    let expected = transposed.asType(dtype) + gpu
                    let joined = ANESurfaceJoin.callAsFunction(ane, gpu: gpu, rows: rows,
                        hidden: hidden, stride: stride, dtype: dtype)
                    let gathered = ANESurfaceJoin.callAsFunction(ane, gpu: nil, rows: rows,
                        hidden: hidden, stride: stride, dtype: dtype)
                    eval(expected, joined, gathered)
                    #expect(MLX.abs(joined - expected).max().item(Float.self) == 0)
                    #expect(MLX.abs(gathered - transposed.asType(dtype)).max().item(Float.self) == 0)
                }
            }
        }
    }
}
