import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpANEDenseLaneTests: XCTestCase {
    /// Needs the real ANE; opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1 on an idle machine.
    func testProjectionMatchesGPUWithinFP16() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let w = MLXRandom.normal([512, 256]).asType(.bfloat16)
        let x = MLXRandom.normal([128, 256]).asType(.bfloat16)
        let proj = try Qwen4ExpANEProjection(weight: w, sequenceLength: 128)
        let got = try proj(x).asType(.float32)
        let want = matmul(x.asType(.float32), w.asType(.float32).transposed())
        XCTAssertEqual(got.shape, [128, 512])
        let rel = (abs(got - want).mean() / abs(want).mean()).item(Float.self)
        XCTAssertLessThan(rel, 2e-2, "ANE fp16 projection drifted \(rel) from the bf16 GPU matmul")
    }

    /// The micro-batched, layer-major prefill loop (GPU projections, no ANE) must
    /// match the plain forward: same tokens, same caches, same logits.
    func testMicroBatchedPrefillMatchesPlainForward() throws {
        let (model, dir) = try Qwen4ExpModelTests().makeModel()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ids = MLXArray([Int32(1), 2, 3, 7, 5, 6, 8]).reshaped(1, 7)  // 3 micro-batches of 2 plus a tail
        let plainCache = model.newCache(parameters: nil)
        let plain = model(ids, cache: plainCache)
        Qwen4ExpTextModel.forcedMicroBatch = 2
        defer { Qwen4ExpTextModel.forcedMicroBatch = nil }
        let pipedCache = model.newCache(parameters: nil)
        let piped = model(ids, cache: pipedCache)
        XCTAssertTrue(allClose(piped, plain, rtol: 2e-2, atol: 2e-3).item())
        XCTAssertEqual(pipedCache[3].offset, 7)
        XCTAssertEqual((pipedCache[1] as! ArraysCache)[3]!.asArray(Int32.self), [6, 8])
        // decode after the pipelined prefill continues from the same state
        let nextPlain = model(MLXArray([Int32(9)]).reshaped(1, 1), cache: plainCache)
        Qwen4ExpTextModel.forcedMicroBatch = nil
        let nextPiped = model(MLXArray([Int32(9)]).reshaped(1, 1), cache: pipedCache)
        XCTAssertTrue(allClose(nextPiped, nextPlain, rtol: 2e-2, atol: 2e-3).item())
    }

    func testLaneIsOffByDefault() {
        if ProcessInfo.processInfo.environment["MLX_ANE_DIRECT"] != "1" {
            XCTAssertFalse(Qwen4ExpANELane.armed(sequenceLength: 4096))
        }
        XCTAssertGreaterThanOrEqual(Qwen4ExpANELane.microBatch, 1)
    }
}
