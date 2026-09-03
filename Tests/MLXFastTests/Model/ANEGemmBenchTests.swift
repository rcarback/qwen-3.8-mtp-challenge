import MLX
import XCTest

@testable import MLXLLM

final class ANEGemmBenchTests: XCTestCase {
    /// Needs the real ANE; opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1 on an idle machine.
    func testSweepReturnsOneSamplePerShapeWithPositiveTimings() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let shapes = [(m: 32, k: 256, n: 128), (m: 64, k: 256, n: 128)]
        let samples = ANEGemmBench.sweep(shapes: shapes, iterations: 3)
        XCTAssertEqual(samples.count, 2)
        for (i, s) in samples.enumerated() {
            XCTAssertEqual(s.m, shapes[i].m)
            XCTAssertEqual(s.k, shapes[i].k)
            XCTAssertEqual(s.n, shapes[i].n)
            XCTAssertGreaterThan(s.aneSeconds, 0)
            XCTAssertGreaterThan(s.gpuSeconds, 0)
            XCTAssertEqual(s.rate, s.gpuSeconds / s.aneSeconds, accuracy: 1e-9)
        }
    }

    /// A shape the ANE cannot build must be reported, not crash the sweep.
    func testUnbuildableShapeIsSkippedNotFatal() throws {
        guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else { return }
        let samples = ANEGemmBench.sweep(shapes: [(m: 0, k: 256, n: 128)], iterations: 1)
        XCTAssertTrue(samples.isEmpty)
    }
}
