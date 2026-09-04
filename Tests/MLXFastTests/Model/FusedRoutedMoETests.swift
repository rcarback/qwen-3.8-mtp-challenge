import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class FusedRoutedMoETests: XCTestCase {
    func testQuantizedFusedMatchesMLXQuantizedMatmul() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        // Small but ASYMMETRIC, and with a skewed row distribution that includes
        // an empty expert and one expert wider than blockRows. Square or uniform
        // shapes hide transposed and block-boundary errors. hidden=32, inDim=96
        // (both multiples of 32, not equal or square) so a transposed-read bug
        // does not hide behind a coincidentally-matching geometry.
        let experts = 5, inDim = 96, hidden = 32
        let counts: [Int32] = [3, 0, 20, 1, 7]  // 31 rows, one > blockRows(16)
        var offsets: [Int32] = [0]
        for c in counts { offsets.append(offsets.last! + c) }
        let rows = Int(offsets.last!)

        MLXRandom.seed(11)
        let x = MLXRandom.normal([rows, inDim]).asType(.float16)
        let gateUpF = MLXRandom.normal([experts, 2 * hidden, inDim]).asType(.float16) * 0.05
        let downF = MLXRandom.normal([experts, inDim, hidden]).asType(.float16) * 0.05
        let (gw, gs, gbOpt) = MLX.quantized(gateUpF, groupSize: 32, bits: 4, mode: .affine)
        let (dw, ds, dbOpt) = MLX.quantized(downF, groupSize: 32, bits: 4, mode: .affine)
        guard let gb = gbOpt, let db = dbOpt else {
            XCTFail("expected biases for affine quantization mode")
            return
        }

        // Reference uses MLX's own quantized matmul on the SAME quantized
        // arrays, so any disagreement is the kernel's dequant, not quantization.
        var ref: [MLXArray] = []
        for e in 0..<experts {
            let lo = Int(offsets[e]), hi = Int(offsets[e + 1])
            if lo == hi { continue }
            let r = x[lo..<hi]
            let gu = MLX.quantizedMM(
                r, gw[e], scales: gs[e], biases: gb[e],
                transpose: true, groupSize: 32, bits: 4)
            let g = gu[.ellipsis, 0..<hidden]
            let u = gu[.ellipsis, hidden..<(2 * hidden)]
            let inter = (MLXNN.silu(g) * u).asType(.float16)
            ref.append(
                MLX.quantizedMM(
                    inter, dw[e], scales: ds[e], biases: db[e],
                    transpose: true, groupSize: 32, bits: 4))
        }
        let want = MLX.concatenated(ref, axis: 0)

        let rowOffsets = MLXArray(offsets)
        let blockOffsets = MoEWorkQueue.blockOffsets(
            rowOffsets: rowOffsets, blockRows: FusedRoutedMoE.blockRows)
        let got = FusedRoutedMoE.forward(
            xSorted: x,
            gateUpWeight: gw, gateUpScales: gs, gateUpBiases: gb,
            downWeight: dw, downScales: ds, downBiases: db,
            rowOffsets: rowOffsets, blockOffsets: blockOffsets,
            hiddenDim: hidden, inDim: inDim, numExperts: experts)

        got.eval()
        want.eval()
        XCTAssertEqual(got.shape, [rows, inDim])
        let maxAbs = MLX.abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxAbs, 3e-2, "fused quantized output diverged from quantizedMatmul")
    }

    /// The gate must not change the routed sum. Runs both arms in one process
    /// against the same synthetic block, because this test does not load the
    /// real checkpoint. Uses the `forceFusedRoutedMoE` debug override instead
    /// of `MLX_QWEN4EXP_FUSED_MOE` because `Qwen4ExpSparseMoeBlock.fusedRoutedMoE`
    /// is a `static let`: it is read once per process, and other tests in this
    /// target construct and forward `Qwen4ExpSparseMoeBlock`s before this one
    /// runs, which would already have cached it `false` regardless of the
    /// environment at the time this test executes.
    func testFusedGateProducesSameRoutedSumAsControl() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")

        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 128
        args.numExperts = 32
        args.numExpertsPerTok = 4
        args.moeIntermediateSize = 64
        args.sharedExpertIntermediateSize = 64

        MLXRandom.seed(3)
        let block = Qwen4ExpSparseMoeBlock(args)
        // Quantization preserves the weight's dtype in its scales/biases, and
        // the fused kernel's Metal source hard-codes `half` for those buffers
        // (see FusedRoutedMoEKernel.swift), so the pre-quantization weights
        // must be fp16 -- matching Task 3's own kernel test and Task 5's
        // brief, which both quantize fp16 arrays for the same reason.
        block.update(parameters: block.mapParameters(map: { $0.asType(.float16) }))
        quantize(model: block) { path, _ in path.contains("switch_mlp") ? (32, 4, .affine) : nil }

        let x = MLXRandom.normal([1, 40, args.hiddenSize]).asType(.float16)

        // Control arm first, gate still off, so this is the ordinary MLX path.
        let control = block(x)

        block.forceFusedRoutedMoE = true
        let fused = try XCTUnwrap(block.fusedRoutedForward(x))

        control.eval()
        fused.eval()

        let maxAbs = MLX.abs(control.asType(.float32) - fused.asType(.float32))
            .max().item(Float.self)
        XCTAssertLessThan(maxAbs, 5e-2, "fused routed path changed the block output")
    }

    /// One arm per process -- a combined test holding several weight stacks
    /// resident measures the allocator and residency behaviour rather than the
    /// kernel (see `MoEGatherTileTests.testFusedGateUpIsolated`'s doc comment
    /// for the prior instance of that mistake). Run:
    ///   for a in control tg64 tg128 tg256 tg512; do
    ///     MOE_FUSED_ARM=$a MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    ///       swift test --force-resolved-versions --filter testFusedArmTiming
    ///   done
    /// Ruling R4: the static work partition (see FusedRoutedMoEKernel.swift's
    /// Ruling C3 note) loses only its ragged final round, and that loss grows
    /// with threadgroup count G -- so this sweeps DOWN from the shipped
    /// default of 256 rather than up. If tg64 wins, rerun with MOE_FUSED_ARM=tg32.
    func testFusedArmTiming() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")
        guard let arm = ProcessInfo.processInfo.environment["MOE_FUSED_ARM"] else {
            throw XCTSkip("set MOE_FUSED_ARM=control|tg64|tg128|tg256|tg512")
        }

        // Real per-layer geometry, with the MEASURED skew rather than uniform
        // routing. Uniform routing is what made the capacity-GEMM result wrong
        // (see MoEGatherTileTests).
        //
        // The brief's original construction clamped each expert's count
        // against a `remaining` row budget while iterating in expert order,
        // which exhausted the budget before reaching the hot expert at index
        // 511 and left it far short of its intended row count. Build the
        // distribution so every expert's count is fixed up front and the
        // counts sum to `rows` exactly by construction, then assert that.
        let experts = 512, inDim = 2560, hidden = 640, rows = 7000
        let idleCount = 190  // within the measured 147-239 idle range
        let hotIndex = experts - 1
        let hotCount: Int32 = 574  // top of the measured 151-574 hot-expert range
        let activeExperts = experts - idleCount - 1
        let remainderRows = rows - Int(hotCount)
        let base = Int32(remainderRows / activeExperts)
        let extra = remainderRows % activeExperts
        var counts = [Int32](repeating: 0, count: experts)
        var activeSeen = 0
        for e in 0 ..< experts {
            if e < idleCount {
                counts[e] = 0
            } else if e == hotIndex {
                counts[e] = hotCount
            } else {
                counts[e] = base + (activeSeen < extra ? 1 : 0)
                activeSeen += 1
            }
        }
        var offsets: [Int32] = [0]
        for c in counts { offsets.append(offsets.last! + c) }
        let realRows = Int(offsets.last!)
        XCTAssertEqual(realRows, rows, "synthetic row counts must sum to the intended total")

        MLXRandom.seed(5)
        let x = MLXRandom.normal([realRows, inDim]).asType(.float16)
        let gateUpF = MLXRandom.normal([experts, 2 * hidden, inDim]).asType(.float16) * 0.02
        let downF = MLXRandom.normal([experts, inDim, hidden]).asType(.float16) * 0.02
        let (gw, gs, gbOpt) = MLX.quantized(gateUpF, groupSize: 32, bits: 4, mode: .affine)
        let (dw, ds, dbOpt) = MLX.quantized(downF, groupSize: 32, bits: 4, mode: .affine)
        guard let gb = gbOpt, let db = dbOpt else {
            XCTFail("expected biases for affine quantization mode")
            return
        }
        let rowOffsets = MLXArray(offsets)
        let blockOffsets = MoEWorkQueue.blockOffsets(
            rowOffsets: rowOffsets, blockRows: FusedRoutedMoE.blockRows)

        let run: () -> MLXArray
        switch arm {
        case "control":
            // MLX.quantizedMM, not the deprecated quantizedMatmul the brief
            // sketched -- this file already prefers quantizedMM above, and
            // quantizedMatmul is `@available(*, deprecated)` in this vendor
            // snapshot, which would add a new compiler warning.
            run = {
                var out: [MLXArray] = []
                for e in 0 ..< experts {
                    let lo = Int(offsets[e]), hi = Int(offsets[e + 1])
                    if lo == hi { continue }
                    let r = x[lo ..< hi]
                    let gu = MLX.quantizedMM(
                        r, gw[e], scales: gs[e], biases: gb[e],
                        transpose: true, groupSize: 32, bits: 4)
                    let inter = (MLXNN.silu(gu[.ellipsis, 0 ..< hidden])
                        * gu[.ellipsis, hidden ..< (2 * hidden)]).asType(.float16)
                    out.append(MLX.quantizedMM(
                        inter, dw[e], scales: ds[e], biases: db[e],
                        transpose: true, groupSize: 32, bits: 4))
                }
                return MLX.concatenated(out, axis: 0)
            }
        default:
            guard arm.hasPrefix("tg"), let tg = Int(arm.dropFirst(2)) else {
                throw XCTSkip("unknown arm \(arm); expected control|tg<N> (e.g. tg64, tg128)")
            }
            run = {
                FusedRoutedMoE.forward(
                    xSorted: x,
                    gateUpWeight: gw, gateUpScales: gs, gateUpBiases: gb,
                    downWeight: dw, downScales: ds, downBiases: db,
                    rowOffsets: rowOffsets, blockOffsets: blockOffsets,
                    hiddenDim: hidden, inDim: inDim, numExperts: experts,
                    threadgroups: tg)
            }
        }

        run().eval()  // warm
        var best = Double.greatestFiniteMagnitude
        for _ in 0 ..< 5 {
            let t0 = Date()
            run().eval()
            best = min(best, Date().timeIntervalSince(t0))
        }
        print("MOE_FUSED_ARM=\(arm) best=\(String(format: "%.3f", best * 1000)) ms")
        XCTAssertLessThan(best, 60.0, "arm \(arm) did not complete in a usable time")
    }
}
