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

    /// The gate must DECLINE on a bf16 checkpoint, not abort the process.
    ///
    /// The kernel's Metal source hard-codes `half` for the activation and for
    /// every scale/bias pointer, but MLX builds the kernel signature from the
    /// ACTUAL input dtypes -- `get_type_string` maps bfloat16 to the distinct
    /// type `bfloat16_t`. The reference Qwen3.8-Flash-Next checkpoint is bf16
    /// (`config.json` says `"dtype": "bfloat16"` and the switch_mlp
    /// scales/biases are BF16), so without a dtype guard every one of the
    /// other five guards passes -- the weights really are 4-bit affine
    /// group-32 with biases -- and the failure lands in the Metal JIT as a
    /// process abort rather than as a named fallback.
    ///
    /// This asserts the behaviour the gate advertises (decline and fall back),
    /// not the implementation's fp16 limitation. The sibling test above is
    /// the fp16 case; together they pin both sides of the guard.
    func testFusedGateDeclinesOnBFloat16Weights() throws {
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
        // bf16, as the real checkpoint is -- deliberately NOT cast to fp16.
        block.update(parameters: block.mapParameters(map: { $0.asType(.bfloat16) }))
        quantize(model: block) { path, _ in path.contains("switch_mlp") ? (32, 4, .affine) : nil }

        let x = MLXRandom.normal([1, 40, args.hiddenSize]).asType(.bfloat16)

        block.forceFusedRoutedMoE = true
        XCTAssertNil(
            block.fusedRoutedForward(x),
            "the fused path must decline on bf16 scales/biases; dispatching would "
                + "abort in the Metal JIT rather than fall back")

        // And the block as a whole still answers, through the fallback.
        let y = block(x)
        y.eval()
        XCTAssertEqual(y.shape, [1, 40, args.hiddenSize])
        XCTAssertTrue(
            MLX.all(MLX.isFinite(y.asType(.float32))).item(Bool.self),
            "fallback output must be finite")
    }

    /// One arm per process -- a combined test holding several weight stacks
    /// resident measures the allocator and residency behaviour rather than the
    /// kernel (see `MoEGatherTileTests.testFusedGateUpIsolated`'s doc comment
    /// for the prior instance of that mistake). Run:
    ///   for a in control gather tg64 tg128 tg256 tg512; do
    ///     MOE_FUSED_ARM=$a MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    ///       swift test --force-resolved-versions --filter testFusedArmTiming
    ///   done
    /// Ruling R4 predicted that the static work partition loses only its
    /// ragged final round and that the loss grows with threadgroup count G,
    /// so the sweep ran DOWN from the then-default of 256. The direction held
    /// from 512 to 64, but tg32 came out slower than tg64, so the curve has an
    /// interior optimum. The whole sweep is now moot for tuning purposes: in a
    /// rested round every arm clusters within ~6%, all of them 30-48x slower
    /// than `gather`. 64 is pinned as the argmin, not as a meaningful win.
    ///
    /// `gather` is the actual production path the fused kernel replaces --
    /// NOT `control`'s per-expert Swift loop, which no real code path takes.
    /// Fix round 1: `control` alone left open which direction a real
    /// `SwitchGLU` gather-QMM comparison would move the result, so this drives
    /// a real `SwitchGLU` (gate_up already fused into one stack, matching what
    /// `Qwen4ExpSparseMoeBlock.fusedRoutedForward` requires before it will even
    /// try the kernel) over the same fixture.
    func testFusedArmTiming() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "needs a real GPU")
        guard let arm = ProcessInfo.processInfo.environment["MOE_FUSED_ARM"] else {
            throw XCTSkip("set MOE_FUSED_ARM=control|gather|tg64|tg128|tg256|tg512")
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
        case "gather":
            // The real production path the fused kernel displaces: a
            // `SwitchGLU` whose gate_up stack is already fused into one
            // `[E, 2*hidden, inDim]` stack -- the same precondition
            // `Qwen4ExpSparseMoeBlock.fusedRoutedForward` requires before it
            // will even attempt the fused kernel (see Qwen4ExpMoE.swift) --
            // driven with the SAME quantized weights as every other arm here,
            // through `callAsFunction`'s real `gatherSort` -> fused gate_up
            // `gatherQuantizedMM` -> silu*up -> down `gatherQuantizedMM` ->
            // `scatterUnsort`, exactly as `SwitchLayers.swift` implements it.
            let switchGLU = SwitchGLU(
                inputDims: inDim, hiddenDims: hidden, numExperts: experts, fuseGateUp: true)
            let gateUpLinear = QuantizedSwitchLinear(
                inputDims: inDim, outputDims: 2 * hidden, numExperts: experts,
                weight: gw, scales: gs, biases: gb, bias: nil,
                groupSize: 32, bits: 4, mode: .affine)
            let downLinear = QuantizedSwitchLinear(
                inputDims: hidden, outputDims: inDim, numExperts: experts,
                weight: dw, scales: ds, biases: db, bias: nil,
                groupSize: 32, bits: 4, mode: .affine)
            var children = ModuleChildren()
            children["gate_up_proj"] = .value(gateUpLinear)
            children["down_proj"] = .value(downLinear)
            switchGLU.update(modules: children)

            // One "token" per fixture row, top-1 routing, assigned to the
            // SAME expert each row already belongs to per `offsets` -- so
            // `gatherSort`'s per-expert grouping reproduces the identical
            // per-expert row distribution the other arms measure against,
            // and `indices.size` (realRows) clears the `doSort` >= 64
            // threshold `SwitchGLU.callAsFunction` uses on real prompts.
            var rowExpertIds = [Int32](repeating: 0, count: realRows)
            for e in 0 ..< experts {
                let lo = Int(offsets[e]), hi = Int(offsets[e + 1])
                for r in lo ..< hi { rowExpertIds[r] = Int32(e) }
            }
            let xTokens = x.reshaped(1, realRows, inDim)
            let gatherIdx = MLXArray(rowExpertIds).reshaped(1, realRows, 1)
            run = { switchGLU(xTokens, gatherIdx) }
        default:
            guard arm.hasPrefix("tg"), let tg = Int(arm.dropFirst(2)) else {
                throw XCTSkip(
                    "unknown arm \(arm); expected gather|control|tg<N> (e.g. tg64, tg128)")
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
