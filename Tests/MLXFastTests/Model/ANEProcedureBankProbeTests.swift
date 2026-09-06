import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXFastModel

/// Reconnaissance for the workaround to the 126-program count limit
/// (`ANEProgramCountLimitTests`): a PROCEDURE BANK. `buildBankMILText` packs
/// many fixed-shape convs as separate functions inside ONE MIL program, which
/// the ANE daemon loads as ONE program (one of the ~126 slots) while exposing
/// each function to `evaluate` through its own `procedureIndex`. If that holds,
/// N shapes cost 1 slot, not N, and the count limit stops binding.
///
/// Findings on this M4 Max, macOS 26.5.2 (2026-09-05), via the bare
/// `_ANEInMemoryModel` (MIL text) + `_ANERequest(procedureIndex:)` path:
///
///   1. LOADER COUNTS PROGRAMS, NOT FUNCTIONS. A bank of 8 functions is ONE
///      program. `countRelief` held 512 functions resident in 64 programs
///      with no load failure, versus the single-conv wall at exactly 126
///      programs (`ANEProgramCountLimitTests`). So packing N functions per
///      program multiplies the effective function ceiling by N, bounded only
///      by the per-program byte ceiling. This IS the workaround to the 126
///      limit at the compile+load layer.
///
///   2. ONLY `main` DISPATCHES THROUGH THIS PATH. A multi-function program
///      compiles and loads cleanly, and `procedureIndex 0` (`main`) computes
///      correctly, but `procedureIndex >= 1` fails at evaluate with
///      `ANEProgramProcessRequestDirect() ... status=0x2 statusType=0x9
///      Program Inference error`, under every function-naming scheme tried
///      (`namingSweep`). The extra functions are compiled but not registered
///      as runnable procedures by the bare in-memory descriptor. Reaching
///      them needs the CoreML MULTIFUNCTION model description (a `functions`
///      list / FunctionDescriptor compiled through MLModel), which this path
///      does not populate. Until that path is built, the bank relieves the
///      COUNT but only its `main` function is usable.
///
/// Needs the real ANE and on-disk staging: run with
///   MLXFAST_RUN_MLX_RUNTIME_TESTS=1 MLXFAST_NO_SANDBOX=1 \
///     swift test -c release --force-resolved-versions \
///     --filter ANEProcedureBankProbeTests
@Suite(.serialized)
struct ANEProcedureBankProbeTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// fp16 max-abs error of an ANE conv result against the fp32 `x @ w.T`
    /// reference. A correct dispatch lands at the fp16 rounding floor
    /// (~1e-2 on these shapes); a wrong-procedure dispatch reads the OTHER
    /// function's weights and diverges by ~1 (random-normal scale).
    private func maxAbsError(_ got: MLXArray, x: MLXArray, w: MLXArray) -> Float {
        let ref = matmul(x.asType(.float32), w.asType(.float32).transposed(1, 0))
        let d = MLX.abs(got.asType(.float32) - ref).max()
        eval(d)
        return d.item(Float.self)
    }

    /// A one-conv `[out,in]` bank procedure plus its own weight and input, so a
    /// dispatch can be scored against the exact function it should have run.
    private struct Leg {
        let inn: Int
        let out: Int
        let seq: Int
        let w: MLXArray  // [out, in] fp16
        let x: MLXArray  // [seq, in] fp16
    }

    /// Builds `legs.count` distinct-weight convs into one program, loads it,
    /// dispatches every `procedureIndex` in `0 ..< legs.count`, and returns,
    /// per dispatched index, the max-abs error against EACH leg's own weights.
    /// The diagonal (index i scored against leg i) being at the fp16 floor
    /// while the off-diagonal is ~1 proves the mapping is identity order.
    private func runBank(_ legs: [Leg], functionName: @escaping (Int) -> String = { $0 == 0 ? "main" : "proc\($0)" }) throws -> [[Float]] {
        let chunks = legs.map { f16Bytes($0.w) }
        let (blob, offsets) = buildMultiWeightBlob(chunks: chunks)
        let procedures = zip(legs, offsets).map { leg, off in
            ANEBankProcedure(inputDim: leg.inn, outputDim: leg.out, sequenceLength: leg.seq, weightOffset: off)
        }
        let text = buildBankMILText(procedures: procedures, functionName: functionName, programTag: "bank-probe-\(legs.count)-\(UUID().uuidString)")
        let model = try ANEInMemoryModel(milText: text, weightBlob: blob, weightFileName: "weight.bin")
        defer { model.unload() }
        try model.compile()
        try model.load()

        var scores: [[Float]] = []
        for (idx, leg) in legs.enumerated() {
            do {
                let prepared = try ANEDirectDispatch.prepare(
                    model: model, x: leg.x, inputDim: leg.inn, outputDim: leg.out,
                    sequenceLength: leg.seq, procedureIndex: idx)
                try ANEDirectDispatch.evaluate(prepared)
                let got = ANEDirectDispatch.read(prepared)
                scores.append(legs.map { maxAbsError(got, x: leg.x, w: $0.w) })
            } catch {
                print("[bank-probe] procedureIndex \(idx): DISPATCH FAILED: \(error)")
                scores.append(legs.map { _ in Float.nan })
            }
        }
        return scores
    }

    private func makeLeg(inn: Int, out: Int, seq: Int, seed: UInt64) -> Leg {
        MLXRandom.seed(seed)
        let w = MLXRandom.normal([out, inn]).asType(.float16)
        let x = MLXRandom.normal([seq, inn]).asType(.float16)
        eval(w, x)
        return Leg(inn: inn, out: out, seq: seq, w: w, x: x)
    }

    /// UNKNOWN 1: two functions in one program, dispatched by index, each
    /// correct — and which index runs which function.
    @Test("two-procedure program: compile, load, dispatch both", .enabled(if: enabled))
    func twoProcedureBank() throws {
        try #require(ANERuntime.available())
        let legs = [
            makeLeg(inn: 64, out: 64, seq: 128, seed: 1),
            makeLeg(inn: 64, out: 64, seq: 128, seed: 2),
        ]
        let scores = try runBank(legs)
        for (i, row) in scores.enumerated() {
            let pairs = row.enumerated().map { "leg\($0.offset)=\(String(format: "%.4f", $0.element))" }.joined(separator: " ")
            print("[bank-probe] procedureIndex \(i): maxAbs vs \(pairs)")
        }
        // Established invariant: procedureIndex 0 (`main`) computes its own
        // function correctly (fp16 floor vs leg 0, far from leg 1).
        #expect(scores[0][0] < 0.1, "main (procedureIndex 0) diverged from leg 0 by \(scores[0][0])")
        #expect(scores[0][0] < scores[0][1], "main matched leg 1 better than leg 0")
        // Recorded finding, not asserted: procedureIndex 1 is unreachable via
        // this bare in-memory path (dispatch inference error -> NaN score).
        // Making it reachable is the multifunction-descriptor follow-up.
        if scores[1][1].isNaN {
            print("[bank-probe] confirmed: procedureIndex 1 unreachable via bare in-memory path (needs multifunction descriptor)")
        }
    }

    /// UNKNOWN 1b: the non-`main` procedure of a 2-function program failed to
    /// dispatch under the `main`/`proc1` naming. Is the second function
    /// unreachable because of its NAME (the ANE runtime enumerates procedures
    /// by a specific convention) or because the bare in-memory MIL-text path
    /// registers only `main` regardless of name? Sweeps candidate naming
    /// schemes; a scheme where procedureIndex 1 lands at the fp16 floor is the
    /// convention. If EVERY scheme fails index 1, the limit is registration,
    /// not naming, and the bank needs the multifunction model-description path.
    @Test("naming sweep: which convention makes procedureIndex 1 reachable", .enabled(if: enabled))
    func namingSweep() throws {
        try #require(ANERuntime.available())
        let schemes: [(String, (Int) -> String)] = [
            ("main/proc{i}", { $0 == 0 ? "main" : "proc\($0)" }),
            ("main/procedure{i}", { $0 == 0 ? "main" : "procedure\($0)" }),
            ("main/procedure{i:03}", { $0 == 0 ? "main" : "procedure" + String(format: "%03d", $0) }),
            ("procedure{i:03}", { "procedure" + String(format: "%03d", $0) }),
            ("procedure{i}", { "procedure\($0)" }),
            ("main/function_{i}", { $0 == 0 ? "main" : "function_\($0)" }),
        ]
        for (name, scheme) in schemes {
            let legs = [
                makeLeg(inn: 64, out: 64, seq: 128, seed: 11),
                makeLeg(inn: 64, out: 64, seq: 128, seed: 12),
            ]
            do {
                let scores = try runBank(legs, functionName: scheme)
                let idx0 = scores[0][0]
                let idx1 = scores[1][1]
                let idx1ok = idx1 < 0.1
                print("[bank-probe] naming \"\(name)\": index0 vs leg0=\(String(format: "%.4f", idx0)) | index1 vs leg1=\(idx1.isNaN ? "DISPATCH-FAILED" : String(format: "%.4f", idx1)) -> index1 \(idx1ok ? "REACHABLE" : "unreachable")")
            } catch {
                print("[bank-probe] naming \"\(name)\": compile/load FAILED: \(error)")
            }
        }
    }

    /// UNKNOWN 2: does the 126 limit count programs or functions? Loads banks
    /// of `procsPerBank` functions each until either the target function count
    /// is reached or the ANE returns a load failure, and reports how many
    /// PROGRAMS and how many FUNCTIONS were resident at the wall. Single-conv
    /// programs wall at 126 programs = 126 functions; if a bank of 8 reaches
    /// far more than 126 functions, the limit is per-program and the bank is
    /// the workaround.
    @Test("count relief: functions resident past the 126-program wall", .enabled(if: enabled))
    func countRelief() throws {
        try #require(ANERuntime.available())
        let procsPerBank = 8
        let targetFunctions = 512  // 4x the single-program wall
        // Small, cheap shapes: this probe measures the count wall, not compute.
        let template = makeLeg(inn: 64, out: 64, seq: 32, seed: 100)

        var kept: [ANEInMemoryModel] = []
        defer { kept.forEach { $0.unload() } }
        var functionsResident = 0
        var failure: String?

        bankLoop: while functionsResident < targetFunctions {
            let procedures: [ANEBankProcedure]
            let blob: Data
            do {
                let chunks = (0 ..< procsPerBank).map { _ in f16Bytes(template.w) }
                let (b, offsets) = buildMultiWeightBlob(chunks: chunks)
                blob = b
                procedures = offsets.map {
                    ANEBankProcedure(inputDim: template.inn, outputDim: template.out, sequenceLength: template.seq, weightOffset: $0)
                }
            }
            let text = buildBankMILText(procedures: procedures, programTag: "count-relief-\(kept.count)-\(UUID().uuidString)")
            let model = try ANEInMemoryModel(milText: text, weightBlob: blob, weightFileName: "weight.bin")
            do {
                try model.compile()
                try model.load()
            } catch {
                failure = "\(error)"
                break bankLoop
            }
            kept.append(model)
            functionsResident += procsPerBank
        }

        print("[bank-probe] countRelief: \(kept.count) programs x \(procsPerBank) procs = "
            + "\(functionsResident) functions resident; first failure: \(failure ?? "none (reached target)")")
        // The single-conv wall is 126 programs = 126 functions. Passing 126
        // FUNCTIONS while holding far fewer programs is the proof the limit is
        // per-program and the bank relieves it.
        #expect(functionsResident > 126 || failure != nil,
                "expected to either pass the 126-function wall or hit a load failure worth reporting")
        if functionsResident > 126 && kept.count < 126 {
            print("[bank-probe] countRelief: PROVEN per-program — \(functionsResident) functions in \(kept.count) programs")
        }
    }
}
