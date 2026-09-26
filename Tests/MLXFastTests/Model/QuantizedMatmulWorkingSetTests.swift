import Foundation
import MLX
import MLXRandom
import Testing

/// WHY DOES THE SAME GEMM RUN AT HALF RATE INSIDE PREFILL?
///
/// A 1024-token prefill chunk achieves 6.11 TFLOPS across its projection GEMMs,
/// while the same shapes microbenchmark at 12.2 TFLOPS and a bf16 square control
/// reads 14.7 TFLOPS in a fresh process. The decisive clue is in
/// `tools/gemm-point-sweep.sh`: that control drops to about 6.5 TFLOPS once ANY
/// other block has run in the same process, the effect SURVIVES
/// `Memory.clearCache()`, and it does NOT survive a process boundary.
///
/// That signature keys on process state, not on elapsed time or temperature —
/// running one block does not heat a GPU — so a thermal explanation does not fit
/// and is not what this probe tests. Three hypotheses do fit, and they are
/// separable by construction:
///
///  * **DATA REUSE.** A looped microbenchmark re-reads the SAME weight tile every
///    iteration. One 5120x17408 affine-4 projection is about 50 MB with its
///    scales and biases, which is the system-level cache class on this part, so
///    iteration k+1 is served warm. Prefill reuses nothing: 64 layers, each
///    reading its own distinct weights exactly once per chunk, always cold.
///  * **ALLOCATION AND RESIDENCY PRESSURE.** MLX wires buffers into an
///    `MTL::ResidencySet` only up to `wired_limit_`, which defaults to 0
///    (`backend/metal/allocator.h:82`, `backend/metal/resident.cpp:29-37`), so
///    everything beyond it is unwired and subject to the system's own paging. A
///    process holding 15 GB of weights is in a different regime from one holding
///    50 MB, whatever it reuses.
///  * **CLOCK OR POWER STATE.** Distinct from thermal throttling: a fresh
///    process may run its first block at boosted clocks and settle lower
///    afterwards, with no heat involved.
///
/// The three arms separate them. `reuse` allocates one weight set and uses it
/// (low allocation, perfect reuse). `stream` allocates W sets and cycles through
/// them (high allocation, no reuse). `pressure` allocates W sets and uses only
/// the first (high allocation, perfect reuse). Data reuse moves `stream` alone;
/// allocation pressure moves `stream` and `pressure` together; a clock effect
/// moves all three and shows up as an ascending-versus-descending gap with a
/// falling GPU clock.
///
/// PREDICTIONS ARE WRITTEN DOWN BEFORE THE RUN. See
/// `task-PREFILL-INSTR-report.md`, "Part 10". A prediction recorded in advance
/// is what makes this decisive rather than interpretable after the fact.
///
/// This loads no model and no real weights: random tensors at the real
/// projection shapes, driven straight into `quantizedMM`, following
/// `QuantizedMatmulTileBenchTests`. `M` is 1024 at every point, so
/// `qmm_row_tile` returns 32 throughout and the JIT kernel-name cache hazard
/// that probe documents cannot arise here — there is only ever one pipeline.
///
///     MLXFAST_RUN_QMM_WORKING_SET_PROBE=1 \
///     swift test -c release --force-resolved-versions --filter workingSetSweep
///
/// Single-point mode, for `tools/working-set-sweep.sh`, which pays a process per
/// point because position in the process is exactly what is under test:
///
///     MLXFAST_QWEN_WS_POINT=gate_up:stream:8
@Suite(.serialized)
struct QuantizedMatmulWorkingSetTests {

    struct Shape {
        let label: String
        let n: Int
        let k: Int
        /// W values to sweep, capped so held weights stay near 1 GB.
        let widths: [Int]

        /// Bytes of one weight set: 4-bit packed weights plus one fp16 scale and
        /// one fp16 bias per group of 64 along K.
        var weightBytes: Double {
            Double(n) * Double(k) * (4.0 / 8.0 + 4.0 / 64.0)
        }
    }

    /// Three widths so the cache knee falls INSIDE the sweep for at least one of
    /// them. `gate_up` is the real prefill shape but one set already sits at the
    /// cache class, so its knee is at the very first step; `mid` and `small`
    /// have room to show a curve.
    static let shapes = [
        Shape(label: "gate_up", n: 17_408, k: 5_120, widths: [1, 2, 4, 8, 16]),
        Shape(
            label: "mid", n: 4_096, k: 5_120,
            widths: [1, 2, 4, 8, 16, 32, 64]),
        Shape(
            label: "small", n: 2_048, k: 5_120,
            widths: [1, 2, 4, 8, 16, 32, 64, 128]),
    ]

    static let rows = 1_024
    static let groupSize = 64
    static let bits = 4
    /// Distinct activations, cycled so no two timed dispatches are the identical
    /// op on identical inputs. Held constant across arms so activation traffic is
    /// a constant, not a variable.
    static let xVariants = 4
    static let itersPerSample = 8
    static let samples = 9

    enum Arm: String, CaseIterable {
        /// One weight set allocated, one used. The W = 1 control.
        case reuse
        /// W allocated, all W used in rotation. No reuse, high allocation.
        case stream
        /// W allocated, only index 0 used. Perfect reuse, high allocation.
        case pressure
    }

    struct Cell {
        let shape: String
        let arm: Arm
        let width: Int
        let medianSeconds: Double
        let minSeconds: Double
        let spread: Double
        let heldBytes: Double
        let activeBytes: Int
    }

    // MARK: - helpers

    private func quantizedWeights(n: Int, k: Int) -> (MLXArray, MLXArray, MLXArray?) {
        let w = MLXRandom.normal([n, k], scale: 0.02).asType(.bfloat16)
        let q = quantized(w, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
        eval(q.wq, q.scales)
        if let b = q.biases { eval(b) }
        return (q.wq, q.scales, q.biases)
    }

    private func project(
        _ x: MLXArray, _ w: (MLXArray, MLXArray, MLXArray?)
    ) -> MLXArray {
        quantizedMM(
            x, w.0, scales: w.1, biases: w.2, transpose: true,
            groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
    }

    private func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s.count % 2 == 1
            ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// GPU frequency in MHz and package power in watts, or nil when `macmon` is
    /// not reachable. Sampling the clock is what lets a flat clock beside a
    /// falling rate rule the clock hypothesis out BY MEASUREMENT rather than by
    /// argument.
    private func gpuTelemetry() -> (mhz: Double, watts: Double)? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["macmon", "pipe", "-s", "1", "-i", "200"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
            let object = try? JSONSerialization.jsonObject(with: data),
            let top = object as? [String: Any]
        else { return nil }
        let gpu = top["gpu_usage"] as? [Any]
        let mhz = (gpu?.first as? NSNumber)?.doubleValue ?? 0
        let watts = (top["gpu_power"] as? NSNumber)?.doubleValue ?? 0
        return (mhz, watts)
    }

    /// One timed cell. Allocates `width` weight sets, then dispatches according
    /// to the arm.
    private func measure(shape: Shape, arm: Arm, width: Int) -> Cell {
        let held = arm == .reuse ? 1 : width
        var weights: [(MLXArray, MLXArray, MLXArray?)] = []
        weights.reserveCapacity(held)
        for _ in 0 ..< held {
            weights.append(quantizedWeights(n: shape.n, k: shape.k))
        }
        let xs = (0 ..< Self.xVariants).map { _ -> MLXArray in
            let x = MLXRandom.normal([Self.rows, shape.k], scale: 1.0)
                .asType(.bfloat16)
            eval(x)
            return x
        }

        // Warm the pipeline so no timed sample pays JIT compilation. Warm on the
        // LAST weight set so the arm's first timed dispatch is not handed a
        // freshly-touched buffer the other arms would not have.
        for _ in 0 ..< 2 {
            eval(project(xs[0], weights[held - 1]))
        }

        var timings: [Double] = []
        var cursor = 0
        for _ in 0 ..< Self.samples {
            var outs: [MLXArray] = []
            outs.reserveCapacity(Self.itersPerSample)
            let t0 = DispatchTime.now().uptimeNanoseconds
            for i in 0 ..< Self.itersPerSample {
                let w: (MLXArray, MLXArray, MLXArray?)
                switch arm {
                case .reuse: w = weights[0]
                case .pressure: w = weights[0]
                case .stream:
                    w = weights[cursor % held]
                    cursor += 1
                }
                outs.append(project(xs[i % Self.xVariants], w))
            }
            eval(outs)
            let t1 = DispatchTime.now().uptimeNanoseconds
            timings.append(
                Double(t1 - t0) / 1e9 / Double(Self.itersPerSample))
        }

        let med = median(timings)
        return Cell(
            shape: shape.label, arm: arm, width: width,
            medianSeconds: med, minSeconds: timings.min()!,
            spread: (timings.max()! - timings.min()!) / med,
            heldBytes: Double(held) * shape.weightBytes,
            activeBytes: Memory.activeMemory)
    }

    private func tflops(_ shape: Shape, _ seconds: Double) -> Double {
        2.0 * Double(Self.rows) * Double(shape.n) * Double(shape.k)
            / seconds / 1e12
    }

    // MARK: - single point, one process

    /// One cell, one process, one line. `tools/working-set-sweep.sh` drives this.
    ///
    /// This is the AUTHORITATIVE arm. The in-process sweep below is a cheap
    /// screen, and it is confounded by the very position effect under test:
    /// `tools/gemm-point-sweep.sh` exists precisely because a second block in the
    /// same process reads differently from the first.
    @Test("working set single point")
    func workingSetSinglePoint() throws {
        guard let raw = ProcessInfo.processInfo
            .environment["MLXFAST_QWEN_WS_POINT"], !raw.isEmpty
        else { return }
        let parts = raw.split(separator: ":").map(String.init)
        guard parts.count == 3,
            let shape = Self.shapes.first(where: { $0.label == parts[0] }),
            let arm = Arm(rawValue: parts[1]),
            let width = Int(parts[2]), width > 0
        else {
            Issue.record("MLXFAST_QWEN_WS_POINT must be <shape>:<arm>:<width>")
            return
        }
        MLXRandom.seed(0x5EED)
        let before = gpuTelemetry()
        let cell = measure(shape: shape, arm: arm, width: width)
        let after = gpuTelemetry()
        print(
            String(
                format:
                    "WSPOINT\t%@\t%@\t%d\t%.4f\t%.3f\t%.3f\t%.3f\t%.1f"
                    + "\t%.0f\t%.0f",
                cell.shape, cell.arm.rawValue, cell.width,
                1000 * cell.medianSeconds, tflops(shape, cell.medianSeconds),
                cell.heldBytes / 1e9, Double(cell.activeBytes) / 1e9,
                100 * cell.spread,
                before?.mhz ?? -1, after?.mhz ?? -1))
        #expect(cell.medianSeconds > 0, "timed cell produced no time")
    }

    // MARK: - in-process sweep

    @Test("working set sweep")
    func workingSetSweep() throws {
        guard ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMM_WORKING_SET_PROBE"] == "1",
            ProcessInfo.processInfo["MLXFAST_QWEN_WS_POINT"] == nil
        else { return }

        MLXRandom.seed(0x5EED)
        var lines: [String] = []

        for shape in Self.shapes {
            let openTelemetry = gpuTelemetry()
            // Ascending then DESCENDING. A genuine width effect appears in both;
            // a monotone in-process position effect appears as a gap between
            // them. The gap is therefore a MEASUREMENT of the confound, not an
            // assumption about it.
            var ascending: [Arm: [Int: Cell]] = [:]
            var descending: [Arm: [Int: Cell]] = [:]
            for arm in Arm.allCases {
                var up: [Int: Cell] = [:]
                for width in shape.widths {
                    up[width] = measure(shape: shape, arm: arm, width: width)
                }
                ascending[arm] = up
                var down: [Int: Cell] = [:]
                for width in shape.widths.reversed() {
                    down[width] = measure(shape: shape, arm: arm, width: width)
                }
                descending[arm] = down
            }
            let closeTelemetry = gpuTelemetry()

            lines.append("")
            lines.append(
                String(
                    format:
                        "  %@  N=%d K=%d  one weight set %.1f MB  "
                        + "GPU %.0f -> %.0f MHz, %.1f -> %.1f W",
                    shape.label, shape.n, shape.k,
                    shape.weightBytes / 1e6,
                    openTelemetry?.mhz ?? -1, closeTelemetry?.mhz ?? -1,
                    openTelemetry?.watts ?? -1, closeTelemetry?.watts ?? -1))
            lines.append(
                "    arm       W   held_GB    up_ms  up_TF  down_ms down_TF"
                    + "  up/down  vs W=1   spread")
            for arm in Arm.allCases {
                guard let baseline = ascending[arm]?[shape.widths[0]] else {
                    continue
                }
                for width in shape.widths {
                    guard let up = ascending[arm]?[width],
                        let down = descending[arm]?[width]
                    else { continue }
                    lines.append(
                        String(
                            format:
                                "    %-8@ %4d  %7.3f  %7.3f %6.2f  %7.3f %6.2f"
                                + "  %7.3f  %6.3f  %5.1f%%",
                            arm.rawValue as NSString, width,
                            up.heldBytes / 1e9,
                            1000 * up.medianSeconds,
                            tflops(shape, up.medianSeconds),
                            1000 * down.medianSeconds,
                            tflops(shape, down.medianSeconds),
                            up.medianSeconds / down.medianSeconds,
                            baseline.medianSeconds / up.medianSeconds,
                            100 * up.spread))
                    #expect(up.medianSeconds > 0, "timed cell produced no time")
                }
            }
        }

        print(
            """

            [working set probe] M=\(Self.rows), affine-4 group-\(Self.groupSize), \
            median of \(Self.samples) samples x \(Self.itersPerSample) dispatches.
              reuse    = 1 weight set held, 1 used   (low allocation, reuse)
              stream   = W held, all W used cycling  (high allocation, no reuse)
              pressure = W held, only index 0 used   (high allocation, reuse)
              'vs W=1' > 1 means FASTER than the W=1 control.
              An up/down ratio far from 1.0 is the in-process position effect \
            `tools/gemm-point-sweep.sh` documents; when it is large, believe the \
            one-point-per-process driver instead of this table.
            """)
        for line in lines { print(line) }
        print("")
    }
}

extension ProcessInfo {
    fileprivate subscript(key: String) -> String? { environment[key] }
}
