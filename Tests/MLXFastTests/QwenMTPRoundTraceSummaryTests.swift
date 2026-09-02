import Foundation
import Testing

/// The summary script is the reader of every trace this plan produces, so
/// its arithmetic is pinned here on a fixture with hand-computed medians.
struct QwenMTPRoundTraceSummaryTests {
    private var repositoryURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static let parentLines = """
        mtp-cache: layers=64 first4=MambaCache,MambaCache,MambaCache,KVCacheSimple kv_policy=bf16 ladder=default compiled_decode=true
        mtp-parent: round=1 offered=0 encode_write_us=100 wait_us=80500 decode_us=150 round_us=81000 gap_us=0 bytes=250 t0_ns=1 t1_ns=2
        mtp-parent: round=2 offered=0 encode_write_us=120 wait_us=81400 decode_us=160 round_us=82000 gap_us=300 bytes=250 t0_ns=3 t1_ns=4
        mtp-parent: round=3 offered=0 encode_write_us=110 wait_us=79500 decode_us=140 round_us=80000 gap_us=250 bytes=250 t0_ns=5 t1_ns=6
        mtp-parent: round=1 offered=8 encode_write_us=999 wait_us=999999 decode_us=999 round_us=999999 gap_us=0 bytes=900 t0_ns=7 t1_ns=8

        """

    private static let sessionLines = """
        mtp-trace0: round=1 offered=0 build_us=3000 eval_wall_us=74000 readout_us=200 round_us=77200 host_thread_cpu_ns=2500000 t0_ns=1 t_eval_done_ns=2
        mtp-trace0: round=2 offered=0 build_us=3200 eval_wall_us=75000 readout_us=220 round_us=78420 host_thread_cpu_ns=2700000 t0_ns=3 t_eval_done_ns=4
        mtp-trace0: round=3 offered=0 build_us=2800 eval_wall_us=73000 readout_us=210 round_us=76010 host_thread_cpu_ns=2400000 t0_ns=5 t_eval_done_ns=6

        """

    /// Worker lines whose `worker_us` sits below the parent's `wait_us`, so
    /// every derived bucket is positive.
    private static let consistentWorkerLines = """
        mtp-worker: id=2 offered=0 decode_us=50 handle_us=79000 encode_us=80 write_us=30 worker_us=79160 bytes=250 t_read_ns=1 t_written_ns=2
        mtp-worker: id=3 offered=0 decode_us=60 handle_us=80000 encode_us=90 write_us=40 worker_us=80190 bytes=250 t_read_ns=3 t_written_ns=4
        mtp-worker: id=4 offered=0 decode_us=55 handle_us=78000 encode_us=85 write_us=35 worker_us=78175 bytes=250 t_read_ns=5 t_written_ns=6

        """

    /// Worker lines from a different run: `worker_us` above the parent's
    /// `wait_us`, so the derived transport bucket goes negative.
    private static let mismatchedWorkerLines = """
        mtp-worker: id=2 offered=0 decode_us=50 handle_us=90000 encode_us=80 write_us=30 worker_us=90160 bytes=250 t_read_ns=1 t_written_ns=2
        mtp-worker: id=3 offered=0 decode_us=60 handle_us=91000 encode_us=90 write_us=40 worker_us=91190 bytes=250 t_read_ns=3 t_written_ns=4
        mtp-worker: id=4 offered=0 decode_us=55 handle_us=89000 encode_us=85 write_us=35 worker_us=89175 bytes=250 t_read_ns=5 t_written_ns=6

        """

    private func runSummary(fixture: String) throws -> (status: Int32, output: String) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-round-trace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let traceURL = directory.appendingPathComponent("trace.log")
        try fixture.write(to: traceURL, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            repositoryURL.appendingPathComponent("tools/mtp-round-trace-summary.sh").path,
            traceURL.path,
        ]
        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = stdout
        try process.run()
        let output = String(
            decoding: stdout.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self)
        process.waitUntilExit()
        return (process.terminationStatus, output)
    }

    private func value(_ label: String, in output: String) -> Int? {
        output.split(separator: "\n")
            .first { $0.hasPrefix(label) }
            .flatMap { line in
                line.dropFirst(label.count).split(separator: " ")
                    .compactMap { Int($0) }.first
            }
    }

    @Test("summary script computes lower medians and derived buckets")
    func summaryBuckets() throws {
        let result = try runSummary(
            fixture: Self.parentLines + Self.consistentWorkerLines + Self.sessionLines)
        let output = result.output
        #expect(result.status == 0, "script failed: \(output)")
        #expect(output.hasPrefix("rounds: parent=3 worker=3 session=3"))
        #expect(value("session eval wall", in: output) == 74000)
        #expect(value("session graph build", in: output) == 3000)
        #expect(value("parent round (measured)", in: output) == 81000)
        #expect(value("transport + scheduling (derived)", in: output) == 1340)
        #expect(value("worker outside the session (derived)", in: output) == 1800)
        #expect(value("parent closure", in: output) == 240)
        #expect(value("session build+eval+readout", in: output) == 77210)
        #expect(value("forward excess", in: output) == 34610)
        #expect(value("parent gap between rounds", in: output) == 250)
        #expect(value("session host thread cpu", in: output) == 2500)
    }

    @Test("a negative derived bucket fails the script after printing the table")
    func negativeDerivedBucketFails() throws {
        let result = try runSummary(
            fixture: Self.parentLines + Self.mismatchedWorkerLines + Self.sessionLines)
        #expect(result.status == 1)
        #expect(result.output.contains("FAIL: derived bucket transport is negative"))
        #expect(value("transport + scheduling (derived)", in: result.output) == -9660)
        #expect(value("parent round (measured)", in: result.output) == 81000)
    }
}
