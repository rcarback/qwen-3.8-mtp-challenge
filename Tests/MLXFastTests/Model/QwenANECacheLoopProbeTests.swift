import Foundation
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANECacheLoopProbeTests {
    @Test func retainedCacheAcrossUpdates() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_CACHE_LOOP_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let batch = 32, capacity = 1024, channels = capacity * 2048
        var result: [String: Any] = ["phase": "initialize", "lanes": batch, "capacity": capacity,
            "scope": "ANE cache recurrence through retained IOSurfaces; CPU stages new KV and slots only; final snapshot is a test oracle"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("cache-loop.json"), options: .atomic)
        }
        try save()
        do {
            let loop = try ANEDirectDispatch.CacheLoop(batch: batch, capacity: capacity)
            #expect(throws: ANEDirectDispatch.CacheLoopError.self) {
                try loop.step(newKV: [], slots: [])
            }
            #expect(throws: ANEDirectDispatch.CacheLoopError.self) {
                try loop.step(newKV: Array(repeating: 0, count: batch * 2048), slots: Array(repeating: capacity, count: batch))
            }
            var oracle = [Float16](repeating: 0, count: batch * channels)
            var trials: [[String: Any]] = []
            for step in 0..<5 {
                let slots = (0..<batch).map { lane in
                    step == 4 && lane % 2 == 0 ? -1 : (step == 2 ? capacity - 1 : (lane + (step % 3) * 7) % capacity)
                }
                let fresh: [Float16] = (0..<(batch * 2048)).map { index in
                    Float16(Float((index + step * 17) % 127 - 63) / 8)
                }
                for lane in 0..<batch where slots[lane] >= 0 {
                    for head in 0..<8 {
                        let destination = lane * channels + (head * capacity + slots[lane]) * 256
                        let source = lane * 2048 + head * 256
                        oracle.replaceSubrange(destination..<(destination + 256), with: fresh[source..<(source + 256)])
                    }
                }
                result["phase"] = "dispatch"; try save()
                let began = Date()
                try loop.step(newKV: fresh, slots: slots)
                trials.append(["step": step, "slots": slots, "stage_and_dispatch_seconds": Date().timeIntervalSince(began)])
                result["trials"] = trials; try save()
            }
            result["phase"] = "audit"; try save()
            let actual = try loop.snapshot()
            try #require(actual.count == oracle.count && actual.allSatisfy { $0.isFinite })
            let mismatches = zip(actual, oracle).filter { $0 != $1 }.count
            result["mismatched_values"] = mismatches
            result["audited_values"] = actual.count
            try save()
            try #require(mismatches == 0)
            result["phase"] = "complete"; try save()
        } catch {
            result["error"] = String(describing: error); try save(); throw error
        }
    }
}
