import Foundation
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANECacheAttentionProbeTests {
    @Test func attentionUsesRetainedUpdatedCache() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_CACHE_ATTENTION_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let batch = 32, capacity = 1024, channels = capacity * 2048
        var report: [String: Any] = ["phase": "initialize", "lanes": batch, "capacity": capacity,
            "scope": "Separate ANE cache update and masked GQA programs sharing retained cache surfaces; synthetic independent histories and CPU oracle"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("cache-attention.json"), options: .atomic)
        }
        try save()
        do {
            let loop = try ANEDirectDispatch.CacheLoop(batch: batch, capacity: capacity, attention: true)
            var oracle = [Float16](repeating: 0, count: batch * channels)
            var trials: [[String: Any]] = []
            for step in 0..<3 {
                let slots = (0..<batch).map { lane in step == 2 && lane % 2 == 0 ? -1 : lane % 4 + step }
                let lengths = (0..<batch).map { lane in lane % 4 + (slots[lane] == -1 ? step : step + 1) }
                var fresh = [Float16](repeating: 0, count: batch * 2048)
                var queries = [Float16](repeating: 0, count: batch * 6144)
                for lane in 0..<batch {
                    for head in 0..<8 {
                        let scalar = Float16(Float((lane + head * 3 + step * 5) % 17 - 8) / 16)
                        for channel in 0..<256 {
                            fresh[lane * 2048 + head * 256 + channel] = scalar
                            if slots[lane] >= 0 {
                                oracle[lane * channels + (head * capacity + slots[lane]) * 256 + channel] = scalar
                            }
                        }
                    }
                    for head in 0..<24 {
                        let scalar = Float16(Float((lane + head + step) % 11 - 5) / 32)
                        for channel in 0..<256 { queries[lane * 6144 + head * 256 + channel] = scalar }
                    }
                }
                report["phase"] = "dispatch"; try save()
                let began = Date()
                let actual = try loop.step(newKV: fresh, slots: slots, queries: queries, validLengths: lengths)
                let elapsed = Date().timeIntervalSince(began)
                try #require(actual.count == batch * 6144 && actual.allSatisfy { $0.isFinite })
                var maximum = 0.0
                for lane in 0..<batch {
                    for head in 0..<24 {
                        let kv = head / 6
                        let query = Double(queries[lane * 6144 + head * 256])
                        var scores: [Double] = []
                        for slot in 0..<lengths[lane] {
                            let key = Double(oracle[lane * channels + (kv * capacity + slot) * 256])
                            scores.append(16 * query * key)
                        }
                        let peak = try #require(scores.max())
                        let probabilities = scores.map { exp($0 - peak) }
                        let total = probabilities.reduce(0, +)
                        var expected = 0.0
                        for slot in 0..<lengths[lane] {
                            let value = Double(oracle[lane * channels + ((kv + 4) * capacity + slot) * 256])
                            expected += probabilities[slot] / total * value
                        }
                        for channel in 0..<256 {
                            maximum = max(maximum, abs(Double(actual[lane * 6144 + head * 256 + channel]) - expected))
                        }
                    }
                }
                trials.append(["step": step, "slots": slots, "valid_lengths": lengths,
                    "stage_dispatch_attention_read_seconds": elapsed, "max_abs_error": maximum])
                report["trials"] = trials; try save()
                try #require(maximum < 0.005)
            }
            report["phase"] = "audit"; try save()
            let cache = try loop.snapshot()
            try #require(cache.count == oracle.count)
            let mismatches = zip(cache, oracle).filter { $0 != $1 }.count
            report["cache_mismatched_values"] = mismatches; try save()
            try #require(mismatches == 0)
            report["phase"] = "complete"; try save()
        } catch {
            report["error"] = String(describing: error); try save(); throw error
        }
    }
}
