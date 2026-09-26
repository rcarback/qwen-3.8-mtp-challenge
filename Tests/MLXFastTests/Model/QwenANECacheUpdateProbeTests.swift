import Foundation
import MLX
import Testing
@testable import MLXLLM

@Suite(.serialized)
struct QwenANECacheUpdateProbeTests {
    @Test func invalidShapes() {
        for (batch, capacity) in [(0, 32), (33, 32), (32, 0), (32, 1025)] {
            #expect(throws: QwenANECacheUpdateError.self) {
                try buildQwenANECacheUpdateMILText(batch: batch, capacity: capacity)
            }
        }
    }

    @Test func independentLaneUpdates() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXFAST_ANE_CACHE_UPDATE_PROBE"] == "1" else { return }
        let root = URL(fileURLWithPath: try #require(env["MLXFAST_ANE_BATCH_ROOT"]))
            .appendingPathComponent("ane-batch")
        let batch = 32
        let capacity = try #require(Int(env["MLXFAST_ANE_CACHE_CAPACITY"] ?? "32"))
        try #require((1...1024).contains(capacity))
        let channels = 2048 * capacity
        let stem = "cache-update-spatial-\(capacity)"
        let text = try buildQwenANECacheUpdateMILText(batch: batch, capacity: capacity)
        try text.write(to: root.appendingPathComponent("\(stem).mil"), atomically: true, encoding: .utf8)
        var report: [String: Any] = ["phase": "compile", "lanes": batch, "capacity": capacity,
            "scope": "ANE cache overwrite with independent slots; host buffer packing and exact CPU oracle; not yet fused with attention"]
        func save() throws {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("\(stem).json"), options: .atomic)
        }
        try save()
        do {
            let model = try ANEInMemoryModel(milText: text, weightBlob: Data())
            try model.compile(); report["phase"] = "load"; try save()
            try model.load(); defer { model.unload() }
            var cache = (0..<(batch * channels)).map { Float16(Float($0 % 101 - 50) / 16) }
            var oracle = cache
            var trials: [[String: Any]] = []
            for step in 0..<5 {
                let slots = (0..<batch).map { lane in
                    step == 4 && lane % 2 == 0 ? -1 : (step == 2 ? capacity - 1 : (lane + step * 7) % capacity)
                }
                try #require(slots.allSatisfy { $0 >= -1 && $0 < capacity })
                var input: [Float16] = []
                input.reserveCapacity(batch * (channels + 2304))
                for lane in 0..<batch {
                    input.append(contentsOf: cache[(lane * channels)..<((lane + 1) * channels)])
                    let fresh: [Float16] = (0..<2048).map { channel in
                        Float16(Float((channel + lane * 13 + step * 17) % 127 - 63) / 8)
                    }
                    input.append(contentsOf: fresh)
                    input.append(Float16(slots[lane]))
                    input.append(contentsOf: repeatElement(Float16(0), count: 255))
                    if slots[lane] >= 0 {
                        for head in 0..<8 {
                            let start = lane * channels + (head * capacity + slots[lane]) * 256
                            oracle.replaceSubrange(start..<(start + 256), with: fresh[(head * 256)..<((head + 1) * 256)])
                        }
                    }
                }
                report["phase"] = "dispatch"; try save()
                let began = Date()
                let output = try ANEDirectDispatch.runConv(model: model,
                    x: MLXArray(input, [batch, channels + 2304]), inputDim: channels + 2304,
                    outputDim: channels, sequenceLength: batch)
                cache = output.asArray(Float16.self)
                let elapsed = Date().timeIntervalSince(began)
                try #require(cache.count == oracle.count && cache.allSatisfy { $0.isFinite })
                let mismatches = zip(cache, oracle).filter { $0 != $1 }.count
                trials.append(["step": step, "slots": slots, "mismatched_values": mismatches,
                               "dispatch_and_read_seconds": elapsed])
                report["trials"] = trials; try save()
                try #require(mismatches == 0)
            }
            report["phase"] = "complete"; try save()
        } catch {
            report["error"] = String(describing: error); try save(); throw error
        }
    }
}
