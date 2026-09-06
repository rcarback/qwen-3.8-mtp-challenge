// A duty-cycled GPU load that stands in for UI or another GPU client during
// a measurement: dispatches the `fma_burn` kernel for a share of every
// period and sleeps for the rest, for `duration` seconds.
//
// Build (from tools/ane-probes):
//   xcrun -sdk macosx metal -c ../cpu-sme-lane/gpu_matmul.metal -o gpu_matmul.air
//   xcrun -sdk macosx metallib gpu_matmul.air -o gpu_matmul.metallib
//   swiftc -O gpu_load.swift -o gpu_load -framework Metal -framework Foundation
// Usage:
//   ./gpu_load <duration_s> <duty_percent> <period_ms>
import Foundation
import Metal

let args = CommandLine.arguments
guard args.count >= 4, let durationSec = Double(args[1]), let duty = Double(args[2]), let periodMs = Double(args[3]) else {
    FileHandle.standardError.write("usage: gpu_load <duration_s> <duty_percent> <period_ms>\n".data(using: .utf8)!)
    exit(1)
}
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let exeDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
let libURL = exeDir.appendingPathComponent("gpu_matmul.metallib")
guard let library = try? device.makeLibrary(URL: libURL) else { fatalError("gpu_matmul.metallib must sit beside gpu_load") }
guard let fn = library.makeFunction(name: "fma_burn"),
      let pipeline = try? device.makeComputePipelineState(function: fn),
      let queue = device.makeCommandQueue()
else { fatalError("pipeline") }

// One dispatch sized to a few milliseconds of GPU work on an M4 Max.
let threads = 1 << 20
var iters: UInt32 = 256
let start = Date()
var busyTotal = 0.0
var dispatches = 0
while Date().timeIntervalSince(start) < durationSec {
    let periodStart = Date()
    let busyBudget = periodMs / 1000.0 * duty / 100.0
    while Date().timeIntervalSince(periodStart) < busyBudget {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
        enc.setComputePipelineState(pipeline)
        enc.setBytes(&iters, length: MemoryLayout<UInt32>.size, index: 0)
        enc.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        dispatches += 1
    }
    busyTotal += Date().timeIntervalSince(periodStart)
    let rest = periodMs / 1000.0 - Date().timeIntervalSince(periodStart)
    if rest > 0 { Thread.sleep(forTimeInterval: rest) }
}
let elapsed = Date().timeIntervalSince(start)
print("gpu_load done: \(dispatches) dispatches, busy \(String(format: "%.0f", busyTotal / elapsed * 100))% of \(String(format: "%.0f", elapsed)) s at duty \(Int(duty))%")
