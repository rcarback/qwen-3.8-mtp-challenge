// GPU compute-throughput driver for the CPU-SME/GPU starvation test.
//
// Dispatches the `fma_burn` compute kernel (gpu_matmul.metal) in a tight
// loop for a fixed wall-clock duration, encoding a fresh command buffer per
// dispatch (so host-side (CPU) command-buffer build/dispatch is exercised
// every iteration -- that CPU-side work is exactly what a concurrent CPU-SME
// lane could starve). Reports achieved GFLOPS and completed dispatch count.
//
// Build:
//   xcrun -sdk macosx metal -c gpu_matmul.metal -o gpu_matmul.air
//   xcrun -sdk macosx metallib gpu_matmul.air -o gpu_matmul.metallib
//   swiftc -O gpu_burn.swift -o gpu_burn -framework Metal -framework Foundation
//
// Usage:
//   ./gpu_burn <duration_seconds> <threadsPerGrid> <itersPerThread>

import Foundation
import Metal

let args = CommandLine.arguments
guard args.count >= 4,
      let durationSec = Double(args[1]),
      let threadCount = Int(args[2]),
      let itersPerThread = UInt32(args[3])
else {
    FileHandle.standardError.write("usage: gpu_burn <duration_sec> <threads> <itersPerThread>\n".data(using: .utf8)!)
    exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("no Metal device")
}
let libURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("gpu_matmul.metallib")
guard let library = try? device.makeLibrary(URL: libURL) else {
    fatalError("failed to load gpu_matmul.metallib next to the binary (run from tools/cpu-sme-lane)")
}
guard let fn = library.makeFunction(name: "fma_burn") else { fatalError("missing fma_burn") }
guard let pipeline = try? device.makeComputePipelineState(function: fn) else { fatalError("pipeline creation failed") }
guard let queue = device.makeCommandQueue() else { fatalError("no command queue") }

let outBuf = device.makeBuffer(length: threadCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
var itersLocal = itersPerThread
let itersBuf = device.makeBuffer(bytes: &itersLocal, length: MemoryLayout<UInt32>.size, options: .storageModeShared)!

let w = pipeline.threadExecutionWidth
let threadsPerTG = MTLSize(width: min(w, threadCount), height: 1, depth: 1)
let grid = MTLSize(width: threadCount, height: 1, depth: 1)

// FLOPs per thread: 4 fma's/iter * 4 lanes * 2 flops(mul+add) = 32 flops/iter
let flopsPerThreadPerIter = 32.0
let flopsPerDispatch = flopsPerThreadPerIter * Double(itersPerThread) * Double(threadCount)

let start = Date()
var dispatches = 0
while Date().timeIntervalSince(start) < durationSec {
    guard let cmdBuf = queue.makeCommandBuffer(),
          let enc = cmdBuf.makeComputeCommandEncoder() else { fatalError("encoder failed") }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(outBuf, offset: 0, index: 0)
    enc.setBuffer(itersBuf, offset: 0, index: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: threadsPerTG)
    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()
    dispatches += 1
}
let elapsed = Date().timeIntervalSince(start)
let totalFlops = flopsPerDispatch * Double(dispatches)
let gflops = totalFlops / elapsed / 1e9
print("gpu_burn dispatches=\(dispatches) elapsed=\(String(format: "%.4f", elapsed))s GFLOPS=\(String(format: "%.2f", gflops)) TFLOPS=\(String(format: "%.4f", gflops / 1000.0)) dispatch_rate_hz=\(String(format: "%.1f", Double(dispatches) / elapsed))")
