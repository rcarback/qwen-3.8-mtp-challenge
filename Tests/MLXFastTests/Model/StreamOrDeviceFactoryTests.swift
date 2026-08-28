import Foundation
import MLX
import Testing

// StreamOrDevice.stream(_:) used to discard its argument and return the calling
// thread's DEFAULT device stream (the GPU here), which made an explicitly
// requested CPU stream silently land back on the GPU. These tests pin both
// halves of the repair: the wrapper carries the stream it was handed, and an op
// dispatched through that wrapper really executes on that stream's device.
@Suite(.serialized)
struct StreamOrDeviceFactoryTests {

    // Identity: the wrapper must expose the stream it was constructed from, not
    // the default one.
    @Test
    func streamFactoryWrapsTheStreamItWasGiven() {
        let cpuStream = Stream(Device.cpu)
        let wrapped = StreamOrDevice.stream(cpuStream)

        #expect(wrapped.stream == cpuStream)
        #expect(wrapped.description == cpuStream.description)
        #expect(wrapped.description.contains("cpu"))
        #expect(wrapped.description != StreamOrDevice.default.description)
    }

    // Placement: `Load` has a CPU implementation and no Metal one
    // (mlx/backend/metal/primitives.cpp: "[Load::eval_gpu] Not implemented."),
    // so where a load evaluates is directly observable through MLX's error
    // handler. A load dispatched on an explicit CPU stream must succeed; the
    // same load on the GPU stream must raise. That pair distinguishes real
    // device dispatch from a wrapper that merely says "cpu".
    @Test
    func opDispatchedOnAnExplicitCPUStreamExecutesOnTheCPU() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("stream-factory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("values.npy")
        try MLX.save(array: MLXArray([1, 2, 3, 4] as [Float]), url: url)

        let cpuStream = Stream(Device.cpu)
        let loaded = try withError { error in
            let array = try MLX.loadArray(url: url, stream: .stream(cpuStream))
            array.eval()
            try error.check()
            return array.asArray(Float.self)
        }
        #expect(loaded == [1, 2, 3, 4])

        // Negative control: the same load on the GPU stream has no
        // implementation, proving the CPU-stream success above was not the GPU
        // quietly doing the work.
        var gpuFailure: String?
        do {
            _ = try withError {
                let array = try MLX.loadArray(url: url, stream: .gpu)
                array.eval()
                return array.shape
            }
        } catch {
            gpuFailure = String(describing: error)
        }
        #expect(gpuFailure?.contains("eval_gpu") == true)
    }
}
