import Foundation
import IOSurface
import Metal
import MLX

/// GPU cast, transpose, and zero padding into a persistent ANE input surface.
final class ANESurfacePacking {
    enum PackingError: Error, CustomStringConvertible {
        case unavailable(String)
        var description: String {
            switch self {
            case let .unavailable(message): return "ANE input packing: \(message)"
            }
        }
    }

    private final class Context: @unchecked Sendable {
        let device: any MTLDevice
        let queue: any MTLCommandQueue
        let pipelines: [DType: any MTLComputePipelineState]

        init() throws {
            guard let device = MTLCreateSystemDefaultDevice(),
                  let queue = device.makeCommandQueue() else {
                throw PackingError.unavailable("Metal device or command queue is unavailable")
            }
            self.device = device
            self.queue = queue
            let body = """
            #include <metal_stdlib>
            using namespace metal;
            template <typename T>
            inline half convert_input(T value) { return half(value); }
            template <>
            inline half convert_input<ushort>(ushort value) {
                return half(as_type<float>(uint(value) << 16));
            }
            template <typename T>
            inline void pack(device const T* x, device half* y,
                             constant uint3& dims, uint index) {
                uint channel = index / dims.z;
                uint row = index % dims.z;
                if (channel < dims.x) {
                    y[index] = row < dims.y
                        ? convert_input<T>(x[row * dims.x + channel]) : half(0);
                }
            }
            kernel void pack_f16(device const half* x [[buffer(0)]],
                                 device half* y [[buffer(1)]],
                                 constant uint3& dims [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) { pack(x, y, dims, i); }
            kernel void pack_bf16(device const ushort* x [[buffer(0)]],
                                  device half* y [[buffer(1)]],
                                  constant uint3& dims [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) { pack(x, y, dims, i); }
            kernel void pack_f32(device const float* x [[buffer(0)]],
                                 device half* y [[buffer(1)]],
                                 constant uint3& dims [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) { pack(x, y, dims, i); }
            """
            let options = MTLCompileOptions()
            if #available(macOS 15.0, iOS 18.0, *) {
                options.mathMode = .safe
                options.mathFloatingPointFunctions = .precise
            } else {
                options.fastMathEnabled = false
            }
            let library = try device.makeLibrary(source: body, options: options)
            var pipelines: [DType: any MTLComputePipelineState] = [:]
            for (dtype, name) in [(DType.float16, "pack_f16"), (.bfloat16, "pack_bf16"), (.float32, "pack_f32")] {
                guard let function = library.makeFunction(name: name) else {
                    throw PackingError.unavailable("missing kernel \(name)")
                }
                pipelines[dtype] = try device.makeComputePipelineState(function: function)
            }
            self.pipelines = pipelines
        }
    }

    /// Retains source bytes until the dedicated queue completes. Only wait()
    /// runs on the ANE thread; no MLX operations run there.
    final class Transfer: @unchecked Sendable {
        private let command: any MTLCommandBuffer
        private let source: MLXArray
        private let sourceBuffer: any MTLBuffer
        init(command: any MTLCommandBuffer, source: MLXArray, sourceBuffer: any MTLBuffer) {
            self.command = command
            self.source = source
            self.sourceBuffer = sourceBuffer
        }
        func wait() throws {
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw PackingError.unavailable(command.error?.localizedDescription ?? "Metal input command did not complete")
            }
        }
    }

    private static let shared = Result { try Context() }
    private final class BufferOwners: @unchecked Sendable {
        let source: MLXArray
        let surface: IOSurface
        init(source: MLXArray, surface: IOSurface) {
            self.source = source
            self.surface = surface
        }
    }
    private let context: Context
    private let surface: IOSurface
    private let destination: any MTLBuffer

    init(surface: IOSurface) throws {
        context = try Self.shared.get()
        self.surface = surface
        guard let buffer = context.device.makeBuffer(bytesNoCopy: surface.baseAddress,
            length: IOSurfaceGetAllocSize(surface), options: .storageModeShared, deallocator: nil) else {
            throw PackingError.unavailable("cannot wrap the input IOSurface")
        }
        destination = buffer
    }

    func submit(_ x: MLXArray, width: Int, rows: Int, stride: Int) throws -> Transfer {
        guard x.shape == [rows, width], rows <= stride,
              let pipeline = context.pipelines[x.dtype] else {
            throw PackingError.unavailable("unsupported input \(x.shape) \(x.dtype), expected [\(rows), \(width)]")
        }
        let source = x.contiguous().reshaped([-1])
        // MLX eval completes the producer before a separate Metal queue reads it.
        eval(source)
        guard let buffer = source.asMTLBuffer(device: context.device, noCopy: true),
              let command = context.queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw PackingError.unavailable("cannot encode the input transfer")
        }
        var dims = SIMD3<UInt32>(UInt32(width), UInt32(rows), UInt32(stride))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        encoder.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        encoder.dispatchThreads(MTLSize(width: width * stride, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        // The no-copy Metal buffers do not own their backing allocations.
        // Keep both owners through completion even if Transfer is discarded.
        let owners = BufferOwners(source: source, surface: surface)
        command.addCompletedHandler { [owners] _ in withExtendedLifetime(owners) {} }
        command.commit()
        return Transfer(command: command, source: source, sourceBuffer: buffer)
    }
}
