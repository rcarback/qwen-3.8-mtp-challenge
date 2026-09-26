// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLXFAST_ANE_DIRECT=1 at call sites.
//
// Executes a compiled+loaded `ANEInMemoryModel` program on the Neural
// Engine via CPU-side IOSurface I/O and a blocking `_ANERequest` evaluate --
// the step after Task 2b's compile+load, proving the loaded program
// actually runs and returns numerically correct output. Ports oMLX's
// `make_surface` / `AneLinearModel::Impl` / evaluate path
// (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`), simplified to
// CPU-side surface I/O and a blocking evaluate -- the Metal shared-event
// GPU/ANE overlap oMLX layers on top is a separate, later performance task.
// See `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-A-brief.md`.
import Foundation
import IOSurface
import MLX
import ObjectiveC

public enum ANEDirectDispatch {
    enum ANEDispatchError: Error, CustomStringConvertible {
        case surfaceAllocationFailed
        case wrapFailed
        case requestFailed
        case evaluateFailed(String)
        case requestBusy

        var description: String {
            switch self {
            case .surfaceAllocationFailed: return "ANEDirectDispatch: IOSurface allocation failed"
            case .wrapFailed: return "ANEDirectDispatch: _ANEIOSurfaceObject wrap failed"
            case .requestFailed: return "ANEDirectDispatch: _ANERequest construction failed"
            case let .evaluateFailed(message): return "ANEDirectDispatch: evaluate failed: \(message)"
            case .requestBusy: return "ANEDirectDispatch: finish the current request before reusing its surfaces"
            }
        }
    }

    /// `objc_msgSend` cast for the 7-payload-arg class factory
    /// `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`
    /// -- not covered by `ANERuntime.send`'s 0/1/3-object-arg overloads.
    private typealias RequestFactory = @convention(c) (
        AnyObject?, Selector, AnyObject?, AnyObject?, AnyObject?, AnyObject?, AnyObject?, AnyObject?, AnyObject?
    ) -> Unmanaged<AnyObject>?

    /// `objc_msgSend` cast for `evaluateWithQoS:options:request:error:` ->
    /// `BOOL(NSInteger,id,id,NSError**)`. Same `Unmanaged<NSError>?`
    /// error-param treatment as `ANEInMemoryModel.SendBoolQoSErr` -- a plain
    /// `UnsafeMutablePointer<NSError?>` would bypass the ARC-aware bridging
    /// that balances the callee's autoreleased out-param write.
    private typealias EvaluateFn = @convention(c) (
        AnyObject?, Selector, Int, AnyObject?, AnyObject?, UnsafeMutablePointer<Unmanaged<NSError>?>?
    ) -> ObjCBool

    /// oMLX `make_surface`: a single-page-aligned (64K) `IOSurface`, at
    /// least 64K even for a smaller payload. Only the first `byteCount`
    /// bytes are ever read or written; the rest is alignment padding the
    /// conv op never touches.
    private static func makeSurface(byteCount: Int) -> IOSurface? {
        let alloc = max(65536, (byteCount + 65535) & ~65535)
        // `NSDictionary`/`CFDictionary` (not the Swift-native
        // `IOSurface(properties:)` overload, whose `[IOSurfacePropertyKey:
        // Any]` parameter trips a Sendable warning under this package's
        // concurrency checking) -- same property keys oMLX's `make_surface`
        // uses.
        let props: NSDictionary = [
            kIOSurfaceWidth: alloc,
            kIOSurfaceHeight: 1,
            kIOSurfaceBytesPerElement: 1,
            kIOSurfaceBytesPerRow: alloc,
            kIOSurfaceAllocSize: alloc,
            kIOSurfacePixelFormat: 0,
        ]
        return IOSurfaceCreate(props as CFDictionary)
    }

    /// Persistent request storage. The ownership lock protects state changes
    /// between caller-side staging/readback and background ANE evaluation.
    /// Only the caller thread accesses pending MLX graphs. Reuse completes
    /// their reads before either surface can be overwritten.
    fileprivate final class Storage: @unchecked Sendable {
        fileprivate enum State { case idle, staging, staged, evaluating, complete, read }
        fileprivate let ownership = NSLock()
        fileprivate var state = State.idle
        // MLX eval materializes array graphs; it does not evaluate source code.
        // Complete these independent outputs before ANE overwrites their source.
        fileprivate var pendingOutputs: [MLXArray] = []
        var generation = 0
        var packer: ANESurfacePacking?
        var inputTransfer: ANESurfacePacking.Transfer?
        let model: ANEInMemoryModel
        let request: AnyObject
        // The input surface is held even though `read` never touches it: the
        // background `evaluate` reads its bytes, so the handle keeps its own
        // strong reference to every surface it needs across the thread
        // boundary rather than depending on the implicit `request` ->
        // `_ANEIOSurfaceObject` retain chain to keep the input alive.
        let inputSurface: IOSurface
        let outputSurface: IOSurface
        let outputDim: Int
        let sequenceLength: Int
        let rowStride: Int

        init(model: ANEInMemoryModel, request: AnyObject, inputSurface: IOSurface, outputSurface: IOSurface, outputDim: Int, sequenceLength: Int, rowStride: Int) {
            self.model = model
            self.request = request
            self.inputSurface = inputSurface
            self.outputSurface = outputSurface
            self.outputDim = outputDim
            self.sequenceLength = sequenceLength
            self.rowStride = rowStride
        }
    }

    /// A single invocation of reusable storage. Old handles cannot execute
    /// or read a later invocation, even though the request identity is stable.
    public final class Prepared: @unchecked Sendable {
        fileprivate let storage: Storage
        fileprivate let generation: Int
        fileprivate init(_ storage: Storage) {
            self.storage = storage
            self.generation = storage.generation
        }
        fileprivate var ownership: NSLock { storage.ownership }
        fileprivate var state: Storage.State {
            get { storage.state }
            set { storage.state = newValue }
        }
        fileprivate var pendingOutputs: [MLXArray] {
            get { storage.pendingOutputs }
            set { storage.pendingOutputs = newValue }
        }
        var model: ANEInMemoryModel { storage.model }
        var request: AnyObject { storage.request }
        var inputSurface: IOSurface { storage.inputSurface }
        var outputSurface: IOSurface { storage.outputSurface }
        var outputDim: Int { storage.outputDim }
        var sequenceLength: Int { storage.sequenceLength }
        var rowStride: Int { storage.rowStride }
    }

    /// CALLER THREAD ONLY (MLX). `x` is `[S, IN]` fp16 (MLX). Allocates the
    /// input/output IOSurfaces, writes `x` into the input surface (MLX
    /// transpose + eval + `asData`), wraps both surfaces, and builds the
    /// `_ANERequest`. Returns a `Prepared` that `evaluate` and `read`
    /// consume -- only `evaluate` may run off this thread.
    /// `procedureIndex` selects which function of a multi-function
    /// (`buildBankMILText`) program this request runs; the default `0` is the
    /// sole function (`main`) of every single-conv program and is unchanged
    /// behaviour for those callers.
    public static func prepare(model: ANEInMemoryModel, x: MLXArray, inputDim: Int, outputDim: Int, sequenceLength: Int, procedureIndex: Int = 0, gpuInput: Bool = false) throws -> Prepared {
        precondition(x.ndim == 2 && x.shape[0] <= sequenceLength && x.shape[1] == inputDim,
                     "ANEDirectDispatch.prepare expected x shape [\(sequenceLength), \(inputDim)], got \(x.shape)")

        // The ANE pads the conv's trailing spatial (sequence) axis to a
        // 32-element tile internally -- confirmed empirically: at S=32
        // (already 32-aligned) a tightly-packed [IN,S]/[OUT,S] transfer
        // matched `matmul` exactly; at S=1, tightly packed on either side
        // produced plausible-looking but wrong output (per-channel rows
        // landing at the wrong byte offset once the real per-channel stride
        // turned out to be the padded one), while padding only the output
        // side left S=1 equally wrong -- so the padded 32-element row
        // stride applies to BOTH the conv's input and output tensors, not
        // just the Core ML `MLMultiArray` prediction path
        // `ANEMILBuilder.multiArray_1C1S_toMLX` documents. A within-row
        // position (e.g. position 0) is unaffected by what garbage sits in
        // the unused padding lanes -- a 1x1 conv never mixes across the
        // spatial axis -- so writing/reading only the real `S` elements at
        // the padded stride is sufficient; the padding itself is never
        // read back.
        let rowStride = ((sequenceLength + 31) / 32) * 32
        let inputSurfaceByteCount = inputDim * rowStride * 2
        let outputSurfaceByteCount = outputDim * rowStride * 2
        guard let inputSurface = makeSurface(byteCount: inputSurfaceByteCount),
              let outputSurface = makeSurface(byteCount: outputSurfaceByteCount) else {
            throw ANEDispatchError.surfaceAllocationFailed
        }

        let request = try makeRequest(inputSurface: inputSurface, outputSurface: outputSurface,
                                      procedureIndex: procedureIndex)

        let storage = Storage(model: model, request: request, inputSurface: inputSurface, outputSurface: outputSurface,
                         outputDim: outputDim, sequenceLength: sequenceLength, rowStride: rowStride)
        return try restage(Prepared(storage), x: x, inputDim: inputDim, gpuInput: gpuInput)
    }

    private static func makeRequest(inputSurface: IOSurface, outputSurface: IOSurface,
                                    procedureIndex: Int = 0) throws -> AnyObject {
        // `objectWithIOSurface:` is a Cocoa class-factory method (not
        // alloc/init/copy/new), so per Cocoa convention it returns an
        // autoreleased (+0) object -- `ANERuntime.send`'s default
        // `retained: false` (`takeUnretainedValue()`) is correct here; using
        // `retained: true` would claim an ownership count the callee never
        // handed over and over-release at scope exit (the same ARC-balance
        // bug class `ANEInMemoryModel`'s doc comments describe for
        // `compileWithQoS:`/`loadWithQoS:`).
        guard let surfaceClass = ANERuntime.cls("_ANEIOSurfaceObject") else { throw ANEDispatchError.wrapFailed }
        guard let inputObject = ANERuntime.send(surfaceClass, Selector(("objectWithIOSurface:")), inputSurface as AnyObject),
              let outputObject = ANERuntime.send(surfaceClass, Selector(("objectWithIOSurface:")), outputSurface as AnyObject) else {
            throw ANEDispatchError.wrapFailed
        }

        guard let requestClass = ANERuntime.cls("_ANERequest") else { throw ANEDispatchError.requestFailed }
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let requestFactory = unsafeBitCast(msgSend, to: RequestFactory.self)
        let requestSel = Selector(("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"))
        guard let requestU = requestFactory(
            requestClass, requestSel,
            [inputObject] as NSArray, [0] as NSArray,
            [outputObject] as NSArray, [0] as NSArray,
            nil, nil, NSNumber(value: procedureIndex)
        ) else {
            throw ANEDispatchError.requestFailed
        }
        // `requestWithInputs:...:` is also a Cocoa class-factory method, so
        // its return is autoreleased (+0) -- `takeUnretainedValue()`
        // performs the one retain that balance requires, matching
        // `ANERuntime.send`'s default above. `takeRetainedValue()` here
        // would be the same over-release bug as above, one level further
        // down the call.
        let request = requestU.takeUnretainedValue()

        return request
    }

    /// Caller thread only. Drain prior GPU readers before changing shared bytes.
    static func restage(_ prepared: Prepared, x: MLXArray, inputDim: Int, gpuInput: Bool = false) throws -> Prepared {
        precondition(x.ndim == 2 && x.dim(0) <= prepared.sequenceLength && x.dim(1) == inputDim)
        prepared.ownership.lock()
        guard prepared.generation == prepared.storage.generation,
              prepared.state == .idle || prepared.state == .read else {
            prepared.ownership.unlock()
            throw ANEDispatchError.requestBusy
        }
        prepared.state = .staging
        eval(prepared.pendingOutputs)
        prepared.pendingOutputs.removeAll(keepingCapacity: true)
        prepared.storage.generation += 1
        let invocation = Prepared(prepared.storage)
        prepared.ownership.unlock()

        prepared.storage.inputTransfer = nil
        do {
            if gpuInput {
                if prepared.storage.packer == nil {
                    prepared.storage.packer = try ANESurfacePacking(surface: prepared.inputSurface)
                }
                prepared.storage.inputTransfer = try prepared.storage.packer!.submit(
                    x, width: inputDim, rows: x.dim(0), stride: prepared.rowStride)
            } else {
                let xT = contiguous(x.transposed(1, 0)).asType(.float16)
                eval(xT)
                let data = xT.asData().data
                let rowBytes = x.dim(0) * 2
                let strideBytes = prepared.rowStride * 2
                precondition(data.count == inputDim * rowBytes)
                prepared.inputSurface.lock(options: [], seed: nil)
                memset(prepared.inputSurface.baseAddress, 0, inputDim * strideBytes)
                data.withUnsafeBytes { raw in
                    for channel in 0 ..< inputDim {
                        _ = memcpy(prepared.inputSurface.baseAddress + channel * strideBytes,
                            raw.baseAddress! + channel * rowBytes, rowBytes)
                    }
                }
                prepared.inputSurface.unlock(options: [], seed: nil)
            }
        } catch {
            prepared.ownership.lock()
            prepared.storage.generation = prepared.generation
            prepared.state = .idle
            prepared.ownership.unlock()
            throw error
        }
        prepared.ownership.lock()
        prepared.state = .staged
        prepared.ownership.unlock()
        return invocation
    }

    /// BACKGROUND-SAFE. Blocking evaluate at qos 21 (0x15, matching oMLX)
    /// against the same empty execution-options dictionary compile/load
    /// used. Touches only the private ANE ObjC API -- no MLX -- so this is
    /// the one phase safe to run on `ConcurrentEngines.run`'s background
    /// queue while the caller thread does MLX `eval` elsewhere.
    public static func evaluate(_ prepared: Prepared) throws {
        prepared.ownership.lock()
        guard prepared.generation == prepared.storage.generation,
              prepared.state == .staged else {
            prepared.ownership.unlock()
            throw ANEDispatchError.requestBusy
        }
        prepared.state = .evaluating
        prepared.ownership.unlock()
        do {
            try prepared.storage.inputTransfer?.wait()
        } catch {
            prepared.ownership.lock()
            prepared.state = .idle
            prepared.ownership.unlock()
            throw error
        }
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let evaluateFn = unsafeBitCast(msgSend, to: EvaluateFn.self)
        let evaluateSel = Selector(("evaluateWithQoS:options:request:error:"))
        var errU: Unmanaged<NSError>?
        let ok = withUnsafeMutablePointer(to: &errU) { p in
            evaluateFn(prepared.model.raw, evaluateSel, 21, NSDictionary(), prepared.request, p).boolValue
        }
        if !ok {
            prepared.ownership.lock()
            prepared.state = .idle
            prepared.ownership.unlock()
            let message = errU?.takeUnretainedValue().localizedDescription ?? "unknown evaluate failure"
            throw ANEDispatchError.evaluateFailed(message)
        }
        prepared.ownership.lock()
        prepared.state = .complete
        prepared.ownership.unlock()
    }

    /// CALLER THREAD ONLY (MLX). Gathers `[OUT,S]` out of the ANE's
    /// `[OUT,paddedS]` row-padded output surface and returns `[S,OUT]` fp16
    /// (MLX). Must run only after `evaluate` has completed for this
    /// `Prepared`.
    public static func read(_ prepared: Prepared) -> MLXArray {
        prepared.ownership.lock()
        defer { prepared.ownership.unlock() }
        precondition(prepared.generation == prepared.storage.generation, "ANE invocation handle has expired")
        precondition(prepared.state == .complete || prepared.state == .read,
            "ANE output requires a completed evaluation")
        prepared.state = .read
        let outputDim = prepared.outputDim
        let sequenceLength = prepared.sequenceLength
        let rowStride = prepared.rowStride
        let outputByteCount = outputDim * sequenceLength * 2
        let outputRowStrideBytes = rowStride * 2
        let outputRowBytes = sequenceLength * 2

        prepared.outputSurface.lock(options: .readOnly, seed: nil)
        var packed = Data(count: outputByteCount)
        let baseAddress = prepared.outputSurface.baseAddress
        packed.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Void in
            let dst = raw.baseAddress!
            for f in 0 ..< outputDim {
                _ = memcpy(dst + f * outputRowBytes, baseAddress + f * outputRowStrideBytes, outputRowBytes)
            }
        }
        prepared.outputSurface.unlock(options: .readOnly, seed: nil)

        // Output is [OUT,S] row-major after the gather above.
        let yT = MLXArray(packed, [outputDim, sequenceLength], type: Float16.self)
        return contiguous(yT.transposed(1, 0)) // [S,OUT]
    }

    /// CALLER THREAD ONLY (MLX). Zero-copy variant of `read`: wraps the
    /// output IOSurface's bytes directly as an MLXArray (no CPU gather),
    /// slices the real `[OUT,S]` region, and returns `[S,OUT]` fp16 (a
    /// `contiguous` copy of the transposed view, so the surface bytes are
    /// materialized into an MLX-owned buffer before the wrapper is dropped).
    /// The surface is `[OUT, paddedS]` fp16; we wrap that full padded shape
    /// and slice `[0..<OUT, 0..<S]` so the padding lanes are never read.
    /// The finalizer keeps the surface alive for the wrapper's lifetime.
    ///
    /// No `lock(.readOnly)` (unlike `read`): `evaluate` is a blocking ANE
    /// call that returns only after the ANE finished writing this surface,
    /// and the D1 concurrent path joins (`ConcurrentEngines.run`'s
    /// `group.wait()`) before calling here -- so the write is complete and
    /// visible before MLX reads the bytes (bit-identity vs the locked gather
    /// path confirms it). mlx-swift's own `MLXArray(rawPointer:)` IOSurface
    /// example is likewise unlocked. Must run only after `evaluate` completed.
    public static func readZeroCopy(_ prepared: Prepared) -> MLXArray {
        // A [S,1] transpose is already contiguous, so contiguous() may alias
        // shared storage. Copy this degenerate shape before allowing reuse.
        if prepared.outputDim == 1 { return read(prepared) }
        prepared.ownership.lock()
        defer { prepared.ownership.unlock() }
        precondition(prepared.generation == prepared.storage.generation, "ANE invocation handle has expired")
        precondition(prepared.state == .complete || prepared.state == .read,
            "ANE output requires a completed evaluation")
        prepared.state = .read
        let outputDim = prepared.outputDim
        let sequenceLength = prepared.sequenceLength
        let rowStride = prepared.rowStride
        let surface = prepared.outputSurface
        // Hold a strong ref for the wrapper's lifetime; released in finalizer.
        let wrapped = MLXArray(
            rawPointer: surface.baseAddress,
            [outputDim, rowStride], dtype: .float16,
            finalizer: { [surface] in _ = surface })
        let real = wrapped[0 ..< outputDim, 0 ..< sequenceLength] // [OUT,S] view
        let output = contiguous(real.transposed(1, 0)) // [S,OUT]
        prepared.pendingOutputs.append(output)
        return output
    }

    /// Fuses strided output conversion and addition into independent storage.
    /// Register the lazy result before allowing another invocation to stage.
    static func readJoined(_ prepared: Prepared, gpu: MLXArray?, rows: Int, dtype: DType) -> MLXArray {
        prepared.ownership.lock()
        defer { prepared.ownership.unlock() }
        precondition(prepared.generation == prepared.storage.generation, "ANE invocation handle has expired")
        precondition(prepared.state == .complete || prepared.state == .read,
            "ANE output requires a completed evaluation")
        precondition(rows <= prepared.sequenceLength)
        let surface = prepared.outputSurface
        let wrapped = MLXArray(rawPointer: surface.baseAddress,
            [prepared.outputDim, prepared.rowStride], dtype: .float16,
            finalizer: { [surface] in _ = surface })
        let output = ANESurfaceJoin.callAsFunction(wrapped, gpu: gpu, rows: rows,
            hidden: prepared.outputDim, stride: prepared.rowStride, dtype: dtype)
        prepared.pendingOutputs.append(output)
        prepared.state = .read
        return output
    }

    /// Convenience that chains `prepare` -> `evaluate` -> `readZeroCopy` on
    /// the calling thread. `x` is `[S, IN]` fp16 (MLX). Returns `[S, OUT]` fp16
    /// (MLX). For concurrent ANE+GPU use, call the three phases separately
    /// instead (only `evaluate` is background-safe -- see `Prepared`'s doc
    /// comment).
    public static func runConv(model: ANEInMemoryModel, x: MLXArray, inputDim: Int, outputDim: Int, sequenceLength: Int) throws -> MLXArray {
        let prepared = try prepare(model: model, x: x, inputDim: inputDim, outputDim: outputDim, sequenceLength: sequenceLength)
        try evaluate(prepared)
        return readZeroCopy(prepared)
    }
}

extension ANEDirectDispatch {
    public enum CacheLoopError: Error {
        case invalidInput(String)
        case invalidated
    }

    /// Serial cache updates retain channel-major cache bytes in alternating surfaces.
    /// The caller stages fresh KV, slots, and optional queries/lengths. No MLX graph runs in `step`.
    public final class CacheLoop {
        private let model: ANEInMemoryModel
        private let attentionModel: ANEInMemoryModel?
        private let attentionSurface: IOSurface?
        private let attentionRequests: [AnyObject]
        private let surfaces: [IOSurface]
        private let requests: [AnyObject]
        private let attention: Bool
        private let batch: Int
        private let capacity: Int
        private let cacheChannels: Int
        private let stride = 32
        private let lock = NSLock()
        private var current = 0
        private var valid = true

        public init(batch: Int, capacity: Int, attention: Bool = false) throws {
            let text = try buildQwenANECacheUpdateMILText(batch: batch, capacity: capacity)
            let cacheChannels = 2048 * capacity
            let bytes = (cacheChannels + (attention ? 8704 : 2304)) * 32 * 2
            guard let a = ANEDirectDispatch.makeSurface(byteCount: bytes),
                  let b = ANEDirectDispatch.makeSurface(byteCount: bytes) else {
                throw ANEDispatchError.surfaceAllocationFailed
            }
            memset(a.baseAddress, 0, bytes)
            memset(b.baseAddress, 0, bytes)
            let model = try ANEInMemoryModel(milText: text, weightBlob: Data())
            try model.compile()
            try model.load()
            var completed = false
            defer { if !completed { model.unload() } }
            guard let surfaceClass = ANERuntime.cls("_ANEIOSurfaceObject"),
                  let objectA = ANERuntime.send(surfaceClass, Selector(("objectWithIOSurface:")), a as AnyObject),
                  let objectB = ANERuntime.send(surfaceClass, Selector(("objectWithIOSurface:")), b as AnyObject) else {
                throw ANEDispatchError.wrapFailed
            }
            guard let requestClass = ANERuntime.cls("_ANERequest") else { throw ANEDispatchError.requestFailed }
            let send = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
            let factory = unsafeBitCast(send, to: RequestFactory.self)
            let selector = Selector(("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"))
            var requests: [AnyObject] = []
            for (input, output) in [(objectA, objectB), (objectB, objectA)] {
                guard let request = factory(requestClass, selector,
                    [input] as NSArray, [0] as NSArray, [output] as NSArray, [0] as NSArray,
                    nil, nil, NSNumber(value: 0)) else { throw ANEDispatchError.requestFailed }
                requests.append(request.takeUnretainedValue())
            }
            var attentionModel: ANEInMemoryModel?
            var attentionSurface: IOSurface?
            var attentionRequests: [AnyObject] = []
            if attention {
                let attentionText = try buildQwenANECachedAttentionMILText(batch: batch, capacity: capacity)
                let program = try ANEInMemoryModel(milText: attentionText, weightBlob: Data())
                try program.compile(); try program.load()
                var attentionCompleted = false
                defer { if !attentionCompleted { program.unload() } }
                guard let surface = ANEDirectDispatch.makeSurface(byteCount: 6144 * 32 * 2),
                      let output = ANERuntime.send(surfaceClass, Selector(("objectWithIOSurface:")), surface as AnyObject) else {
                    throw ANEDispatchError.surfaceAllocationFailed
                }
                for input in [objectA, objectB] {
                    guard let request = factory(requestClass, selector,
                        [input] as NSArray, [0] as NSArray, [output] as NSArray, [0] as NSArray,
                        nil, nil, NSNumber(value: 0)) else { throw ANEDispatchError.requestFailed }
                    attentionRequests.append(request.takeUnretainedValue())
                }
                attentionModel = program; attentionSurface = surface; attentionCompleted = true
            }
            self.attentionModel = attentionModel
            self.attentionSurface = attentionSurface
            self.attentionRequests = attentionRequests
            self.model = model
            self.surfaces = [a, b]
            self.requests = requests
            self.attention = attention
            self.batch = batch
            self.capacity = capacity
            self.cacheChannels = cacheChannels
            completed = true
        }

        deinit { attentionModel?.unload(); model.unload() }

        /// `newKV` is lane-major [batch, 2, 4, 256]. Slots must be -1 or valid local indices.
        /// Attention mode requires queries [batch, 24, 256] and positive valid lengths.
        /// Returns lane-major attention in that mode, otherwise an empty array.
        @discardableResult
        public func step(newKV: [Float16], slots: [Int], queries: [Float16] = [], validLengths: [Int] = []) throws -> [Float16] {
            guard newKV.count == batch * 2048, slots.count == batch,
                  newKV.allSatisfy({ $0.isFinite }),
                  slots.allSatisfy({ $0 >= -1 && $0 < capacity }) else {
                throw CacheLoopError.invalidInput("Expected finite [batch, 2048] KV rows and slots in -1..<capacity")
            }
            if attention {
                guard queries.count == batch * 6144, queries.allSatisfy({ $0.isFinite }),
                      validLengths.count == batch, validLengths.allSatisfy({ $0 > 0 && $0 <= capacity }),
                      zip(slots, validLengths).allSatisfy({ $0 < $1 }) else {
                    throw CacheLoopError.invalidInput("Attention requires finite queries and positive valid lengths containing each write slot")
                }
            } else if !queries.isEmpty || !validLengths.isEmpty {
                throw CacheLoopError.invalidInput("Queries require attention mode")
            }
            lock.lock(); defer { lock.unlock() }
            guard valid else { throw CacheLoopError.invalidated }
            let pointer = surfaces[current].baseAddress.assumingMemoryBound(to: Float16.self)
            for lane in 0..<batch {
                for channel in 0..<2048 {
                    pointer[(cacheChannels + channel) * stride + lane] = newKV[lane * 2048 + channel]
                }
                pointer[(cacheChannels + 2048) * stride + lane] = Float16(slots[lane])

            }
            let send = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
            let evaluate = unsafeBitCast(send, to: EvaluateFn.self)
            var error: Unmanaged<NSError>?
            let ok = withUnsafeMutablePointer(to: &error) {
                evaluate(model.raw, Selector(("evaluateWithQoS:options:request:error:")),
                         21, NSDictionary(), requests[current], $0).boolValue
            }
            guard ok else {
                valid = false
                throw ANEDispatchError.evaluateFailed(error?.takeUnretainedValue().localizedDescription ?? "cache loop evaluation failed")
            }
            current = 1 - current
            guard attention else { return [] }
            guard let attentionModel, let attentionSurface else { throw CacheLoopError.invalidated }
            let attentionInput = surfaces[current].baseAddress.assumingMemoryBound(to: Float16.self)
            for lane in 0..<batch {
                for channel in 0..<6144 {
                    attentionInput[(cacheChannels + 2304 + channel) * stride + lane] = queries[lane * 6144 + channel]
                }
                attentionInput[(cacheChannels + 8448) * stride + lane] = Float16(validLengths[lane])
            }
            var attentionError: Unmanaged<NSError>?
            let attentionOK = withUnsafeMutablePointer(to: &attentionError) {
                evaluate(attentionModel.raw, Selector(("evaluateWithQoS:options:request:error:")),
                         21, NSDictionary(), attentionRequests[current], $0).boolValue
            }
            guard attentionOK else {
                valid = false
                throw ANEDispatchError.evaluateFailed(attentionError?.takeUnretainedValue().localizedDescription ?? "cached attention evaluation failed")
            }
            let output = attentionSurface.baseAddress.assumingMemoryBound(to: Float16.self)
            var attended = [Float16](repeating: 0, count: batch * 6144)
            for lane in 0..<batch {
                for channel in 0..<6144 {
                    attended[lane * 6144 + channel] = output[channel * stride + lane]
                }
            }
            return attended
        }

        /// Explicit diagnostic readback. Normal updates leave the cache on shared surfaces.
        public func snapshot() throws -> [Float16] {
            lock.lock(); defer { lock.unlock() }
            guard valid else { throw CacheLoopError.invalidated }
            let pointer = surfaces[current].baseAddress.assumingMemoryBound(to: Float16.self)
            var result = [Float16](repeating: 0, count: batch * cacheChannels)
            for lane in 0..<batch {
                for channel in 0..<cacheChannels {
                    result[lane * cacheChannels + channel] = pointer[channel * stride + lane]
                }
            }
            return result
        }
    }
}

extension ANEDirectDispatch {
    public enum FP16LoopError: Error {
        case invalidShape
        case invalidInput
        case invalidated
        case nonfiniteOutput
    }

    /// Owns a loaded program and reusable surfaces. Execution uses CPU buffer packing
    /// and ANE evaluation only; returned lane-major values own their storage.
    public final class FP16Loop {
        private let storage: Storage
        private let inputDim: Int
        private let outputDim: Int
        private let batch: Int
        private let lock = NSLock()
        private var valid = true

        public init(milText: String, weightBlob: Data, inputDim: Int, outputDim: Int, batch: Int, weightFileName: String = "weight_data.bin") throws {
            let inputBytes = inputDim.multipliedReportingOverflow(by: 64)
            let outputBytes = outputDim.multipliedReportingOverflow(by: 64)
            guard inputDim > 0, outputDim > 0, (1...32).contains(batch),
                  !inputBytes.overflow, !outputBytes.overflow else { throw FP16LoopError.invalidShape }
            guard let input = ANEDirectDispatch.makeSurface(byteCount: inputBytes.partialValue),
                  let output = ANEDirectDispatch.makeSurface(byteCount: outputBytes.partialValue) else {
                throw ANEDispatchError.surfaceAllocationFailed
            }
            memset(input.baseAddress, 0, inputBytes.partialValue)
            memset(output.baseAddress, 0, outputBytes.partialValue)
            let model = try ANEInMemoryModel(milText: milText, weightBlob: weightBlob, weightFileName: weightFileName)
            try model.compile(); try model.load()
            var completed = false
            defer { if !completed { model.unload() } }
            let request = try ANEDirectDispatch.makeRequest(inputSurface: input, outputSurface: output)
            self.storage = Storage(model: model, request: request, inputSurface: input,
                outputSurface: output, outputDim: outputDim, sequenceLength: batch, rowStride: 32)
            self.inputDim = inputDim; self.outputDim = outputDim; self.batch = batch
            completed = true
        }

        deinit { storage.model.unload() }

        public func callAsFunction(_ input: [Float16]) throws -> [Float16] {
            guard input.count == batch * inputDim, input.allSatisfy({ $0.isFinite }) else {
                throw FP16LoopError.invalidInput
            }
            lock.lock(); defer { lock.unlock() }
            guard valid else { throw FP16LoopError.invalidated }
            let pointer = storage.inputSurface.baseAddress.assumingMemoryBound(to: Float16.self)
            for lane in 0..<batch {
                for channel in 0..<inputDim { pointer[channel * 32 + lane] = input[lane * inputDim + channel] }
            }
            storage.ownership.lock()
            storage.generation += 1; storage.state = .staged
            let invocation = Prepared(storage)
            storage.ownership.unlock()
            do { try ANEDirectDispatch.evaluate(invocation) }
            catch { valid = false; throw error }
            let output = storage.outputSurface.baseAddress.assumingMemoryBound(to: Float16.self)
            var values = [Float16](repeating: 0, count: batch * outputDim)
            for lane in 0..<batch {
                for channel in 0..<outputDim { values[lane * outputDim + channel] = output[channel * 32 + lane] }
            }
            storage.ownership.lock(); storage.state = .read; storage.ownership.unlock()
            guard values.allSatisfy({ $0.isFinite }) else {
                valid = false; throw FP16LoopError.nonfiniteOutput
            }
            return values
        }
    }
}
