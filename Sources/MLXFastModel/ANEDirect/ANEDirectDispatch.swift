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

enum ANEDirectDispatch {
    enum ANEDispatchError: Error, CustomStringConvertible {
        case surfaceAllocationFailed
        case wrapFailed
        case requestFailed
        case evaluateFailed(String)

        var description: String {
            switch self {
            case .surfaceAllocationFailed: return "ANEDirectDispatch: IOSurface allocation failed"
            case .wrapFailed: return "ANEDirectDispatch: _ANEIOSurfaceObject wrap failed"
            case .requestFailed: return "ANEDirectDispatch: _ANERequest construction failed"
            case let .evaluateFailed(message): return "ANEDirectDispatch: evaluate failed: \(message)"
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

    /// Cross-thread carrier for the state `prepare` builds on the caller
    /// thread and that `evaluate` (background-safe) and `read` (caller
    /// thread) consume. `@unchecked Sendable` because `AnyObject`/
    /// `IOSurface` aren't `Sendable`, but the fields are written once in
    /// `prepare` and thereafter only read -- first by `evaluate` on
    /// `ConcurrentEngines.run`'s background queue, then by `read` back on
    /// the caller thread -- so there is no concurrent mutation, the same
    /// one-shot handoff pattern as `ANEGemm`'s `LoadInputs`/`LoadResult`.
    final class Prepared: @unchecked Sendable {
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

    /// CALLER THREAD ONLY (MLX). `x` is `[S, IN]` fp16 (MLX). Allocates the
    /// input/output IOSurfaces, writes `x` into the input surface (MLX
    /// transpose + eval + `asData`), wraps both surfaces, and builds the
    /// `_ANERequest`. Returns a `Prepared` that `evaluate` and `read`
    /// consume -- only `evaluate` may run off this thread.
    static func prepare(model: ANEInMemoryModel, x: MLXArray, inputDim: Int, outputDim: Int, sequenceLength: Int) throws -> Prepared {
        precondition(x.ndim == 2 && x.shape[0] == sequenceLength && x.shape[1] == inputDim,
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
        let inputByteCount = inputDim * sequenceLength * 2
        let inputSurfaceByteCount = inputDim * rowStride * 2
        let outputSurfaceByteCount = outputDim * rowStride * 2
        guard let inputSurface = makeSurface(byteCount: inputSurfaceByteCount),
              let outputSurface = makeSurface(byteCount: outputSurfaceByteCount) else {
            throw ANEDispatchError.surfaceAllocationFailed
        }

        // MIL conv input is tensor<fp16,[1,IN,1,S]> = [IN,S] row-major
        // (channel-major); our x is [S,IN], so transpose before writing.
        let xT = contiguous(x.transposed(1, 0)).asType(.float16) // [IN,S], row-contiguous
        eval(xT)
        let srcData = xT.asData().data
        precondition(srcData.count == inputByteCount,
                     "ANEDirectDispatch.prepare: transposed input is \(srcData.count) bytes, expected \(inputByteCount)")

        // Scatter [IN,S] into the ANE's [IN,paddedS] row-padded input --
        // one memcpy per channel row (degenerates to one bulk copy when
        // `sequenceLength == rowStride`, e.g. S=32).
        let inputRowStrideBytes = rowStride * 2
        let inputRowBytes = sequenceLength * 2
        inputSurface.lock(options: [], seed: nil)
        let inputBase = inputSurface.baseAddress
        srcData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Void in
            let src = raw.baseAddress!
            for c in 0 ..< inputDim {
                _ = memcpy(inputBase + c * inputRowStrideBytes, src + c * inputRowBytes, inputRowBytes)
            }
        }
        inputSurface.unlock(options: [], seed: nil)

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
            nil, nil, NSNumber(value: 0)
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

        return Prepared(model: model, request: request, inputSurface: inputSurface, outputSurface: outputSurface,
                         outputDim: outputDim, sequenceLength: sequenceLength, rowStride: rowStride)
    }

    /// BACKGROUND-SAFE. Blocking evaluate at qos 21 (0x15, matching oMLX)
    /// against the same empty execution-options dictionary compile/load
    /// used. Touches only the private ANE ObjC API -- no MLX -- so this is
    /// the one phase safe to run on `ConcurrentEngines.run`'s background
    /// queue while the caller thread does MLX `eval` elsewhere.
    static func evaluate(_ prepared: Prepared) throws {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let evaluateFn = unsafeBitCast(msgSend, to: EvaluateFn.self)
        let evaluateSel = Selector(("evaluateWithQoS:options:request:error:"))
        var errU: Unmanaged<NSError>?
        let ok = withUnsafeMutablePointer(to: &errU) { p in
            evaluateFn(prepared.model.raw, evaluateSel, 21, NSDictionary(), prepared.request, p).boolValue
        }
        if !ok {
            let message = errU?.takeUnretainedValue().localizedDescription ?? "unknown evaluate failure"
            throw ANEDispatchError.evaluateFailed(message)
        }
    }

    /// CALLER THREAD ONLY (MLX). Gathers `[OUT,S]` out of the ANE's
    /// `[OUT,paddedS]` row-padded output surface and returns `[S,OUT]` fp16
    /// (MLX). Must run only after `evaluate` has completed for this
    /// `Prepared`.
    static func read(_ prepared: Prepared) -> MLXArray {
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

    /// Convenience that chains `prepare` -> `evaluate` -> `read` on the
    /// calling thread. `x` is `[S, IN]` fp16 (MLX). Returns `[S, OUT]` fp16
    /// (MLX). For concurrent ANE+GPU use, call the three phases separately
    /// instead (only `evaluate` is background-safe -- see `Prepared`'s doc
    /// comment).
    static func runConv(model: ANEInMemoryModel, x: MLXArray, inputDim: Int, outputDim: Int, sequenceLength: Int) throws -> MLXArray {
        let prepared = try prepare(model: model, x: x, inputDim: inputDim, outputDim: outputDim, sequenceLength: sequenceLength)
        try evaluate(prepared)
        return read(prepared)
    }
}
