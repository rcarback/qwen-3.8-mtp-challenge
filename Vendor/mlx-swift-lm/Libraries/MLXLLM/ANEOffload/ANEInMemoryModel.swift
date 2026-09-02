// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLXFAST_ANE_DIRECT=1 at call sites.
//
// In-memory compile + load of a MIL program through the private
// `_ANEInMemoryModelDescriptor` / `_ANEInMemoryModel` classes -- the path
// `ANEGemm` reaches indirectly through `MLModelAsset`/`MLModel.load`, but
// exercised here directly so later tasks (IOSurface I/O, the procedure bank)
// can drive the ANE compiler without going through Core ML's public model
// wrapper at all. See
// `.superpowers/sdd/2026-08-31-ane-iosurface-procedure-bank/task-2-brief.md`.
import Foundation
import ObjectiveC

public final class ANEInMemoryModel {
    public enum ANEError: Error, Equatable { case unavailable, descriptor, model, compile(String), load(String) }

    /// The `_ANEInMemoryModel` instance (for Task 4).
    let raw: AnyObject
    public private(set) var programHandle: UInt64 = 0
    private let scratchURL: URL

    /// `objc_msgSend` cast for a plain 0-arg call returning `Unmanaged` --
    /// used only for `alloc`, kept unmanaged (see `init` below) rather than
    /// going through `ANERuntime.send`'s always-ARC-converting return.
    private typealias AllocFn = @convention(c) (AnyObject?, Selector) -> Unmanaged<AnyObject>?

    /// `objc_msgSend` cast for the descriptor's 4-payload-arg init:
    /// `initWithNetworkText:weights:optionsPlist:isMILModel:` -> `(id,id,id,BOOL)`.
    /// Not covered by `ANERuntime.send`'s 0/1/3-object-arg overloads because
    /// the last argument here is a `BOOL`, not an object.
    private typealias Init4 = @convention(c) (AnyObject?, Selector, AnyObject?, AnyObject?, AnyObject?, ObjCBool) -> Unmanaged<AnyObject>?

    /// `objc_msgSend` cast for a `BOOL(id,SEL,NSInteger,id,NSError**)` call
    /// with a correctly-bridged `NSError**` out-param -- covers
    /// `compileWithQoS:options:error:` and `loadWithQoS:options:error:`.
    ///
    /// Deliberately NOT `ANERuntime.sendBoolQoS`, even though that shim now
    /// covers the same selector shape via a correctly-bridged
    /// `AutoreleasingUnsafeMutablePointer<NSError?>` (see `SendBoolQoS`'s
    /// doc comment in `ANERuntimeBridge.swift`): this call site was
    /// authored before that fix, against a plain `UnsafeMutablePointer<NSError?>`
    /// shim, and writing an ObjC `__autoreleasing`-convention out-param
    /// through that type from a raw `@convention(c)` call bypasses the
    /// retain Swift's ARC-aware bridging would normally insert on that
    /// write. The callee hands back an autoreleased (+0) `NSError*`;
    /// reading it into a plain `UnsafeMutablePointer<NSError?>.pointee`
    /// gives Swift's `err` variable the bit pattern without ever
    /// performing that retain, so Swift believes it owns a reference it
    /// never actually retained. The later release Swift inserts for that
    /// variable -- at explicit reassignment or at scope exit, the latter
    /// coinciding with an enclosing `autoreleasepool`'s own pop -- is then
    /// unbalanced and corrupts the allocator. Confirmed live and
    /// root-caused by bisection (reproduced standalone, outside Swift
    /// Testing, in single-threaded scripts): passing a null error pointer
    /// never crashes; passing that plain-pointer shim always crashes at
    /// the next release of `err`, deterministically, regardless of how
    /// long the scope is kept open first. `UnsafeMutablePointer<Unmanaged<NSError>?>`
    /// (what `SendBoolQoSErr` below uses) receives the same raw write but
    /// performs NO implicit ARC on its own; calling `.takeUnretainedValue()`
    /// on the result explicitly performs exactly the one retain the
    /// callee's +0 convention requires, which balances correctly against
    /// Swift's later release.
    private typealias SendBoolQoSErr = @convention(c) (AnyObject?, Selector, Int, AnyObject?, UnsafeMutablePointer<Unmanaged<NSError>?>?) -> ObjCBool

    /// `objc_msgSend` cast for `unloadWithQoS:error:` -> `(NSInteger,NSError**)`,
    /// confirmed against the live class dump to take only two payload
    /// arguments (no `options:`), unlike `compileWithQoS:options:error:` and
    /// `loadWithQoS:options:error:` which both take three. Same
    /// `Unmanaged<NSError>?` error-param treatment as `SendBoolQoSErr`, one
    /// argument shorter.
    private typealias SendBoolQoSErrNoOptions = @convention(c) (AnyObject?, Selector, Int, UnsafeMutablePointer<Unmanaged<NSError>?>?) -> ObjCBool

    /// Calls `compileWithQoS:options:error:` / `loadWithQoS:options:error:`
    /// and returns `(ok, message)` -- see `SendBoolQoSErr` for why this
    /// exists instead of `ANERuntime.sendBoolQoS`.
    private static func callBoolQoSErr(_ msgSend: UnsafeMutableRawPointer, _ r: AnyObject?, _ s: Selector, qos: Int, options: AnyObject?) -> (ok: Bool, message: String?) {
        let f = unsafeBitCast(msgSend, to: SendBoolQoSErr.self)
        var errU: Unmanaged<NSError>?
        let ok = withUnsafeMutablePointer(to: &errU) { p in f(r, s, qos, options, p).boolValue }
        let e: NSError? = errU?.takeUnretainedValue()
        return (ok, e.map(ANEInMemoryModel.describe))
    }

    /// `compileWithQoS:` wraps the ANE compiler's actual failure (e.g.
    /// `ANECCompile(...) FAILED: err=(InvalidCompilationParam)`) inside a
    /// generic `_ANECompiler : ANECCompile() FAILED` outer error and puts
    /// the useful detail under `NSUnderlyingErrorKey` -- `localizedDescription`
    /// alone drops that detail, so this folds one level of underlying error
    /// into the message.
    private static func describe(_ e: NSError) -> String {
        guard let under = e.userInfo[NSUnderlyingErrorKey] as? NSError else { return e.localizedDescription }
        return "\(e.localizedDescription): \(under.localizedDescription)"
    }

    /// Calls `unloadWithQoS:error:` (no `options:` argument) and returns
    /// `(ok, message)`.
    private static func callBoolQoSErrNoOptions(_ msgSend: UnsafeMutableRawPointer, _ r: AnyObject?, _ s: Selector, qos: Int) -> (ok: Bool, message: String?) {
        let f = unsafeBitCast(msgSend, to: SendBoolQoSErrNoOptions.self)
        var errU: Unmanaged<NSError>?
        let ok = withUnsafeMutablePointer(to: &errU) { p in f(r, s, qos, p).boolValue }
        return (ok, errU?.takeUnretainedValue().localizedDescription)
    }

    /// `milText`: a MIL TEXT program (see `ANEMILBuilder.buildConvMILText`),
    /// not the binary MIL protobuf `buildConvMILProgram` emits -- Task 2's
    /// binary-proto program failed `compileWithQoS:` with
    /// `InvalidCompilationParam`; the open-source oMLX project's
    /// `fp16_linear_mil`/`load_or_compile_ane_model`
    /// (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_ane.mm`) prove the
    /// private ANE compiler accepts MIL text, unentitled, when the
    /// descriptor's referenced weight file is staged on disk first.
    /// `weightBlob`: the full on-disk blob the MIL text's `BLOBFILE`
    /// reference reads (see `ANEMILBuilder.buildConvWeightBlob` -- the
    /// `make_blob` chunk-descriptor header the MIL text's `offset=uint64(64)`
    /// points into, with the fp16 weight payload itself starting at
    /// absolute offset 128).
    /// `weightFileName`: the file the MIL text's `BLOBFILE` references are
    /// staged under (`weights/<weightFileName>`). Defaults to
    /// `weight_data.bin`, matching `buildConvMILText`'s single-weight
    /// `BLOBFILE` path; a fused multi-weight program (see `ANEFusedMLP`)
    /// passes `weight.bin` to match `buildSwiGLUDownMILText`'s path instead.
    public init(milText: String, weightBlob: Data, weightFileName: String = "weight_data.bin") throws {
        guard ANERuntime.available() else { throw ANEError.unavailable }
        guard let Desc = ANERuntime.cls("_ANEInMemoryModelDescriptor") else { throw ANEError.descriptor }

        let milData = Data(milText.utf8)

        // alloc/init dance, kept manually balanced: `alloc` hands back a
        // single owned (+1) reference that `init...` consumes and returns
        // through (self, unchanged in the common case, still the same +1).
        // Converting BOTH the `alloc` result and the `init...` result to
        // ARC-tracked Swift values via `takeRetainedValue()` -- as
        // `ANERuntime.send(..., retained: true)` would for the `alloc` call
        // -- double-counts that single +1 as two independent ownership
        // claims, which crashes (over-release / use-after-free, confirmed
        // live as a SIGSEGV inside `objc_release`/`AutoreleasePoolPage`)
        // once the first claim's scope ends and frees an object the second
        // claim still points at. So `alloc`'s Unmanaged result is used only
        // transiently (`takeUnretainedValue()` as a call argument, a
        // self-balancing temporary retain/release) and never bound to a
        // persistent Swift reference; only the `init...` result is taken as
        // owned, which correctly consumes the one outstanding +1 from alloc.
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let allocFn = unsafeBitCast(msgSend, to: AllocFn.self)
        guard let allocU = allocFn(Desc, NSSelectorFromString("alloc")) else { throw ANEError.descriptor }

        // `optionsPlist` must be a real serialized property list, even an
        // empty one -- passing `NSData()` (zero bytes) here is what produced
        // `InvalidCompilationParam` even with correctly staged MIL text and
        // weight files: the framework writes this argument verbatim to
        // `compiler_options.plist` in the staging directory, and the ANE
        // compiler rejects a zero-byte file as an invalid plist before it
        // ever reaches MIL parsing. An empty dictionary serialized as an XML
        // plist clears that gate; the actual content is otherwise unused for
        // this program (no per-model compiler options needed here).
        let emptyOptionsPlist = try PropertyListSerialization.data(fromPropertyList: [String: Any](), format: .xml, options: 0)
        let initSel = Selector(("initWithNetworkText:weights:optionsPlist:isMILModel:"))
        let initFn = unsafeBitCast(msgSend, to: Init4.self)
        guard let descU = initFn(allocU.takeUnretainedValue(), initSel, milData as NSData, NSDictionary(), emptyOptionsPlist as NSData, true) else {
            throw ANEError.descriptor
        }
        let desc = descU.takeRetainedValue()

        guard let Mem = ANERuntime.cls("_ANEInMemoryModel") else { throw ANEError.model }
        guard let model = ANERuntime.send(Mem, Selector(("inMemoryModelWithDescriptor:")), desc) else {
            throw ANEError.model
        }

        // `hexStringIdentifier` is the descriptor-content hash
        // `_ANEInMemoryModel` uses to derive its own on-disk staging
        // location under `NSTemporaryDirectory()`. It covers the NETWORK
        // TEXT only (the weights dictionary passed above is empty and the
        // blob is a staged side file), so two programs with the same MIL
        // text share one identity, one staging directory, and one compiled
        // program regardless of their weights. The MIL builders therefore
        // stamp a unique `programTag` into `buildInfo`. Per oMLX (which never
        // calls `setModelURL:` on this path -- overriding the derived URL
        // breaks per-file bundle-hash verification on newer macOS), staging
        // `model.mil` and `weights/weight_data.bin` at that SAME derived
        // path is what lets `compileWithQoS:options:error:` find them
        // without us pointing `modelURL` anywhere ourselves. Task 2's nil-
        // modelURL segfault inside `saveModelFiles` was reproduced with NO
        // staged files present at all (a binary-proto program, no on-disk
        // inputs); staging the files this way clears that crash on this
        // macOS 26.5.2 build without needing the `setModelURL:` fallback.
        guard let identifierObj = ANERuntime.send(model, Selector(("hexStringIdentifier"))),
              let identifier = identifierObj as? String else {
            throw ANEError.model
        }
        let dir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(identifier)
        let weightsDir = dir.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)
        try milData.write(to: dir.appendingPathComponent("model.mil"))
        try weightBlob.write(to: weightsDir.appendingPathComponent(weightFileName))

        scratchURL = dir
        raw = model
    }

    deinit {
        try? FileManager.default.removeItem(at: scratchURL)
    }

    public func compile(qos: Int = 0x15) throws {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let (ok, message) = ANEInMemoryModel.callBoolQoSErr(msgSend, raw, Selector(("compileWithQoS:options:error:")), qos: qos, options: NSDictionary())
        if !ok { throw ANEError.compile(message ?? "unknown compile failure") }
    }

    public func load(qos: Int = 0x15) throws {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let (ok, message) = ANEInMemoryModel.callBoolQoSErr(msgSend, raw, Selector(("loadWithQoS:options:error:")), qos: qos, options: NSDictionary())
        if !ok { throw ANEError.load(message ?? "unknown load failure") }
        programHandle = ANERuntime.sendUInt64(raw, Selector(("programHandle")))
    }

    public func unload() {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        _ = ANEInMemoryModel.callBoolQoSErrNoOptions(msgSend, raw, Selector(("unloadWithQoS:error:")), qos: 0)
        programHandle = 0
    }
}
