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

final class ANEInMemoryModel {
    enum ANEError: Error, Equatable { case unavailable, descriptor, model, compile(String), load(String) }

    /// The `_ANEInMemoryModel` instance (for Task 4).
    let raw: AnyObject
    private(set) var programHandle: UInt64 = 0
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

    /// `objc_msgSend` cast for a void 1-object-arg setter, e.g. `setModelURL:`.
    /// `ANERuntime.send` always casts the raw call through a
    /// `Unmanaged<AnyObject>?`-returning shim and then takes that "result" --
    /// harmless for methods that really return an object, but for a
    /// void-returning method the x0 register on return holds whatever the
    /// callee last left there (not guaranteed to be anything meaningful),
    /// and `send` would still retain/release it as if it were a real return
    /// value. Calling through a `Void`-returning shim instead reads no
    /// bogus return value at all.
    private typealias SetterFn = @convention(c) (AnyObject?, Selector, AnyObject?) -> Void

    /// `objc_msgSend` cast for a `BOOL(id,SEL,NSInteger,id,NSError**)` call
    /// with a correctly-bridged `NSError**` out-param -- covers
    /// `compileWithQoS:options:error:` and `loadWithQoS:options:error:`.
    ///
    /// Deliberately NOT `ANERuntime.sendBoolQoS`, despite that shim covering
    /// the same selector shape: its `error` parameter is a plain
    /// `UnsafeMutablePointer<NSError?>`, and writing an ObjC
    /// `__autoreleasing`-convention out-param through that type from a raw
    /// `@convention(c)` call bypasses the retain Swift's ARC-aware bridging
    /// would normally insert on that write. The callee hands back an
    /// autoreleased (+0) `NSError*`; reading it into a plain
    /// `UnsafeMutablePointer<NSError?>.pointee` gives Swift's `err` variable
    /// the bit pattern without ever performing that retain, so Swift
    /// believes it owns a reference it never actually retained. The later
    /// release Swift inserts for that variable -- at explicit reassignment
    /// or at scope exit, the latter coinciding with an enclosing
    /// `autoreleasepool`'s own pop -- is then unbalanced and corrupts the
    /// allocator. Confirmed live and root-caused by bisection (reproduced
    /// standalone, outside Swift Testing, in single-threaded scripts):
    /// passing a null error pointer never crashes; passing this same
    /// `UnsafeMutablePointer<NSError?>` shim always crashes at the next
    /// release of `err`, deterministically, regardless of how long the
    /// scope is kept open first. `UnsafeMutablePointer<Unmanaged<NSError>?>`
    /// receives the same raw write but performs NO implicit ARC on its own;
    /// calling `.takeUnretainedValue()` on the result explicitly performs
    /// exactly the one retain the callee's +0 convention requires, which
    /// balances correctly against Swift's later release.
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

    /// milProgram: bare MIL program bytes (the `program` submessage
    /// `buildConvMILProgram` returns), not a full CoreML `Model` proto.
    /// weights are baked as consts in the program, so the descriptor's
    /// `weights` argument is an empty dictionary.
    init(milProgram: Data) throws {
        guard ANERuntime.available() else { throw ANEError.unavailable }
        guard let Desc = ANERuntime.cls("_ANEInMemoryModelDescriptor") else { throw ANEError.descriptor }

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

        let initSel = Selector(("initWithNetworkText:weights:optionsPlist:isMILModel:"))
        let initFn = unsafeBitCast(msgSend, to: Init4.self)
        guard let descU = initFn(allocU.takeUnretainedValue(), initSel, milProgram as NSData, NSDictionary(), NSData(), true) else {
            throw ANEError.descriptor
        }
        let desc = descU.takeRetainedValue()

        guard let Mem = ANERuntime.cls("_ANEInMemoryModel") else { throw ANEError.model }
        guard let model = ANERuntime.send(Mem, Selector(("inMemoryModelWithDescriptor:")), desc) else {
            throw ANEError.model
        }

        // `inMemoryModelWithDescriptor:` leaves `modelURL` nil -- `_ANEInMemoryModel`
        // is "in-memory" only from the caller's perspective; `saveModelFiles`
        // (called from inside `compileWithQoS:options:error:`) still writes the
        // descriptor's MIL/weights out to a scratch directory on disk for the
        // out-of-process ANE compiler service to read, and dereferences `modelURL`
        // unconditionally to find it. Confirmed live: leaving it nil segfaults
        // inside `-[_ANEInMemoryModel saveModelFiles]` (EXC_BAD_ACCESS, objc_retain
        // on a garbage ivar) before any NSError is ever produced -- this is Apple's
        // framework code assuming a caller (normally Core ML's own model-loading
        // path) already set a working directory, not a validation gate we can
        // observe by inspecting the descriptor alone. One scratch directory per
        // instance avoids collisions across concurrent `ANEInMemoryModel`s.
        let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ane-in-memory-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let setModelURL = unsafeBitCast(msgSend, to: SetterFn.self)
        setModelURL(model, Selector(("setModelURL:")), scratch as NSURL)
        scratchURL = scratch
        raw = model
    }

    deinit {
        try? FileManager.default.removeItem(at: scratchURL)
    }

    func compile(qos: Int = 0x21) throws {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let (ok, message) = ANEInMemoryModel.callBoolQoSErr(msgSend, raw, Selector(("compileWithQoS:options:error:")), qos: qos, options: NSDictionary())
        if !ok { throw ANEError.compile(message ?? "unknown compile failure") }
    }

    func load(qos: Int = 0x21) throws {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        let (ok, message) = ANEInMemoryModel.callBoolQoSErr(msgSend, raw, Selector(("loadWithQoS:options:error:")), qos: qos, options: NSDictionary())
        if !ok { throw ANEError.load(message ?? "unknown load failure") }
        programHandle = ANERuntime.sendUInt64(raw, Selector(("programHandle")))
    }

    func unload() {
        let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
        _ = ANEInMemoryModel.callBoolQoSErrNoOptions(msgSend, raw, Selector(("unloadWithQoS:error:")), qos: 0)
        programHandle = 0
    }
}
