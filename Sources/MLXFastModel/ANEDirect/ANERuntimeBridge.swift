// LOCAL M4 FORK ONLY. Private Apple frameworks; breaks on macOS updates.
// Never wired into the ranked forward; gated by MLXFAST_ANE_DIRECT=1 at call sites.
import Darwin
import Foundation
import ObjectiveC

enum ANERuntime {
    nonisolated(unsafe) private static let handle: UnsafeMutableRawPointer? = {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)
    }()
    // objc_msgSend cast per-arity. dlopen(nil) resolves the already-linked symbol.
    nonisolated(unsafe) private static let raw: UnsafeMutableRawPointer = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!

    static func available() -> Bool {
        guard handle != nil else { return false }
        return ["_ANEInMemoryModelDescriptor", "_ANEInMemoryModel", "_ANEClient",
                "_ANERequest", "_ANEIOSurfaceObject"].allSatisfy { cls($0) != nil }
    }
    static func cls(_ name: String) -> AnyClass? { NSClassFromString(name) }

    typealias Send0 = @convention(c) (AnyObject?, Selector) -> Unmanaged<AnyObject>?
    typealias Send1 = @convention(c) (AnyObject?, Selector, AnyObject?) -> Unmanaged<AnyObject>?
    typealias Send3 = @convention(c) (AnyObject?, Selector, AnyObject?, AnyObject?, AnyObject?) -> Unmanaged<AnyObject>?
    typealias SendU64 = @convention(c) (AnyObject?, Selector) -> UInt64
    /// `NSError**` out-param bridged with the correct Swift type for an
    /// ObjC `__autoreleasing` out-param -- see `ANEInMemoryModel`'s
    /// `SendBoolQoSErr` doc comment for the ARC-corruption bug this avoids:
    /// a plain `UnsafeMutablePointer<NSError?>` writes the callee's +0
    /// autoreleased `NSError*` without the retain Swift's bridging would
    /// normally insert, so the later implicit release of the Swift `NSError?`
    /// variable is unbalanced and corrupts the allocator. Swift bridges an
    /// `inout NSError?` argument to `AutoreleasingUnsafeMutablePointer`
    /// automatically at the call site, which performs that retain.
    typealias SendBoolQoS = @convention(c) (AnyObject?, Selector, Int, AnyObject?, AutoreleasingUnsafeMutablePointer<NSError?>?) -> ObjCBool

    static func send(_ r: AnyObject?, _ s: Selector, retained: Bool = false) -> AnyObject? {
        let f = unsafeBitCast(raw, to: Send0.self)
        guard let u = f(r, s) else { return nil }
        return retained ? u.takeRetainedValue() : u.takeUnretainedValue()
    }
    static func send(_ r: AnyObject?, _ s: Selector, _ a: AnyObject?, retained: Bool = false) -> AnyObject? {
        let f = unsafeBitCast(raw, to: Send1.self)
        guard let u = f(r, s, a) else { return nil }
        return retained ? u.takeRetainedValue() : u.takeUnretainedValue()
    }
    static func send(_ r: AnyObject?, _ s: Selector, _ a: AnyObject?, _ b: AnyObject?, _ c: AnyObject?, retained: Bool = false) -> AnyObject? {
        let f = unsafeBitCast(raw, to: Send3.self)
        guard let u = f(r, s, a, b, c) else { return nil }
        return retained ? u.takeRetainedValue() : u.takeUnretainedValue()
    }
    static func sendUInt64(_ r: AnyObject?, _ s: Selector) -> UInt64 {
        unsafeBitCast(raw, to: SendU64.self)(r, s)
    }
    static func sendBoolQoS(_ r: AnyObject?, _ s: Selector, qos: Int, options: AnyObject?, error: inout NSError?) -> Bool {
        // `&error` bridges the `inout NSError?` to the
        // `AutoreleasingUnsafeMutablePointer<NSError?>` `SendBoolQoS` now
        // declares, which is what performs the correct retain on the
        // callee's autoreleased out-param write -- see `SendBoolQoS`'s doc
        // comment.
        unsafeBitCast(raw, to: SendBoolQoS.self)(r, s, qos, options, &error).boolValue
    }
}
