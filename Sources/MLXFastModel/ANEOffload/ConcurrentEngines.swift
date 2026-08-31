// Small helper that runs an ANE closure and a GPU closure truly
// concurrently and joins, returning both results. See
// `.superpowers/sdd/2026-08-30-ane-gpu-concurrent-offload/task-3-brief.md`.
import Foundation

/// Cross-thread result box for the `ane` closure's outcome. `DispatchGroup`
/// enter/leave provides the happens-before/-after relationship that makes
/// writing on the background queue and reading after `group.wait()` safe
/// without a lock -- the same pattern `ANEGemm.init` and
/// `ANEChannelSplitPoCTests.ResultBox` use.
private final class ANEResultBox<A>: @unchecked Sendable {
    var value: A?
    var error: Error?
}

public enum ConcurrentEngines {
    /// Runs `ane` on a background QoS-userInitiated queue and `gpu` on the
    /// calling thread, joins, returns both. MLX `eval` is only ever called
    /// from ONE thread (the gpu closure, i.e. the calling thread) -- the
    /// `ane` closure must touch only Core ML, never MLX eval.
    ///
    /// Declared `throws`, not `rethrows`: the ANE closure's error is thrown
    /// from the *calling* thread after `group.wait()`, not from a `catch`
    /// that directly caught a closure-parameter call on that thread, which
    /// is what `rethrows` requires. `throws` is strictly more permissive and
    /// does not change any call site.
    public static func run<A, G>(ane: @escaping () throws -> A, gpu: () throws -> G) throws -> (A, G) {
        let aneBox = ANEResultBox<A>()
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                aneBox.value = try ane()
            } catch {
                aneBox.error = error
            }
            group.leave()
        }

        var gpuValue: G?
        var gpuError: Error?
        do {
            gpuValue = try gpu()
        } catch {
            gpuError = error
        }

        group.wait()

        if let aneError = aneBox.error { throw aneError }
        if let gpuError { throw gpuError }
        guard let aneResult = aneBox.value, let gpuResult = gpuValue else {
            throw NSError(domain: "ConcurrentEngines", code: 1,
                           userInfo: [NSLocalizedDescriptionKey: "run produced neither a result nor an error on one side"])
        }
        return (aneResult, gpuResult)
    }
}
