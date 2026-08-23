import Foundation

/// Whether an incoming request can continue the session the worker already
/// holds, or has to start over.
///
/// EXTENSION-ONLY BY CONSTRUCTION. The session's 48 recurrent gated-delta layers
/// cannot rewind -- `trim()` rolls back the 16 full-attention caches and nothing
/// else -- so the only reusable position is the exact end of what the worker has
/// already processed. Anything else (a shorter prompt, a diverging token, an
/// edited earlier message) is a restart, and treating it as anything else would
/// silently produce output conditioned on text the caller replaced.
///
/// This is pure logic on token arrays so it can be tested without a model.
enum ServePrefixDecision: Equatable {
    case restart
    case extend([Int])

    static func make(previous: [Int], incoming: [Int]) -> ServePrefixDecision {
        // An equal-length prompt is a restart, not an empty extension: an
        // extension with no tokens leaves no row to read the next token from.
        guard !previous.isEmpty, incoming.count > previous.count else {
            return .restart
        }
        guard Array(incoming.prefix(previous.count)) == previous else {
            return .restart
        }
        return .extend(Array(incoming.dropFirst(previous.count)))
    }
}
