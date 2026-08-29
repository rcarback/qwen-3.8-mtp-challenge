import Foundation

/// Full text of tool results that compaction replaced with a stub.
///
/// Compaction removes bytes from the PROMPT, not from the conversation: the
/// model still has to be able to reach the original when it turns out to
/// matter. Without retrieval, a stub is indistinguishable from having deleted
/// the context, and the model's only recourse is re-running a tool call that
/// may be expensive, non-deterministic, or no longer valid.
///
/// Process-lifetime and in-memory. A handle that has fallen out (evicted, or
/// the server restarted) resolves to a plain explanation rather than an error,
/// because a failed expansion should degrade the answer, never the request.
final class CompactionStore: @unchecked Sendable {
    static let shared = CompactionStore()

    private let lock = NSLock()
    private var texts: [String: String] = [:]
    /// Insertion order, for eviction. Small enough that an array is right.
    private var order: [String] = []
    private var bytes = 0

    /// Cap on retained originals. These are the bytes compaction just removed
    /// from the prompt, so the store trades RAM for the ability to undo that.
    /// 512 MiB holds thousands of tool results against a 128 GB box.
    private let byteLimit = 512 * 1024 * 1024

    func put(handle: String, text: String) {
        lock.lock()
        defer { lock.unlock() }
        if texts[handle] != nil { return }
        texts[handle] = text
        order.append(handle)
        bytes += text.utf8.count
        while bytes > byteLimit, let oldest = order.first {
            order.removeFirst()
            if let dropped = texts.removeValue(forKey: oldest) {
                bytes -= dropped.utf8.count
            }
        }
    }

    func get(handle: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return texts[handle]
    }

    /// Test seam. The store is process-global, so a test that did not reset it
    /// would inherit whatever an earlier test wrote.
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        texts.removeAll()
        order.removeAll()
        bytes = 0
    }
}
