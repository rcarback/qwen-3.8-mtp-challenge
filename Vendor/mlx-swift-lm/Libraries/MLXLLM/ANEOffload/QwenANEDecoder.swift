import Foundation

public enum QwenANEDecoderError: Error {
    case invalidInput
    case invalidStageOutput
    case invalidated
}

/// A 32-lane decoder with retained cache. Positions and visible lengths are supplied by the replay scheduler.
public final class QwenANEDecoder {
    private let inputProjection: ANEDirectDispatch.FP16Loop
    private let qkvProjection: ANEDirectDispatch.FP16Loop
    private let qkPreparation: ANEDirectDispatch.FP16Loop
    private let outputProjection: ANEDirectDispatch.FP16Loop
    private let tail: ANEDirectDispatch.FP16Loop
    private let cache: ANEDirectDispatch.CacheLoop
    private let capacity: Int
    private let lock = NSLock()
    private var invalidated = false

    public init(inputProjection: ANEDirectDispatch.FP16Loop, qkvProjection: ANEDirectDispatch.FP16Loop,
                qkPreparation: ANEDirectDispatch.FP16Loop, outputProjection: ANEDirectDispatch.FP16Loop,
                tail: ANEDirectDispatch.FP16Loop, capacity: Int = 1024) throws {
        self.inputProjection = inputProjection
        self.qkvProjection = qkvProjection
        self.qkPreparation = qkPreparation
        self.outputProjection = outputProjection
        self.tail = tail
        self.capacity = capacity
        cache = try ANEDirectDispatch.CacheLoop(batch: 32, capacity: capacity, attention: true)
    }

    public func callAsFunction(input: [Float16], positions: [Int], slots: [Int], validLengths: [Int]) throws -> [Float16] {
        guard input.count == 32 * 10240 && input.allSatisfy({ $0.isFinite }),
            positions.count == 32 && positions.allSatisfy({ $0 >= 0 && $0 < 1024 }),
            slots.count == 32 && validLengths.count == 32,
            zip(slots, validLengths).allSatisfy({ slot, length in
                length > 0 && length <= capacity && slot >= -1 && slot < length
            }) else { throw QwenANEDecoderError.invalidInput }
        lock.lock(); defer { lock.unlock() }
        guard !invalidated else { throw QwenANEDecoderError.invalidated }
        do {
            let residual = try inputProjection(input)
            let qkv = try qkvProjection(residual)
            guard residual.count == 32 * 5120 && qkv.count == 32 * 14336 else {
                throw QwenANEDecoderError.invalidStageOutput
            }
            let qk = try qkPreparation(packQwenANEQKRows(qkv: qkv, positions: positions))
            guard qk.count == 32 * 7168 else { throw QwenANEDecoderError.invalidStageOutput }
            var queries: [Float16] = [], fresh: [Float16] = []
            for lane in 0..<32 {
                queries.append(contentsOf: qk[(lane * 7168)..<(lane * 7168 + 6144)])
                fresh.append(contentsOf: qk[(lane * 7168 + 6144)..<((lane + 1) * 7168)])
                fresh.append(contentsOf: qkv[(lane * 14336 + 13312)..<((lane + 1) * 14336)])
            }
            let attention = try cache.step(newKV: fresh, slots: slots, queries: queries, validLengths: validLengths)
            guard attention.count == 32 * 6144 && attention.allSatisfy({ $0.isFinite }) else {
                throw QwenANEDecoderError.invalidStageOutput
            }
            var gatedInput: [Float16] = []
            for lane in 0..<32 {
                gatedInput.append(contentsOf: attention[(lane * 6144)..<((lane + 1) * 6144)])
                for head in 0..<24 {
                    let offset = lane * 14336 + head * 512 + 256
                    gatedInput.append(contentsOf: qkv[offset..<(offset + 256)])
                }
            }
            let projected = try outputProjection(gatedInput)
            guard projected.count == 32 * 5120 else { throw QwenANEDecoderError.invalidStageOutput }
            var tailInput: [Float16] = []
            for lane in 0..<32 {
                let range = (lane * 5120)..<((lane + 1) * 5120)
                tailInput.append(contentsOf: residual[range])
                tailInput.append(contentsOf: projected[range])
            }
            let hidden = try tail(tailInput)
            guard hidden.count == 32 * 5120 else { throw QwenANEDecoderError.invalidStageOutput }
            return hidden
        } catch {
            invalidated = true
            throw error
        }
    }

    public func snapshot() throws -> [Float16] {
        lock.lock(); defer { lock.unlock() }
        guard !invalidated else { throw QwenANEDecoderError.invalidated }
        return try cache.snapshot()
    }
}
