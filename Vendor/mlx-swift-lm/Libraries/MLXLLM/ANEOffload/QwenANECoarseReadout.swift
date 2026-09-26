import CryptoKit
import Foundation

public enum QwenANECoarseReadoutError: Error {
    case invalidWeights
    case invalidInput
}

public final class QwenANECoarseReadout {
    private let tiles: [(start: Int, count: Int, loop: ANEDirectDispatch.FP16Loop)]

    public init(tileWeightBlobs: [Int: Data]) throws {
        let starts = Array(stride(from: 0, to: 98336, by: 2048))
        guard Set(tileWeightBlobs.keys) == Set(starts) else {
            throw QwenANECoarseReadoutError.invalidWeights
        }
        for start in starts {
            let count = min(2048, 98336 - start)
            guard tileWeightBlobs[start]?.count == 128 + count * 5120 * 2 else {
                throw QwenANECoarseReadoutError.invalidWeights
            }
        }
        var prepared: [(start: Int, count: Int, loop: ANEDirectDispatch.FP16Loop)] = []
        for start in starts {
            guard let blob = tileWeightBlobs[start] else {
                throw QwenANECoarseReadoutError.invalidWeights
            }
            let count = min(2048, 98336 - start)
            let digest = SHA256.hash(data: blob).map { String(format: "%02x", $0) }.joined()
            let mil = buildConvMILText(inputDim: 5120, outputDim: count, sequenceLength: 32, programTag: digest)
            let loop = try ANEDirectDispatch.FP16Loop(milText: mil, weightBlob: blob,
                inputDim: 5120, outputDim: count, batch: 32)
            prepared.append((start, count, loop))
        }
        tiles = prepared
    }

    public func callAsFunction(_ hidden: [Float16]) throws -> [Float16] {
        guard hidden.count == 32 * 5120 && hidden.allSatisfy({ $0.isFinite }) else {
            throw QwenANECoarseReadoutError.invalidInput
        }
        var logits = [Float16](repeating: 0, count: 32 * 98336)
        for tile in tiles {
            let values = try tile.loop(hidden)
            for lane in 0..<32 {
                let offset = lane * 98336 + tile.start
                logits.replaceSubrange(offset..<(offset + tile.count),
                    with: values[(lane * tile.count)..<((lane + 1) * tile.count)])
            }
        }
        return logits
    }
}
