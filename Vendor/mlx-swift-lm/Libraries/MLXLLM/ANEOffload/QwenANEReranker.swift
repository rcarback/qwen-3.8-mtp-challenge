import Foundation

public enum QwenANERerankerError: Error {
    case invalidShortlist
}

public final class QwenANEReranker {
    private let projection: QwenANECoarseReadout
    private let selector: QwenANEFinalSelector

    /// Tile blobs contain target-head weights in compact vocabulary order, not draft-head weights.
    public init(targetTileWeightBlobs: [Int: Data]) throws {
        projection = try QwenANECoarseReadout(tileWeightBlobs: targetTileWeightBlobs)
        selector = try QwenANEFinalSelector()
    }

    public func callAsFunction(hidden: [Float16], compactIDs: [Int32]) throws
        -> (tokens: [Int32], scores: [Float16]) {
        guard compactIDs.count == 1024 && compactIDs.allSatisfy({ $0 >= 0 && $0 < 98330 }) else {
            throw QwenANERerankerError.invalidShortlist
        }
        let logits = try projection(hidden)
        let scores = compactIDs.enumerated().map { index, id in logits[(index / 32) * 98336 + Int(id)] }
        let ids = compactIDs.map { $0 < 98304 ? $0 : 248044 + $0 - 98304 }
        return (try selector(scores: scores, tokenIDs: ids), scores)
    }
}
