import CryptoKit
import Foundation

public enum QwenANEEmbeddingError: Error {
    case invalidConfiguration
    case invalidTokens
    case missingTile(Int)
    case programLoadFailed(tile: Int, retainedTiles: [Int], underlying: String)
}

/// Resident blobs are feature-major FP16 [5120, 2048], padded with zero rows in the final tile.
/// Only the bounded working set of tile programs is loaded on ANE.
public final class QwenANEEmbedding {
    private let blobs: [Int: Data]
    private let digests: [Int: String]
    private let limit: Int
    private var programs: [Int: ANEDirectDispatch.FP16Loop] = [:]
    private var recent: [Int] = []
    private let lock = NSLock()

    public init(tileWeightBlobs: [Int: Data], maxLoadedTiles: Int = 8) throws {
        guard maxLoadedTiles > 0, !tileWeightBlobs.isEmpty,
              tileWeightBlobs.allSatisfy({ (0..<122).contains($0.key) && $0.value.count == 128 + 5120 * 2048 * 2 }) else {
            throw QwenANEEmbeddingError.invalidConfiguration
        }
        blobs = tileWeightBlobs
        digests = tileWeightBlobs.mapValues { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
        limit = maxLoadedTiles
    }

    public var loadedTileCount: Int {
        lock.lock(); defer { lock.unlock() }
        return programs.count
    }

    public func callAsFunction(tokens: [Int]) throws -> [Float16] {
        guard tokens.count == 32, tokens.allSatisfy({ (0..<248320).contains($0) }) else {
            throw QwenANEEmbeddingError.invalidTokens
        }
        let tiles = Set(tokens.map { $0 / 2048 }).sorted()
        for tile in tiles where blobs[tile] == nil { throw QwenANEEmbeddingError.missingTile(tile) }
        lock.lock(); defer { lock.unlock() }
        var result = [Float16](repeating: 0, count: 32 * 5120)
        for tile in tiles {
            try autoreleasepool {
                if programs[tile] == nil {
                    if programs.count == limit {
                        let evicted = recent.removeFirst()
                        programs.removeValue(forKey: evicted)
                    }
                    guard let blob = blobs[tile], let digest = digests[tile] else { throw QwenANEEmbeddingError.missingTile(tile) }
                    do {
                        programs[tile] = try ANEDirectDispatch.FP16Loop(milText: Self.mil(programTag: digest),
                            weightBlob: blob, inputDim: 1, outputDim: 5120, batch: 32)
                    } catch {
                        throw QwenANEEmbeddingError.programLoadFailed(tile: tile,
                            retainedTiles: programs.keys.sorted(), underlying: String(describing: error))
                    }
                }
                recent.removeAll { $0 == tile }; recent.append(tile)
                guard let program = programs[tile] else { throw QwenANEEmbeddingError.missingTile(tile) }
                let indices = tokens.map { Float16($0 / 2048 == tile ? $0 % 2048 : -1) }
                let selected = try program(indices)
                for lane in 0..<32 where tokens[lane] / 2048 == tile {
                    let range = (lane * 5120)..<((lane + 1) * 5120)
                    result.replaceSubrange(range, with: selected[range])
                }
            }
        }
        return result
    }

    private static func mil(programTag: String) -> String {
        var text = buildConvMILText(inputDim: 2048, outputDim: 5120, sequenceLength: 32, programTag: programTag)
        let signature = "func main<ios18>(tensor<fp16, [1, 2048, 1, 32]> x) {"
        precondition(text.contains(signature))
        let indices = (0..<2048).map(String.init).joined(separator: ", ")
        let preparation = """
        func main<ios18>(tensor<fp16, [1, 1, 1, 32]> x) {
          tensor<fp16, [2048]> indices = const()[name=string("indices"), val=tensor<fp16, [2048]>([\(indices)])];
          tensor<int32, [4]> indexShape = const()[name=string("indexShape"), val=tensor<int32, [4]>([1, 2048, 1, 1])];
          tensor<fp16, [1, 2048, 1, 1]> rows = reshape(x=indices, shape=indexShape)[name=string("rows")];
          fp16 one = const()[name=string("one"), val=fp16(1.0)];
          fp16 zero = const()[name=string("zero"), val=fp16(0.0)];
          tensor<bool, [1, 2048, 1, 32]> match = equal(x=x, y=rows)[name=string("match")];
          tensor<fp16, [1, 2048, 1, 32]> selector = select(cond=match, a=one, b=zero)[name=string("selector")];
        """
        text = text.replacingOccurrences(of: signature, with: preparation)
            .replacingOccurrences(of: "weight=w, x=x)", with: "weight=w, x=selector)")
        return text
    }
}
