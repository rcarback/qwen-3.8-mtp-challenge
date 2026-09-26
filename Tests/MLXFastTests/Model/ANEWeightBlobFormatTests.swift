import Foundation
import Testing
@testable import MLXLLM

struct ANEWeightBlobFormatTests {
    @Test(arguments: [0, 2, 65538, 2048 * 5120 * 2])
    func preservesHeaderAndPayload(count: Int) throws {
        var payload = Data(repeating: 0xA5, count: count)
        let blob = buildConvWeightBlob(payload)
        #expect(blob.count == 128 + count)
        #expect(blob.dropFirst(128) == payload)
        let bytes = Array(blob.prefix(128))
        #expect(bytes[0] == 1 && bytes[4] == 2)
        #expect(Array(bytes[64..<69]) == [0xEF, 0xBE, 0xAD, 0xDE, 1])
        let size = (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[72 + $1]) << ($1 * 8) }
        let offset = (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[80 + $1]) << ($1 * 8) }
        #expect(size == UInt32(count) && offset == 128)
        let fields = Set([0, 4] + Array(64..<69) + Array(72..<76) + Array(80..<84))
        #expect(bytes.indices.filter { !fields.contains($0) }.allSatisfy { bytes[$0] == 0 })
        if count > 0 {
            payload[0] = 0
            #expect(blob[128] == 0xA5)
        }
    }
}
