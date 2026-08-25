import Foundation
import Testing

@testable import MLXFastModel

@Suite(.serialized)
struct KVQuantizationPolicyTests {
    private typealias Policy = Qwen36MTPBlockSession.KVQuantization

    @Test("absent bits variable means no policy")
    func absentMeansNil() {
        #expect(Policy.fromEnvironment([:]) == nil)
        #expect(Policy.fromEnvironment(["DARKBLOOM_KV_QUANT_GROUP": "64"]) == nil)
    }

    @Test("every MLX-supported affine bit width is accepted")
    func acceptedBitWidths() {
        for bits in [2, 3, 4, 5, 6, 8] {
            let policy = Policy.fromEnvironment(
                ["DARKBLOOM_KV_QUANT_BITS": String(bits)])
            #expect(policy?.bits == bits)
        }
        // 7 and 16 are not affine bit widths MLX implements.
        #expect(Policy.fromEnvironment(["DARKBLOOM_KV_QUANT_BITS": "7"]) == nil)
        #expect(Policy.fromEnvironment(["DARKBLOOM_KV_QUANT_BITS": "16"]) == nil)
    }

    @Test("rotation is on by default and switchable off")
    func rotationDefault() {
        #expect(Policy.fromEnvironment(
            ["DARKBLOOM_KV_QUANT_BITS": "4"])?.rotate == true)
        #expect(Policy.fromEnvironment([
            "DARKBLOOM_KV_QUANT_BITS": "4",
            "DARKBLOOM_KV_QUANT_ROTATE": "0",
        ])?.rotate == false)
        #expect(Policy.fromEnvironment([
            "DARKBLOOM_KV_QUANT_BITS": "4",
            "DARKBLOOM_KV_QUANT_ROTATE": "1",
        ])?.rotate == true)
    }

    @Test("a group size that does not divide the head dimension is refused")
    func groupValidation() {
        #expect(Policy.fromEnvironment([
            "DARKBLOOM_KV_QUANT_BITS": "4",
            "DARKBLOOM_KV_QUANT_GROUP": "48",
        ]) == nil)
        #expect(Policy.fromEnvironment([
            "DARKBLOOM_KV_QUANT_BITS": "4",
            "DARKBLOOM_KV_QUANT_GROUP": "32",
        ])?.groupSize == 32)
    }
}
