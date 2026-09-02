import Darwin
import Testing

@testable import MLXFastRuntimeWorkerSupport

struct QwenMTPHostQoSPolicyTests {
    @Test("interactive and initiated map to their QoS classes")
    func mapsKnownValues() {
        #expect(QwenMTPHostQoSPolicy.resolve(["MLX_MTP_HOST_QOS": "interactive"])
            == QOS_CLASS_USER_INTERACTIVE)
        #expect(QwenMTPHostQoSPolicy.resolve(["MLX_MTP_HOST_QOS": "initiated"])
            == QOS_CLASS_USER_INITIATED)
    }

    @Test("absent or unknown values leave the class unchanged")
    func ignoresUnknownValues() {
        #expect(QwenMTPHostQoSPolicy.resolve([:]) == nil)
        #expect(QwenMTPHostQoSPolicy.resolve(["MLX_MTP_HOST_QOS": "bogus"]) == nil)
        #expect(QwenMTPHostQoSPolicy.resolve(["MLX_MTP_HOST_QOS": ""]) == nil)
    }
}
