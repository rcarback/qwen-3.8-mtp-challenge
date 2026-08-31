import Foundation
import ObjectiveC
import Testing
@testable import MLXFastModel

@Suite(.serialized)
struct ANERuntimeBridgeTests {
    @Test("all five private ANE classes resolve in-process")
    func classesResolve() {
        #expect(ANERuntime.available())
        for n in ["_ANEInMemoryModelDescriptor", "_ANEInMemoryModel",
                  "_ANEClient", "_ANERequest", "_ANEIOSurfaceObject"] {
            #expect(ANERuntime.cls(n) != nil, "\(n) missing")
        }
    }

    @Test("descriptor round-trips a hexStringIdentifier via the msgSend shims")
    func descriptorHexId() throws {
        let Desc = try #require(ANERuntime.cls("_ANEInMemoryModelDescriptor"))
        let mil = "program(1.0){ func main() {} }".data(using: .utf8)! as NSData
        let desc = ANERuntime.send(Desc, Selector(("modelWithMILText:weights:optionsPlist:")),
                                   mil, NSDictionary(), NSData())
        #expect(desc != nil)
        let hid = ANERuntime.send(desc, Selector(("hexStringIdentifier"))) as? NSString
        #expect((hid?.length ?? 0) > 0)
    }
}
