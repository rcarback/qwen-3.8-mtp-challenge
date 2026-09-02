#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// Reachability probe for maderix's private in-memory compile entry point.
// We do NOT need a valid MIL here: we want to know (a) is the class callable
// from an unentitled process, and (b) how far does compileWithQoS get before
// it blocks (entitlement? signature? compiler rejection?).
int main() {
    @autoreleasepool {
        void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        printf("dlopen: %s\n", h ? "OK" : dlerror());

        Class Desc = objc_getClass("_ANEInMemoryModelDescriptor");
        Class Mem  = objc_getClass("_ANEInMemoryModel");
        Class Cli  = objc_getClass("_ANEClient");
        printf("classes: Desc=%p Mem=%p Client=%p\n", Desc, Mem, Cli);

        // Try instantiating a descriptor with a tiny bogus MIL text + empty weights.
        // modelWithMILText:weights:optionsPlist:
        NSData *mil = [@"program(1.0){ func main() {} }" dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *weights = @{};
        NSData *opts = [NSData data];
        id desc = ((id(*)(id,SEL,id,id,id))objc_msgSend)(Desc,
            sel_registerName("modelWithMILText:weights:optionsPlist:"), mil, weights, opts);
        printf("descriptor instance: %p  (%s)\n", desc, desc ? object_getClassName(desc) : "nil");
        if (desc) {
            id hid = ((id(*)(id,SEL))objc_msgSend)(desc, sel_registerName("hexStringIdentifier"));
            printf("  hexStringIdentifier: %s\n", hid ? [[hid description] UTF8String] : "nil");
        }

        // Build the in-memory model and attempt compile (unentitled).
        id mem = ((id(*)(id,SEL,id))objc_msgSend)(Mem,
            sel_registerName("inMemoryModelWithDescriptor:"), desc);
        printf("inMemoryModel: %p\n", mem);
        if (mem) {
            NSError *err = nil;
            // compileWithQoS:options:error:  (QoS as NSInteger, options dict)
            BOOL ok = ((BOOL(*)(id,SEL,NSInteger,id,NSError**))objc_msgSend)(mem,
                sel_registerName("compileWithQoS:options:error:"), 0x21, @{}, &err);
            printf("compileWithQoS -> ok=%d err=%s\n", ok,
                   err ? [[err description] UTF8String] : "nil");
        }
    }
    return 0;
}
