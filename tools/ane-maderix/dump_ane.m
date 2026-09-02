#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static void dumpClass(const char *name) {
    Class c = objc_getClass(name);
    printf("\n===== %s (%s) =====\n", name, c ? "FOUND" : "MISSING");
    if (!c) return;
    unsigned int n=0;
    Method *m = class_copyMethodList(c, &n);
    for (unsigned i=0;i<n;i++) printf("  - [%s %s]\n", name, sel_getName(method_getName(m[i])));
    free(m);
    // metaclass (class methods)
    Method *cm = class_copyMethodList(object_getClass(c), &n);
    for (unsigned i=0;i<n;i++) printf("  + [%s %s]\n", name, sel_getName(method_getName(cm[i])));
    free(cm);
    unsigned int iv=0;
    Ivar *ivl = class_copyIvarList(c, &iv);
    for (unsigned i=0;i<iv;i++) printf("    ivar %s : %s\n", ivar_getName(ivl[i]), ivar_getTypeEncoding(ivl[i]));
    free(ivl);
}

int main() {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    printf("dlopen AppleNeuralEngine: %s\n", h ? "OK" : dlerror());
    dumpClass("_ANEInMemoryModelDescriptor");
    dumpClass("_ANEInMemoryModel");
    dumpClass("_ANEModel");
    dumpClass("_ANEClient");
    dumpClass("_ANERequest");
    return 0;
}
