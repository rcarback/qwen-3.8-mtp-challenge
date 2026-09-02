// ANECCompile feasibility probe (throwaway reconnaissance, not shipped code).
//
// Pipeline, reverse-engineered from freedomtan/coreml_to_ane_hwx
// (https://github.com/freedomtan/coreml_to_ane_hwx, coreml_util.m /
// coreml_util.h) and cross-checked against mdaiter/ane's ANECompiler
// teardown:
//
//   1. A real, already-compiled CoreML model.espresso.net (Espresso IR,
//      *not* MIL) is loaded through libEspresso's private plan API:
//        espresso_create_context -> espresso_create_plan
//        -> espresso_plan_add_network(path) -> espresso_plan_build
//        -> espresso_dump_ir(&outDir)
//      This produces an IR dump directory containing "net.plist" (plus
//      sibling weight files) -- this is a DIFFERENT, ANE-compiler-specific
//      serialization from model.espresso.net itself.
//   2. That "net.plist" is what ANECCompile actually wants, referenced via
//      an InputNetworks dict of {NetworkPlistName, NetworkPlistPath}.
//
// We source model.espresso.net from a real on-disk system model
// (AltruisticBodyPoseKit's 2DHumanPoseDetectorFull.mlmodelc) rather than
// hand-building a .mlmodel, since it already ships model.espresso.net
// directly -- no CoreML compileModelAtURL step needed.
//
// Build: clang -fobjc-arc -framework Foundation -o compile_probe compile_probe.m
// Run:   ./compile_probe [path/to/model.espresso.net]

#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <sys/stat.h>

typedef unsigned int ANECStatus;
typedef int (*espresso_create_context_t)(uint64_t, uint64_t, void **);
typedef int (*espresso_create_plan_t)(void *, uint64_t, void **);
typedef int (*espresso_plan_add_network_t)(void *, const char *, uint64_t, uint64_t[2]);
typedef int (*espresso_plan_build_t)(void *);
typedef int (*espresso_dump_ir_t)(void *, char **);
typedef int (*espresso_plan_destroy_t)(void *);
typedef int (*espresso_context_destroy_t)(void *);

typedef int (*ANECCompile_t)(NSDictionary *, NSDictionary *,
                              void (^)(ANECStatus, NSDictionary *));

static void *mustDlsym(void *handle, const char *name, const char *lib) {
  void *sym = dlsym(handle, name);
  if (!sym) {
    NSLog(@"FATAL: dlsym(%s) in %s failed: %s", name, lib, dlerror());
  } else {
    NSLog(@"  dlsym ok: %s = %p", name, sym);
  }
  return sym;
}

int main(int argc, char *argv[]) {
  @autoreleasepool {
    const char *defaultModel =
        "/System/Library/PrivateFrameworks/AltruisticBodyPoseKit.framework/"
        "2DHumanPoseDetectorFull.mlmodelc/model.espresso.net";
    const char *espressoNetPath = (argc > 1) ? argv[1] : defaultModel;

    NSLog(@"=== ANECCompile feasibility probe ===");
    NSLog(@"model.espresso.net: %s", espressoNetPath);
    struct stat st;
    if (stat(espressoNetPath, &st) != 0) {
      NSLog(@"FATAL: input file does not exist / not readable: %s", espressoNetPath);
      return 2;
    }
    NSLog(@"  size: %lld bytes", (long long)st.st_size);

    NSLog(@"--- dlopen frameworks ---");
    void *espressoHandle = dlopen(
        "/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
        RTLD_NOW);
    if (!espressoHandle) {
      NSLog(@"FATAL: dlopen(Espresso) failed: %s", dlerror());
      return 3;
    }
    NSLog(@"  dlopen(Espresso) ok: %p", espressoHandle);

    void *aneCompilerHandle = dlopen(
        "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
        RTLD_NOW);
    if (!aneCompilerHandle) {
      NSLog(@"FATAL: dlopen(ANECompiler) failed: %s", dlerror());
      return 3;
    }
    NSLog(@"  dlopen(ANECompiler) ok: %p", aneCompilerHandle);

    NSLog(@"--- dlsym libEspresso plan API ---");
    espresso_create_context_t p_create_context =
        (espresso_create_context_t)mustDlsym(espressoHandle, "espresso_create_context", "Espresso");
    espresso_create_plan_t p_create_plan =
        (espresso_create_plan_t)mustDlsym(espressoHandle, "espresso_create_plan", "Espresso");
    espresso_plan_add_network_t p_add_network =
        (espresso_plan_add_network_t)mustDlsym(espressoHandle, "espresso_plan_add_network", "Espresso");
    espresso_plan_build_t p_build =
        (espresso_plan_build_t)mustDlsym(espressoHandle, "espresso_plan_build", "Espresso");
    espresso_dump_ir_t p_dump_ir =
        (espresso_dump_ir_t)mustDlsym(espressoHandle, "espresso_dump_ir", "Espresso");
    espresso_plan_destroy_t p_plan_destroy =
        (espresso_plan_destroy_t)dlsym(espressoHandle, "espresso_plan_destroy");
    espresso_context_destroy_t p_ctx_destroy =
        (espresso_context_destroy_t)dlsym(espressoHandle, "espresso_context_destroy");

    if (!p_create_context || !p_create_plan || !p_add_network || !p_build || !p_dump_ir) {
      NSLog(@"FATAL: missing required libEspresso symbols, cannot proceed to IR dump stage");
      return 4;
    }

    ANECCompile_t p_ANECCompile =
        (ANECCompile_t)mustDlsym(aneCompilerHandle, "ANECCompile", "ANECompiler");
    if (!p_ANECCompile) {
      NSLog(@"FATAL: ANECCompile not resolvable, aborting");
      return 4;
    }

    NSLog(@"--- stage 1: model.espresso.net -> Espresso IR dump (net.plist) ---");
    // NOTE: the known-good call signature for these functions is inferred
    // from coreml_util.m's Objective-C call sites, which pass NSString/const
    // char* etc. through ARC-managed calls without explicit prototypes (the
    // header there declares plain `void*`/`char*`/`uint64_t` args and lets
    // clang do implicit int-returning calls). We mirror that here as best
    // effort; if the real ABI differs (e.g. return via first arg, or a
    // different arg count) this stage will crash or misbehave rather than
    // cleanly fail -- noted as a risk in the report.
    void *ctx = NULL;
    // coreml_util.m calls this as `void* ctx = espresso_create_context(0x2718LL, 0xFFFFFFFFLL);`
    // i.e. a 2-arg function returning a pointer directly, not out-param style.
    typedef void *(*espresso_create_context_ptr_t)(uint64_t, uint64_t);
    espresso_create_context_ptr_t p_create_context_ptr =
        (espresso_create_context_ptr_t)p_create_context;
    ctx = p_create_context_ptr(0x2718LL, 0xFFFFFFFFLL);
    NSLog(@"  espresso_create_context -> ctx=%p", ctx);
    if (!ctx) {
      NSLog(@"FATAL: espresso_create_context returned NULL");
      return 5;
    }

    typedef void *(*espresso_create_plan_ptr_t)(void *, uint64_t);
    espresso_create_plan_ptr_t p_create_plan_ptr =
        (espresso_create_plan_ptr_t)p_create_plan;
    void *plan = p_create_plan_ptr(ctx, 0LL);
    NSLog(@"  espresso_create_plan -> plan=%p", plan);
    if (!plan) {
      NSLog(@"FATAL: espresso_create_plan returned NULL");
      return 5;
    }

    uint64_t vals[2] = {0, 0};
    int ret = p_add_network(plan, espressoNetPath, 0x10010LL, vals);
    NSLog(@"  espresso_plan_add_network -> ret=%d vals=[%llu, %llu]", ret,
          (unsigned long long)vals[0], (unsigned long long)vals[1]);
    if (ret) {
      NSLog(@"FATAL: espresso_plan_add_network failed with ret=%d", ret);
      return 6;
    }

    ret = p_build(plan);
    NSLog(@"  espresso_plan_build -> ret=%d", ret);
    if (ret) {
      NSLog(@"FATAL: espresso_plan_build failed with ret=%d", ret);
      return 6;
    }

    const char *irDumpDir = "/tmp/ane-hwx-probe/espresso_ir_dump/";
    mkdir("/tmp/ane-hwx-probe", 0755);
    mkdir(irDumpDir, 0755);
    char *irDumpDirBuf = strdup(irDumpDir);
    ret = p_dump_ir(plan, &irDumpDirBuf);
    NSLog(@"  espresso_dump_ir -> ret=%d dir=%s", ret, irDumpDirBuf ? irDumpDirBuf : "(null)");
    if (ret) {
      NSLog(@"FATAL: espresso_dump_ir failed with ret=%d", ret);
      return 6;
    }

    NSString *irDir = [NSString stringWithUTF8String:irDumpDirBuf];
    NSError *lsErr = nil;
    NSArray *dumped = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:irDir error:&lsErr];
    NSLog(@"  IR dump directory listing (%@): %@", irDir, dumped);

    if (p_plan_destroy) p_plan_destroy(plan);
    if (p_ctx_destroy) p_ctx_destroy(ctx);

    if (![dumped containsObject:@"net.plist"]) {
      NSLog(@"WARNING: net.plist not found in IR dump dir; ANECCompile stage will likely fail");
    }

    NSLog(@"--- stage 2: net.plist -> ANECCompile -> model.hwx ---");
    const char *outputBase = "/tmp/ane-hwx-probe/hwx_output/";
    mkdir(outputBase, 0755);

    NSDictionary *inputNetworkEntry = @{
      @"NetworkPlistName" : @"net.plist",
      @"NetworkPlistPath" : irDir,
    };
    NSArray *inputNetworks = @[ inputNetworkEntry ];

    NSMutableDictionary *optionsDictionary = [NSMutableDictionary dictionaryWithCapacity:4];
    NSMutableDictionary *flagsDictionary = [NSMutableDictionary dictionaryWithCapacity:4];
    optionsDictionary[@"InputNetworks"] = inputNetworks;
    optionsDictionary[@"OutputFilePath"] = [NSString stringWithUTF8String:outputBase];
    optionsDictionary[@"OutputFileName"] = @"model.hwx";

    // Target architecture codename attempts, in order. h13 is the value
    // used in every published example (M1-era); M5 is a newer generation so
    // we try a small set of plausible successors too. Comment in the
    // published examples claims "h11 (or anything?) works here too, and
    // creates different outputs that don't run" -- i.e. the flag may be
    // permissive but produce a non-loadable hwx for the wrong generation.
    NSArray<NSString *> *archCandidates = @[ @"h13", @"h17", @"h16", @"h15", @"h14" ];

    for (NSString *arch in archCandidates) {
      flagsDictionary[@"TargetArchitecture"] = arch;
      NSLog(@"  attempting TargetArchitecture=%@", arch);

      __block ANECStatus blockStatus = 0xffffffff;
      __block NSDictionary *blockStatusDict = nil;
      __block BOOL called = NO;
      void (^completion)(ANECStatus, NSDictionary *) =
          ^(ANECStatus status, NSDictionary *statusDictionary) {
            blockStatus = status;
            blockStatusDict = statusDictionary;
            called = YES;
          };

      int compileRet = p_ANECCompile(optionsDictionary, flagsDictionary, completion);
      NSLog(@"  ANECCompile(arch=%@) -> ret=%d, callback_called=%d, status=0x%x",
            arch, compileRet, called, blockStatus);
      if (blockStatusDict) {
        NSLog(@"  statusDictionary: %@", blockStatusDict);
      }

      NSString *hwxPath = [NSString stringWithFormat:@"%s%@", outputBase, @"model.hwx"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:hwxPath]) {
        NSLog(@"  *** SUCCESS: hwx produced at %@ ***", hwxPath);
        NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:hwxPath error:nil];
        NSLog(@"  hwx size: %@ bytes", attrs[NSFileSize]);
        return 0;
      }
    }

    NSLog(@"=== No hwx produced for any tried TargetArchitecture value ===");
    return 1;
  }
}
