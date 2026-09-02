#include <stdio.h>
#include <dlfcn.h>
int main(void){
  const char *fw[] = {
    "/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices",
    "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
    NULL};
  const char *syms[] = {"ANECCompile","ANECReserveDevice","_ZN14ANECModelParserC1Ev",
                        "ANEProgramCreate","ANEDeviceOpenGated", NULL};
  for(int i=0;fw[i];i++){
    void *h = dlopen(fw[i], RTLD_NOW|RTLD_LOCAL);
    printf("dlopen %-70s %s\n", fw[i], h?"OK":dlerror());
    if(h){ for(int s=0;syms[s];s++){ void*p=dlsym(h,syms[s]); if(p) printf("   sym %s -> %p\n",syms[s],p);} }
  }
  return 0;
}
