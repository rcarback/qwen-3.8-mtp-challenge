// Open the direct-path client and probe each of the 9 selectors with correctly-sized
// zeroed structs to see which are reachable vs entitlement/stub-gated.
#include <stdio.h>
#include <string.h>
#include <IOKit/IOKitLib.h>

int main(void){
  io_iterator_t it; io_service_t svc;
  IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("H11ANEIn"), &it);
  svc = IOIteratorNext(it);
  io_connect_t c=0;
  kern_return_t o = IOServiceOpen(svc, mach_task_self(), 1, &c);
  printf("open kr=0x%08x\n", o);
  if (o) return 1;

  // Per paper Table 27.4: struct-in sizes for sels 0..8.
  size_t sin[9]  = {104, 0, 2376, 40, 3104, 2080, 2080, 16, 32};
  size_t sout[9] = {104, 0, 40, 0, 0, 0, 0, 24, 0};
  int scin[9]    = {0,0,1,0,0,1,0,0,0};
  int scout[9]   = {0,0,0,0,0,0,1,0,0};

  for (int s=0;s<9;s++){
    unsigned char in[4096]; memset(in,0,sizeof in);
    unsigned char out[4096]; size_t osz=sout[s];
    uint64_t sci[4]={0,0,0,0}; uint64_t sco[4]={0,0,0,0}; uint32_t scoc=scout[s];
    kern_return_t kr = IOConnectCallMethod(c, s,
        scin[s]?sci:NULL, scin[s], sin[s]?in:NULL, sin[s],
        scout[s]?sco:NULL, scout[s]?&scoc:NULL,
        sout[s]?out:NULL, sout[s]?&osz:NULL);
    printf("sel %d kr=0x%08x\n", s, kr);
  }
  IOServiceClose(c);
  return 0;
}
