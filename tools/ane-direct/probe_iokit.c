// Probe: can an ordinary unprivileged process open the ANE kernel user client?
#include <stdio.h>
#include <IOKit/IOKitLib.h>

static void try_open(const char *cls) {
  io_iterator_t it = 0;
  kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault,
                                                  IOServiceMatching(cls), &it);
  if (kr != KERN_SUCCESS) { printf("%-24s match kr=0x%x\n", cls, kr); return; }
  io_service_t svc; int n = 0;
  while ((svc = IOIteratorNext(it))) {
    n++;
    io_name_t name; IORegistryEntryGetName(svc, name);
    for (int type = 0; type <= 8; type++) {
      io_connect_t conn = 0;
      kern_return_t o = IOServiceOpen(svc, mach_task_self(), type, &conn);
      printf("%-24s svc=%s type=%d IOServiceOpen kr=0x%08x %s\n",
             cls, name, type, o, o == KERN_SUCCESS ? "OPENED" : "");
      if (o == KERN_SUCCESS) IOServiceClose(conn);
    }
    IOObjectRelease(svc);
  }
  if (!n) printf("%-24s no matching service\n", cls);
  IOObjectRelease(it);
}

int main(void) {
  const char *classes[] = {"H11ANEIn", "AppleT6041ANEHAL", "H1xANELoadBalancer",
                           "AppleH16ANE", "ANEUserClient", NULL};
  for (int i = 0; classes[i]; i++) try_open(classes[i]);
  return 0;
}
