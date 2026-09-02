// Hold each openable connection and report the user-client class the kernel vended.
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <IOKit/IOKitLib.h>

int main(int argc, char **argv) {
  int want = atoi(argv[1]);
  io_iterator_t it; io_service_t svc;
  IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("H11ANEIn"), &it);
  svc = IOIteratorNext(it);
  io_connect_t conn = 0;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), want, &conn);
  printf("open type=%d kr=0x%08x conn=%u pid=%d\n", want, kr, conn, getpid());
  fflush(stdout);
  if (kr == KERN_SUCCESS) { sleep(20); IOServiceClose(conn); }
  return 0;
}
