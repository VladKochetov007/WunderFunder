/* glibc >= 2.38 headers redirect strtoul() to __isoc23_strtoul in C23/_GNU_SOURCE builds, and the
 * static libstdc++ of the build host references it. The scorer image (Debian bookworm) has glibc
 * 2.36, which lacks that symbol. This file is compiled as plain C11 (no redirect), so the definition
 * below forwards to the classic strtoul@GLIBC_2.2.5; it is hidden, so it only satisfies references
 * inside this module. */
#include <stdlib.h>

__attribute__((visibility("hidden"))) unsigned long __isoc23_strtoul(const char* s, char** end, int base) {
  return strtoul(s, end, base);
}
