// Single-core read bandwidth vs working set and number of interleaved streams.
//   gcc -O3 -march=x86-64-v3 bw.c -o bw && taskset -c 15 ./bw
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

static int cyc_fd = -1;
static uint64_t cycles(void) {
  uint64_t v = 0;
  if (read(cyc_fd, &v, 8) != 8) return 0;
  return v;
}

// Sum `bytes` with `S` interleaved streams (each a contiguous 1/S of the buffer), PF bytes T0
// prefetch ahead (0 = none).
static __m256i run(const char *buf, size_t bytes, int S, size_t pf, int hint) {
  __m256i acc[8];
  for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_si256();
  size_t part = bytes / S;
  for (size_t off = 0; off < part; off += 128) {
    for (int s = 0; s < S; s++) {
      const char *p = buf + s * part + off;
      if (pf) {
        if (hint == 0) { _mm_prefetch(p + pf, _MM_HINT_T0); _mm_prefetch(p + pf + 64, _MM_HINT_T0); }
        else if (hint == 1) { _mm_prefetch(p + pf, _MM_HINT_T1); _mm_prefetch(p + pf + 64, _MM_HINT_T1); }
        else { _mm_prefetch(p + pf, _MM_HINT_T2); _mm_prefetch(p + pf + 64, _MM_HINT_T2); }
      }
      __m256i a = _mm256_load_si256((const __m256i *)p);
      __m256i b = _mm256_load_si256((const __m256i *)(p + 32));
      __m256i c = _mm256_load_si256((const __m256i *)(p + 64));
      __m256i d = _mm256_load_si256((const __m256i *)(p + 96));
      acc[(2*s)&7] = _mm256_add_epi32(acc[(2*s)&7], _mm256_add_epi32(a, b));
      acc[(2*s+1)&7] = _mm256_add_epi32(acc[(2*s+1)&7], _mm256_add_epi32(c, d));
    }
  }
  __m256i t = acc[0];
  for (int i = 1; i < 8; i++) t = _mm256_add_epi32(t, acc[i]);
  return t;
}

int main(int argc, char **argv) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE; attr.size = sizeof attr; attr.config = PERF_COUNT_HW_CPU_CYCLES;
  attr.exclude_kernel = 1; attr.exclude_hv = 1;
  cyc_fd = syscall(298, &attr, 0, -1, -1, 0);
  size_t sizes[] = {512 << 10, 1 << 20, 2200 << 10, 4400 << 10, 16 << 20, 64 << 20};
  int streams[] = {1, 2, 4, 8};
  size_t pfs[] = {0, 1024, 4096};
  char *buf = aligned_alloc(4096, 64 << 20);
  memset(buf, 1, 64 << 20);
  volatile int sink = 0;
  for (int si = 0; si < 6; si++) {
    size_t bytes = sizes[si];
    for (int k = 0; k < 4; k++)
      for (int pi = 0; pi < 3; pi++)
        for (int hint = 0; hint < (pfs[pi] ? 2 : 1); hint++) {
          int S = streams[k];
          int reps = (int)((256u << 20) / bytes);
          uint64_t best = UINT64_MAX;
          for (int r = 0; r < 3; r++) {
            run(buf, bytes, S, pfs[pi], hint);
            uint64_t c0 = cycles();
            for (int i = 0; i < reps; i++) {
              __m256i t = run(buf, bytes, S, pfs[pi], hint);
              sink += _mm256_extract_epi32(t, 0);
            }
            uint64_t c = cycles() - c0;
            if (c < best) best = c;
          }
          printf("size %6zu KB streams %d pf %5zu hint T%d: %5.1f B/cyc\n", bytes >> 10, S, pfs[pi], hint,
                 (double)bytes * reps / best);
        }
  }
  return 0;
}
