// Dense float32 kernels for one-row inference, written so GCC vectorises them without -ffast-math.
//
// gemv:   row-major W (out x in). Each dot product accumulates into kLanes independent lanes (a plain
//         in-order float sum per lane, so vectorising across lanes needs no reassociation), and four
//         rows are processed together so every load of x feeds four FMA chains.
// saxpy:  transposed W (in x out) for sparse inputs: y += x[i] * Wt[i, :], zero x[i] skipped. Used for
//         the input projection, where most event features are zero on most rows.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace wnn::kernels {

inline constexpr int kLanes = 16;  // reduce_lanes assumes 16

// 64-byte aligned storage for weights and scratch.
template <class T>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U>&) {}
  T* allocate(std::size_t n) {
    void* p = std::aligned_alloc(64, ((n * sizeof(T) + 63) / 64) * 64);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) { std::free(p); }
  template <class U>
  bool operator==(const AlignedAllocator<U>&) const { return true; }
  template <class U>
  bool operator!=(const AlignedAllocator<U>&) const { return false; }
};
using Vec = std::vector<float, AlignedAllocator<float>>;

// Row-major W (out x in) -> transposed (in x out).
inline Vec transpose(const float* W, int out, int in) {
  if (out <= 0 || in <= 0) throw std::invalid_argument("transpose: empty matrix");
  Vec t(static_cast<std::size_t>(out) * static_cast<std::size_t>(in));
  for (int o = 0; o < out; ++o)
    for (int i = 0; i < in; ++i) t[static_cast<std::size_t>(i) * out + o] = W[static_cast<std::size_t>(o) * in + i];
  return t;
}

// Sum of kLanes accumulators by halving (8+8, 4+4, 2+2, 1+1): the first steps are vector adds.
inline float reduce_lanes(float* acc) {
  for (int k = 0; k < 8; ++k) acc[k] += acc[k + 8];
  for (int k = 0; k < 4; ++k) acc[k] += acc[k + 4];
  for (int k = 0; k < 2; ++k) acc[k] += acc[k + 2];
  return acc[0] + acc[1];
}

inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  float acc[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= n; i += kLanes)
    for (int k = 0; k < kLanes; ++k) acc[k] += a[i + k] * b[i + k];
  float s = reduce_lanes(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

// y[o] = b[o] + W[o, :] . x. Four rows at a time share each load of x.
inline void gemv(const float* __restrict W, const float* __restrict b, const float* __restrict x,
                 float* __restrict y, int out, int in) {
  int o = 0;
  for (; o + 4 <= out; o += 4) {
    const float* w0 = W + static_cast<std::size_t>(o) * in;
    const float* w1 = w0 + in;
    const float* w2 = w1 + in;
    const float* w3 = w2 + in;
    float a0[kLanes] = {}, a1[kLanes] = {}, a2[kLanes] = {}, a3[kLanes] = {};
    int i = 0;
    for (; i + kLanes <= in; i += kLanes) {
      for (int k = 0; k < kLanes; ++k) {
        const float xv = x[i + k];
        a0[k] += w0[i + k] * xv;
        a1[k] += w1[i + k] * xv;
        a2[k] += w2[i + k] * xv;
        a3[k] += w3[i + k] * xv;
      }
    }
    float s0 = reduce_lanes(a0), s1 = reduce_lanes(a1), s2 = reduce_lanes(a2), s3 = reduce_lanes(a3);
    for (; i < in; ++i) {
      s0 += w0[i] * x[i];
      s1 += w1[i] * x[i];
      s2 += w2[i] * x[i];
      s3 += w3[i] * x[i];
    }
    y[o] = s0 + b[o];
    y[o + 1] = s1 + b[o + 1];
    y[o + 2] = s2 + b[o + 2];
    y[o + 3] = s3 + b[o + 3];
  }
  for (; o < out; ++o) y[o] = dot(W + static_cast<std::size_t>(o) * in, x, in) + b[o];
}

// y[o] = b[o] + sum_i Wt[i, o] * x[i], skipping zero x[i]. y may not alias x.
inline void saxpy_sparse(const float* __restrict Wt, const float* __restrict b, const float* __restrict x,
                         float* __restrict y, int out, int in) {
  for (int o = 0; o < out; ++o) y[o] = b[o];
  for (int i = 0; i < in; ++i) {
    const float xi = x[i];
    if (xi == 0.0f) continue;
    const float* __restrict w = Wt + static_cast<std::size_t>(i) * out;
    for (int o = 0; o < out; ++o) y[o] += xi * w[o];
  }
}

}  // namespace wnn::kernels
