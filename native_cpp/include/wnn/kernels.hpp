// Dense float32 kernels for one-row inference, written so GCC vectorises them without -ffast-math.
//
// gemv:   row-major W (out x in). Each dot product accumulates into kLanes independent lanes (a plain
//         in-order float sum per lane, so vectorising across lanes needs no reassociation), and four
//         rows are processed together so every load of x feeds four FMA chains.
// saxpy:  transposed W (in x out) for sparse inputs: y += x[i] * Wt[i, :], zero x[i] skipped. Used for
//         the input projection, where most event features are zero on most rows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define WNN_HAVE_F16 1
#else
#define WNN_HAVE_F16 0
#endif

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


// ---- half-precision weight storage (x86-64-v3: F16C + AVX2 + FMA) ------------------------------
// Weights stored as IEEE fp16 halve the per-row weight stream; products and sums stay float32.
using HalfVec = std::vector<std::uint16_t, AlignedAllocator<std::uint16_t>>;

inline bool half_supported() { return WNN_HAVE_F16 != 0; }

#if WNN_HAVE_F16
inline HalfVec to_half(const float* v, std::size_t n) {
  HalfVec h((n + 15) / 16 * 16, 0);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(h.data() + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(v + i), _MM_FROUND_TO_NEAREST_INT));
  for (; i < n; ++i) h[i] = static_cast<std::uint16_t>(_cvtss_sh(v[i], _MM_FROUND_TO_NEAREST_INT));
  return h;
}

inline __m256 load_h8(const std::uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

inline float hsum8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// y[o] = b[o] + W[o, :] . x with fp16 row-major W; requires in % 16 == 0.
inline void gemv_half(const std::uint16_t* __restrict W, const float* __restrict b, const float* __restrict x,
                      float* __restrict y, int out, int in) {
  int o = 0;
  for (; o + 4 <= out; o += 4) {
    const std::uint16_t* w0 = W + static_cast<std::size_t>(o) * in;
    const std::uint16_t* w1 = w0 + in;
    const std::uint16_t* w2 = w1 + in;
    const std::uint16_t* w3 = w2 + in;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    for (int i = 0; i < in; i += 16) {
      const __m256 x0 = _mm256_loadu_ps(x + i), x1 = _mm256_loadu_ps(x + i + 8);
      a0 = _mm256_fmadd_ps(load_h8(w0 + i), x0, a0);
      c0 = _mm256_fmadd_ps(load_h8(w0 + i + 8), x1, c0);
      a1 = _mm256_fmadd_ps(load_h8(w1 + i), x0, a1);
      c1 = _mm256_fmadd_ps(load_h8(w1 + i + 8), x1, c1);
      a2 = _mm256_fmadd_ps(load_h8(w2 + i), x0, a2);
      c2 = _mm256_fmadd_ps(load_h8(w2 + i + 8), x1, c2);
      a3 = _mm256_fmadd_ps(load_h8(w3 + i), x0, a3);
      c3 = _mm256_fmadd_ps(load_h8(w3 + i + 8), x1, c3);
    }
    y[o] = hsum8(_mm256_add_ps(a0, c0)) + b[o];
    y[o + 1] = hsum8(_mm256_add_ps(a1, c1)) + b[o + 1];
    y[o + 2] = hsum8(_mm256_add_ps(a2, c2)) + b[o + 2];
    y[o + 3] = hsum8(_mm256_add_ps(a3, c3)) + b[o + 3];
  }
  for (; o < out; ++o) {
    const std::uint16_t* w = W + static_cast<std::size_t>(o) * in;
    __m256 a = _mm256_setzero_ps();
    for (int i = 0; i < in; i += 8) a = _mm256_fmadd_ps(load_h8(w + i), _mm256_loadu_ps(x + i), a);
    y[o] = hsum8(a) + b[o];
  }
}
#endif

// ---- int8 weight storage with one float scale per output row (x86-64-v3) -------------------------
using Int8Vec = std::vector<std::int8_t, AlignedAllocator<std::int8_t>>;

// Symmetric per-row quantisation: W[o, :] ~= scale[o] * q[o, :], q in [-127, 127].
inline void to_int8(const float* W, int out, int in, Int8Vec& q, std::vector<float>& scale) {
  q.assign(static_cast<std::size_t>(out) * in, 0);
  scale.assign(out, 0.f);
  for (int o = 0; o < out; ++o) {
    const float* w = W + static_cast<std::size_t>(o) * in;
    float m = 0.f;
    for (int i = 0; i < in; ++i) m = std::max(m, std::fabs(w[i]));
    const float s = m > 0.f ? m / 127.f : 1.f;
    scale[o] = s;
    for (int i = 0; i < in; ++i) q[static_cast<std::size_t>(o) * in + i] = static_cast<std::int8_t>(std::lrint(w[i] / s));
  }
}

#if WNN_HAVE_F16
inline __m256 load_i8x8(const std::int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

// y[o] = b[o] + scale[o] * (q[o, :] . x); requires in % 16 == 0.
inline void gemv_int8(const std::int8_t* __restrict Q, const float* __restrict scale, const float* __restrict b,
                      const float* __restrict x, float* __restrict y, int out, int in) {
  int o = 0;
  for (; o + 4 <= out; o += 4) {
    const std::int8_t* q0 = Q + static_cast<std::size_t>(o) * in;
    const std::int8_t* q1 = q0 + in;
    const std::int8_t* q2 = q1 + in;
    const std::int8_t* q3 = q2 + in;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    for (int i = 0; i < in; i += 16) {
      const __m256 x0 = _mm256_loadu_ps(x + i), x1 = _mm256_loadu_ps(x + i + 8);
      a0 = _mm256_fmadd_ps(load_i8x8(q0 + i), x0, a0);
      c0 = _mm256_fmadd_ps(load_i8x8(q0 + i + 8), x1, c0);
      a1 = _mm256_fmadd_ps(load_i8x8(q1 + i), x0, a1);
      c1 = _mm256_fmadd_ps(load_i8x8(q1 + i + 8), x1, c1);
      a2 = _mm256_fmadd_ps(load_i8x8(q2 + i), x0, a2);
      c2 = _mm256_fmadd_ps(load_i8x8(q2 + i + 8), x1, c2);
      a3 = _mm256_fmadd_ps(load_i8x8(q3 + i), x0, a3);
      c3 = _mm256_fmadd_ps(load_i8x8(q3 + i + 8), x1, c3);
    }
    y[o] = scale[o] * hsum8(_mm256_add_ps(a0, c0)) + b[o];
    y[o + 1] = scale[o + 1] * hsum8(_mm256_add_ps(a1, c1)) + b[o + 1];
    y[o + 2] = scale[o + 2] * hsum8(_mm256_add_ps(a2, c2)) + b[o + 2];
    y[o + 3] = scale[o + 3] * hsum8(_mm256_add_ps(a3, c3)) + b[o + 3];
  }
  for (; o < out; ++o) {
    const std::int8_t* q = Q + static_cast<std::size_t>(o) * in;
    __m256 a = _mm256_setzero_ps();
    for (int i = 0; i < in; i += 8) a = _mm256_fmadd_ps(load_i8x8(q + i), _mm256_loadu_ps(x + i), a);
    y[o] = scale[o] * hsum8(a) + b[o];
  }
}
#endif

}  // namespace wnn::kernels
