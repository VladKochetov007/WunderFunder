// Activation functions: exact (libm) and fast branchless approximations that vectorise.
//
// The fast forms are the ones of the production Rust engine (native/src/approx.rs): exp as 2^n e^r
// with a degree-7 Taylor core, sigmoid/tanh from it, erf by Abramowitz & Stegun 7.1.26. Errors on
// [-12, 12]: sigmoid < 2e-6, tanh < 4e-6, GELU < 5e-6 (pinned by tests/test_activations.py).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace wnn::act {

inline float exp_fast(float x) {
  constexpr float kLn2 = 0.693147180559945309f;
  constexpr float kInvLn2 = 1.442695040888963407f;
  x = x < -40.f ? -40.f : (x > 40.f ? 40.f : x);
  // round to nearest (ties to even) by the 1.5 * 2^23 magic constant: plain float adds, so it
  // vectorises (std::round's ties-away mode has no single SIMD instruction). |x/ln2| <= 58 here.
  constexpr float kMagic = 12582912.0f;
  const float n = (x * kInvLn2 + kMagic) - kMagic;
  const float r = x - n * kLn2;
  const float p =
      1.f + r * (1.f + r * (0.5f + r * (1.f / 6 + r * (1.f / 24 + r * (1.f / 120 + r * (1.f / 720 + r * (1.f / 5040)))))));
  const std::int32_t bits = (static_cast<std::int32_t>(n) + 127) << 23;  // |n| <= 58: normal range
  float pow2;
  std::memcpy(&pow2, &bits, sizeof pow2);
  return p * pow2;
}

inline float sigmoid_fast(float x) { return 1.f / (1.f + exp_fast(-x)); }
inline float tanh_fast(float x) { return 2.f * sigmoid_fast(2.f * x) - 1.f; }

inline float erf_fast(float x) {
  const float a = std::fabs(x);
  const float t = 1.f / (1.f + 0.3275911f * a);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float v = 1.f - poly * exp_fast(-a * a);
  return std::copysign(v, x);
}

inline float gelu_fast(float x) { return 0.5f * x * (1.f + erf_fast(x * 0.70710678118654752f)); }

inline float sigmoid_exact(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float gelu_exact(float x) { return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752f)); }

}  // namespace wnn::act
