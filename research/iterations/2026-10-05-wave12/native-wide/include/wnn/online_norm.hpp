// Causal per-sequence z-scores of the 112 raw features: C++ twin of wnn.features.online_zscore_torch as
// training evaluates it on CUDA (members trained with --online-norm take [raw | event | z] inputs).
//
// With n rows seen (current one included): mean = sum / n, var = max(sumsq / n - mean^2, 0),
// z = (x - mean) / sqrt(var + 1e-4). The sums are float32 and accumulated in row order on purpose:
// CUDA cumsum along time is a sequential float32 loop per column, so the model was trained on that
// rounding. Every op is one float32 op in torch's order; FMA contraction is disabled for this function
// (a fused x*x + sum would round differently), matching the Rust engine's bit-exact OnlineNorm.
#pragma once

#include <array>
#include <cmath>

#include "wnn/event_features.hpp"

namespace wnn {

inline constexpr float kOnlineEps = 1e-4f;

class OnlineNorm {
 public:
  void reset() {
    n_ = 0;
    sum_.fill(0.f);
    sumsq_.fill(0.f);
  }

  __attribute__((optimize("fp-contract=off"))) void update(const float* row, float* out) {
    ++n_;
    const float n = static_cast<float>(n_);  // exact below 2^24 rows, like torch.arange in float32
    for (int c = 0; c < kRaw; ++c) {
      const float x = row[c];
      const float xx = x * x;
      sum_[c] += x;
      sumsq_[c] += xx;
      const float mean = sum_[c] / n;
      const float m2 = mean * mean;
      float var = sumsq_[c] / n - m2;
      var = var > 0.f ? var : 0.f;
      out[c] = (x - mean) / std::sqrt(var + kOnlineEps);
    }
  }

 private:
  unsigned n_ = 0;
  std::array<float, kRaw> sum_{};
  std::array<float, kRaw> sumsq_{};
};

}  // namespace wnn
