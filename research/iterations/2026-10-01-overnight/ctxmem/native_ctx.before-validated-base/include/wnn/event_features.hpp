// Streaming causal event features: C++ twin of wnn.features.StreamingEventFeatures.
//
// Layout of the 99 outputs (wnn.features.FEATURE_NAMES):
//   [0, 22)   d_i0px   change of the 22 i0 price codes vs the previous row
//   [22, 44)  d_i0vol  change of the 22 i0 volume codes
//   [44, 66)  d_i1px   change of the 22 i1 price codes
//   [66, 74)  eight event flags (i0 joint bid+ask price move, i0 bid vol, i0 ask vol, i0 trade price,
//             i0 trade volume, i1 bid-or-ask price move, i1 volume, i1 trade volume)
//   [74, 77)  log1p(min(rows since event, 10000)) for i0 price, i0 trade volume, i1 price events
//   [77, 99)  d_i0px held from the last i0 price event
#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace wnn {

inline constexpr int kRaw = 112;
inline constexpr int kEvent = 99;
inline constexpr int kInputs = kRaw + kEvent;  // model input width

class EventFeatures {
 public:
  EventFeatures() { reset(); }

  void reset() {
    has_prev_ = false;
    step_ = 0;
    last_ = {0, 0, 0};
    held_.fill(0.0f);
  }

  // Writes the 99 features for `row` (112 floats) into `out` and advances the state.
  void update(const float* row, float* out) {
    const float* prev = has_prev_ ? prev_.data() : row;
    // i0 block starts at 0, i1 block at 52; inside a block: bid px 0-10, ask px 11-21,
    // bid vol 22-32, ask vol 33-43, trade px 44-47, trade vol 48-51.
    auto ne = [&](int j) { return prev[j] != row[j]; };
    auto any_ne = [&](int a, int b) {
      for (int j = a; j < b; ++j)
        if (prev[j] != row[j]) return true;
      return false;
    };
    const bool f_i0px = ne(0) && ne(11);
    const bool flags[8] = {f_i0px,        any_ne(22, 33),  any_ne(33, 44),       any_ne(44, 48),
                           any_ne(48, 52), ne(52) || ne(63), any_ne(74, 96), any_ne(100, 104)};
    if (flags[0]) last_[0] = step_;
    if (flags[4]) last_[1] = step_;
    if (flags[5]) last_[2] = step_;

    for (int j = 0; j < 22; ++j) out[j] = row[j] - prev[j];
    for (int j = 0; j < 22; ++j) out[22 + j] = row[22 + j] - prev[22 + j];
    for (int j = 0; j < 22; ++j) out[44 + j] = row[52 + j] - prev[52 + j];
    if (flags[0])
      for (int j = 0; j < 22; ++j) held_[j] = out[j];
    for (int k = 0; k < 8; ++k) out[66 + k] = flags[k] ? 1.0f : 0.0f;
    for (int k = 0; k < 3; ++k) {
      const std::int64_t since = step_ - last_[k];
      out[74 + k] = std::log1p(static_cast<float>(since < 10000 ? since : 10000));
    }
    for (int j = 0; j < 22; ++j) out[77 + j] = held_[j];

    for (int j = 0; j < kRaw; ++j) prev_[j] = row[j];
    has_prev_ = true;
    ++step_;
  }

 private:
  std::array<float, kRaw> prev_{};
  bool has_prev_ = false;
  std::int64_t step_ = 0;
  std::array<std::int64_t, 3> last_{};
  std::array<float, 22> held_{};
};

}  // namespace wnn
