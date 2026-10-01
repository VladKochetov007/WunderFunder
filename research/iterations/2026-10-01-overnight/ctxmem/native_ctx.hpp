#pragma once
#include <array>
#include <stdexcept>

namespace wnn {
inline constexpr int kCtxRank = 16;
inline constexpr int kCtxWidth = 32;
inline constexpr double kCtxFast = 0x1.ae89f995ad3adp-1;
inline constexpr double kCtxSlow = 0x1.f50765b6e4540p-1;
inline constexpr float kCtxScale = 0.0883883461356163f;
inline constexpr std::array<int,16> kCtxRows = {68,93,98,43,72,42,49,44,7,104,76,95,26,64,41,65};

struct CtxConfig {
  bool enabled = false;
  bool current = false;
  void validate() const { if (!enabled && current) throw std::invalid_argument("disabled current CTX config"); }
};

inline float ctx_projection_value(int r, int c) {
  unsigned bits = static_cast<unsigned>(kCtxRows[r] & c);
  unsigned parity = 0;
  while(bits) { parity ^= bits & 1u; bits >>= 1; }
  return parity ? -kCtxScale : kCtxScale;
}

class CtxMemory {
 public:
  explicit CtxMemory(CtxConfig config = {}) : config_(config) { config_.validate(); reset(); }
  void reset() { initialized_ = false; fast_.fill(0.0); slow_.fill(0.0); output_.fill(0.f); }
  const float* update(const float* q, bool event) {
    if (!initialized_) {
      for(int k=0;k<kCtxRank;++k) fast_[k]=slow_[k]=static_cast<double>(q[k]);
      initialized_=true;
    } else if(event) {
      for(int k=0;k<kCtxRank;++k) {
        // Volatile intermediates explicitly forbid a fused multiply-add only in
        // this recurrence; original backbone compilation/numerics stay unchanged.
        volatile double old_fast=kCtxFast*fast_[k];
        volatile double new_fast=(1.0-kCtxFast)*static_cast<double>(q[k]);
        volatile double old_slow=kCtxSlow*slow_[k];
        volatile double new_slow=(1.0-kCtxSlow)*static_cast<double>(q[k]);
        fast_[k]=old_fast+new_fast; slow_[k]=old_slow+new_slow;
      }
    }
    for(int k=0;k<kCtxRank;++k) {
      output_[k]=config_.current?q[k]:static_cast<float>(fast_[k]);
      output_[k+kCtxRank]=config_.current?q[k]:static_cast<float>(slow_[k]);
    }
    return output_.data();
  }
 private:
  CtxConfig config_;
  bool initialized_ = false;
  std::array<double,kCtxRank> fast_{},slow_{};
  std::array<float,kCtxWidth> output_{};
};
} // namespace wnn
