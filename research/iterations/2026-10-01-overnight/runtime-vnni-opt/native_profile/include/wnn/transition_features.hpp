#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace wnn {
constexpr int kTransitionWidth = 68;
struct TransitionConfig {
  bool enabled = false, conditional = false;
  std::array<double,4> centers{};
  std::array<double,16> pi{};
  std::array<double,256> prior{};
  void validate() const {
    if (!enabled) return;
    double total=0;
    for(double c:centers) if(!std::isfinite(c)) throw std::invalid_argument("nonfinite center");
    for(double p:pi) { if(!(p>0) || !std::isfinite(p)) throw std::invalid_argument("invalid pi"); total+=p; }
    if(std::abs(total-1)>1e-10) throw std::invalid_argument("pi normalization");
    for(int i=0;i<16;++i) { total=0; for(int j=0;j<16;++j) { double p=prior[16*i+j]; if(!(p>0) || !std::isfinite(p)) throw std::invalid_argument("invalid R"); total+=p; }
      if(std::abs(total-1)>1e-10) throw std::invalid_argument("R normalization"); }
  }
};
class TransitionFeatures {
 public:
  explicit TransitionFeatures(TransitionConfig cfg=TransitionConfig{}) : cfg_(cfg) {cfg_.validate(); reset();}
  void reset() {
    previous_.fill(0); count_.fill(0); transition_.fill(0); outgoing_.fill(0); out_.fill(0);
    have_previous_=false; previous_valid_=false; last_mark_=-1; row_=0; last_event_=0; total_=0;
    if(cfg_.enabled) for(int j=0;j<16;++j) out_[35+j]=static_cast<float>(cfg_.pi[j]);
  }
  const float* update(const float* raw4) {
    bool valid=true, changed=false; float lo=raw4[0],hi=raw4[0];
    for(int j=0;j<4;++j) {
      valid=valid && std::isfinite(raw4[j]) && std::abs(raw4[j])<5.19f;
      changed=changed || raw4[j]!=previous_[j]; lo=std::min(lo,raw4[j]); hi=std::max(hi,raw4[j]);
    }
    const bool event=have_previous_ && previous_valid_ && valid && changed && (hi-lo)>1e-6f;
    out_[34]=event?1.f:0.f;
    if(event) {
      int mark=0; for(int j=0;j<4;++j) if(static_cast<double>(raw4[j])>cfg_.centers[j]) mark|=1<<j;
      for(int j=0;j<16;++j) {
        double p;
        if(!cfg_.conditional) p=(count_[j]+16*cfg_.pi[j])/(total_+16.);
        else if(last_mark_<0) p=cfg_.pi[j];
        else p=(transition_[16*last_mark_+j]+16*cfg_.prior[16*last_mark_+j])/(outgoing_[last_mark_]+16.);
        out_[16+j]=out_[j]; out_[j]=j==mark?1.f:0.f;
        out_[35+j]=static_cast<float>(p); out_[51+j]=static_cast<float>((j==mark?1.:0.)-p);
        if(j==mark) out_[67]=static_cast<float>(std::min(-std::log(p),10.)/10.);
      }
      if(last_mark_>=0) {++transition_[16*last_mark_+mark]; ++outgoing_[last_mark_];}
      ++count_[mark]; ++total_; last_mark_=mark; last_event_=row_;
      out_[32]=static_cast<float>(std::log1p(std::min(total_,20000))/std::log(20001.));
    }
    out_[33]=static_cast<float>(std::log1p(std::min(row_-last_event_,10000))/std::log(10001.));
    std::copy(raw4,raw4+4,previous_.begin()); previous_valid_=valid; have_previous_=true; ++row_;
    return out_.data();
  }
 private:
  TransitionConfig cfg_;
  std::array<float,4> previous_{};
  std::array<int,16> count_{},outgoing_{};
  std::array<int,256> transition_{};
  std::array<float,kTransitionWidth> out_{};
  bool have_previous_=false,previous_valid_=false;
  int row_=0,last_event_=0,total_=0,last_mark_=-1;
};
} // namespace wnn
