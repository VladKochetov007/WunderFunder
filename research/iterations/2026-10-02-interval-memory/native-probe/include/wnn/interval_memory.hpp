// Fixed FP32 added branch; current event closes the interval. Qualified trunk
// and head precision/backend remain unchanged. No branch skips zero weights.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "wnn/kernels.hpp"
#include "wnn/activations.hpp"
namespace wnn {
class IntervalMemory {
  kernels::Vec projection_, projection_bias_, ih_, ib_, hh_, hb_, extra_;
  kernels::Vec total_, slow_, input_, latent_, gi_, gh_, rz_, addition_, zero_;
  std::uint64_t count_=0; bool started_=false; int arm_;
 public:
  // Packed order: projection W,b; event GRU W_ih,b_ih,W_hh,b_hh; extra W.
  explicit IntervalMemory(const float* p,int arm):total_(32,0.f),slow_(64,0.f),
    input_(33),latent_(96),gi_(192),gh_(192),rz_(128),addition_(256),zero_(256,0.f),arm_(arm) {
    if(arm!=0&&arm!=1) throw std::invalid_argument("interval arm");
    auto take=[&](kernels::Vec& v,int n){v.assign(p,p+n);p+=n;};
    take(projection_,32*256);take(projection_bias_,32);
    take(ih_,192*33);take(ib_,192);take(hh_,192*64);take(hb_,192);take(extra_,256*96);
  }
  void reset(){std::fill(total_.begin(),total_.end(),0.f);std::fill(slow_.begin(),slow_.end(),0.f);count_=0;started_=false;}
  void add(const float* fast,bool event,float* preactivation,bool approximate){
    if(!started_){event=false;started_=true;}
    kernels::gemv(projection_.data(),projection_bias_.data(),fast,input_.data(),32,256);
    ++count_;
    for(int j=0;j<32;++j){float v=approximate?act::tanh_fast(input_[j]):std::tanh(input_[j]);total_[j]+=v;latent_[j]=arm_==0?total_[j]/float(count_):v;input_[j]=latent_[j];}
    if(event){
      input_[32]=std::log1p(float(count_));
      kernels::gemv(ih_.data(),ib_.data(),input_.data(),gi_.data(),192,33);
      kernels::gemv(hh_.data(),hb_.data(),slow_.data(),gh_.data(),192,64);
      for(int j=0;j<128;++j)rz_[j]=approximate?act::sigmoid_fast(gi_[j]+gh_[j]):act::sigmoid_exact(gi_[j]+gh_[j]);
      for(int j=0;j<64;++j){float x=gi_[128+j]+rz_[j]*gh_[128+j];float n=approximate?act::tanh_fast(x):std::tanh(x);slow_[j]=(1.f-rz_[64+j])*n+rz_[64+j]*slow_[j];}
    }
    std::copy(slow_.begin(),slow_.end(),latent_.begin()+32);
    kernels::gemv(extra_.data(),zero_.data(),latent_.data(),addition_.data(),256,96);
    for(int j=0;j<256;++j)preactivation[j]+=addition_[j];
    if(event){std::fill(total_.begin(),total_.end(),0.f);count_=0;}
  }
};
}
