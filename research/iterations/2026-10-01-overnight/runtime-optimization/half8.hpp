#pragma once
#include "wnn/kernels.hpp"
namespace runtime_opt {
inline void gemv_half8(const std::uint16_t* __restrict W,const float* __restrict b,const float* __restrict x,
                      float* __restrict y,int out,int in){
 int o=0;
 for(;o+8<=out;o+=8){const auto*p=W+std::size_t(o)*in;
  __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
  for(int i=0;i<in;i+=8){const __m256 v=_mm256_loadu_ps(x+i);
   a0=_mm256_fmadd_ps(wnn::kernels::load_h8(p+i),v,a0);a1=_mm256_fmadd_ps(wnn::kernels::load_h8(p+in+i),v,a1);
   a2=_mm256_fmadd_ps(wnn::kernels::load_h8(p+2*in+i),v,a2);a3=_mm256_fmadd_ps(wnn::kernels::load_h8(p+3*in+i),v,a3);
   a4=_mm256_fmadd_ps(wnn::kernels::load_h8(p+4*in+i),v,a4);a5=_mm256_fmadd_ps(wnn::kernels::load_h8(p+5*in+i),v,a5);
   a6=_mm256_fmadd_ps(wnn::kernels::load_h8(p+6*in+i),v,a6);a7=_mm256_fmadd_ps(wnn::kernels::load_h8(p+7*in+i),v,a7);
  }
  y[o]=wnn::kernels::hsum8(a0)+b[o];y[o+1]=wnn::kernels::hsum8(a1)+b[o+1];
  y[o+2]=wnn::kernels::hsum8(a2)+b[o+2];y[o+3]=wnn::kernels::hsum8(a3)+b[o+3];
  y[o+4]=wnn::kernels::hsum8(a4)+b[o+4];y[o+5]=wnn::kernels::hsum8(a5)+b[o+5];
  y[o+6]=wnn::kernels::hsum8(a6)+b[o+6];y[o+7]=wnn::kernels::hsum8(a7)+b[o+7];
 }
 if(o<out)wnn::kernels::gemv_half(W+std::size_t(o)*in,b+o,x,y+o,out-o,in);
}
}
