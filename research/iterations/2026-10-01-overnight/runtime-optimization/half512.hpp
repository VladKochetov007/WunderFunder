#pragma once
#include <immintrin.h>
#include <cstdint>
#include "wnn/kernels.hpp"
namespace runtime_opt {
inline __m512 load16(const std::uint16_t* p) {
 return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
inline float reduce16(__m512 v) {
 const __m256 lo=_mm512_castps512_ps256(v), hi=_mm512_extractf32x8_ps(v,1);
 return wnn::kernels::hsum8(_mm256_add_ps(lo,hi));
}
// Same16independentFMA chains and reduction tree as original AVX2 half kernel.
inline void gemv_half512(const std::uint16_t* __restrict W,const float* __restrict b,const float* __restrict x,
                         float* __restrict y,int out,int in) {
 int o=0;
 for(;o+8<=out;o+=8) {
  const std::uint16_t* p=W+std::size_t(o)*in;
  __m512 a0=_mm512_setzero_ps(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
  for(int i=0;i<in;i+=16) {
   const __m512 v=_mm512_loadu_ps(x+i);
   a0=_mm512_fmadd_ps(load16(p+i),v,a0);
   a1=_mm512_fmadd_ps(load16(p+in+i),v,a1);
   a2=_mm512_fmadd_ps(load16(p+2*in+i),v,a2);
   a3=_mm512_fmadd_ps(load16(p+3*in+i),v,a3);
   a4=_mm512_fmadd_ps(load16(p+4*in+i),v,a4);
   a5=_mm512_fmadd_ps(load16(p+5*in+i),v,a5);
   a6=_mm512_fmadd_ps(load16(p+6*in+i),v,a6);
   a7=_mm512_fmadd_ps(load16(p+7*in+i),v,a7);
  }
  y[o]=reduce16(a0)+b[o]; y[o+1]=reduce16(a1)+b[o+1];
  y[o+2]=reduce16(a2)+b[o+2]; y[o+3]=reduce16(a3)+b[o+3];
  y[o+4]=reduce16(a4)+b[o+4]; y[o+5]=reduce16(a5)+b[o+5];
  y[o+6]=reduce16(a6)+b[o+6]; y[o+7]=reduce16(a7)+b[o+7];
 }
 if(o<out) wnn::kernels::gemv_half(W+std::size_t(o)*in,b+o,x,y+o,out-o,in);
}
}
