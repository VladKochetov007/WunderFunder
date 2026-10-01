#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define WNN_VNNI_TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
namespace wnn::vnni {
inline bool supported() { return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"); }
inline void validate_width(int in) { if(in<=0 || in>512 || in%64) throw std::invalid_argument("dynamic int8 requires input width64..512 divisible by64"); }
WNN_VNNI_TARGET inline float quant_input(const float* x, int in, std::uint8_t* u) {
  __m512 m=_mm512_setzero_ps();
  for(int i=0;i<in;i+=16) {
    __m512 a=_mm512_abs_ps(_mm512_loadu_ps(x+i));
    if(_mm512_cmp_ps_mask(a,_mm512_set1_ps(INFINITY),_CMP_NLT_UQ)) throw std::invalid_argument("nonfinite dynamic int8 input");
    m=_mm512_max_ps(m,a);
  }
  float mx=_mm512_reduce_max_ps(m), s=mx>0.f?mx/127.f:1.f;
  __m512 scale=_mm512_set1_ps(s);
  for(int i=0;i<in;i+=16) {
    __m512 z=_mm512_div_ps(_mm512_loadu_ps(x+i),scale);
    __m512i q=_mm512_cvt_roundps_epi32(z,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    q=_mm512_add_epi32(q,_mm512_set1_epi32(128));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(u+i),_mm512_cvtusepi32_epi8(q));
  }
  return s;
}
WNN_VNNI_TARGET inline std::int32_t dot(const std::uint8_t* u,const std::int8_t* w,int in,std::int32_t rowsum) {
  __m512i a=_mm512_setzero_si512();
  for(int i=0;i<in;i+=64) a=_mm512_dpbusd_epi32(a,_mm512_loadu_si512(u+i),_mm512_loadu_si512(w+i));
  return _mm512_reduce_add_epi32(a)-128*rowsum;
}
WNN_VNNI_TARGET inline void gemv(const std::int8_t* Q,const float* scale,const std::int32_t* rowsum,const float* b,const float* x,float* y,int out,int in) {
  alignas(64) std::uint8_t u[512]; const float sx=quant_input(x,in,u);
  int o=0;
  for(;o+4<=out;o+=4) {
    __m512i a0=_mm512_setzero_si512(),a1=a0,a2=a0,a3=a0;
    const std::int8_t* q=Q+static_cast<std::size_t>(o)*in;
    for(int i=0;i<in;i+=64) {
      const __m512i v=_mm512_loadu_si512(u+i);
      a0=_mm512_dpbusd_epi32(a0,v,_mm512_loadu_si512(q+i));
      a1=_mm512_dpbusd_epi32(a1,v,_mm512_loadu_si512(q+in+i));
      a2=_mm512_dpbusd_epi32(a2,v,_mm512_loadu_si512(q+2*in+i));
      a3=_mm512_dpbusd_epi32(a3,v,_mm512_loadu_si512(q+3*in+i));
    }
    const std::int32_t d[4]={_mm512_reduce_add_epi32(a0)-128*rowsum[o],_mm512_reduce_add_epi32(a1)-128*rowsum[o+1],_mm512_reduce_add_epi32(a2)-128*rowsum[o+2],_mm512_reduce_add_epi32(a3)-128*rowsum[o+3]};
    for(int j=0;j<4;++j) y[o+j]=std::fma(static_cast<float>(d[j]),scale[o+j]*sx,b[o+j]);
  }
  for(;o<out;++o) y[o]=std::fma(static_cast<float>(dot(u,Q+static_cast<std::size_t>(o)*in,in,rowsum[o])),scale[o]*sx,b[o]);
}
} // namespace wnn::vnni
