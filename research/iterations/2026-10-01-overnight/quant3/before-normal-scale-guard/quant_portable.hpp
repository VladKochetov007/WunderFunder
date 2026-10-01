#pragma once
// Q3: the frozen VNNI quantizer/kernel, with exact signed-int32 fallback dots.
#include "wnn/vnni_block128.hpp"
#include <cfenv>
#include <string>

#define WNN_AVX2_TARGET __attribute__((target("avx2,fma")))
namespace wnn::vnni {
inline bool avx2_supported() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
inline int resolve_backend(const std::string& name) {
  if(name=="auto") return supported()?2:avx2_supported()?1:0;
  if(name=="scalar") return 0;
  if(name=="avx2" && avx2_supported()) return 1;
  if(name=="vnni" && supported()) return 2;
  throw std::invalid_argument("unavailable or unknown quant backend: "+name);
}

inline float quant_input_scalar(const float* x,int in,std::int8_t* q) {
  float mx=0.f;
  for(int i=0;i<in;++i) {
    if(!std::isfinite(x[i])) throw std::invalid_argument("nonfinite dynamic int8 input");
    mx=std::max(mx,std::fabs(x[i]));
  }
  const float sx=mx>0.f?mx/127.f:1.f;
  if(!(sx>0.f) || !std::isfinite(sx)) throw std::invalid_argument("invalid dynamic int8 scale");
  for(int i=0;i<in;++i) q[i]=static_cast<std::int8_t>(std::lrint(x[i]/sx));
  return sx;
}

inline std::int32_t dot_scalar(const std::int8_t* x,const std::int8_t* w,int in) {
  std::int32_t d=0;
  for(int i=0;i<in;++i) d+=static_cast<std::int32_t>(x[i])*static_cast<std::int32_t>(w[i]);
  return d;
}

WNN_AVX2_TARGET inline std::int32_t dot_avx2(const std::int8_t* x,const std::int8_t* w,int in) {
  // Widen signed bytes before pair multiplication. No saturating maddubs operation.
  __m256i d=_mm256_setzero_si256();
  for(int i=0;i<in;i+=16) {
    const __m256i a=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+i)));
    const __m256i b=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+i)));
    d=_mm256_add_epi32(d,_mm256_madd_epi16(a,b));
  }
  alignas(32) std::int32_t lanes[8];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),d);
  std::int32_t s=0;for(int j=0;j<8;++j)s+=lanes[j];return s;
}

inline void gemv_scalar(const std::int8_t* Q,const float* scale,const float* b,const float* x,float* y,int out,int in) {
  alignas(64) std::int8_t q[512];const float sx=quant_input_scalar(x,in,q);
  for(int o=0;o<out;++o) {
    const auto d=dot_scalar(q,Q+static_cast<std::size_t>(o)*in,in);
    const float s=scale[o]*sx;
    y[o]=std::fma(static_cast<float>(d),s,b[o]);
  }
}

WNN_AVX2_TARGET inline void gemv_avx2(const std::int8_t* Q,const float* scale,const float* b,const float* x,float* y,int out,int in) {
  alignas(64) std::int8_t q[512];const float sx=quant_input_scalar(x,in,q);
  for(int o=0;o<out;++o) {
    const auto d=dot_avx2(q,Q+static_cast<std::size_t>(o)*in,in);
    const float s=scale[o]*sx;
    y[o]=std::fma(static_cast<float>(d),s,b[o]);
  }
}

inline void gemv_dispatch(int backend,const std::int8_t* Q,const std::int8_t* P,const float* scale,const std::int32_t* sums,const float* b,const float* x,float* y,int out,int in) {
  if(backend==2) gemv_block128(P,scale,sums,b,x,y,out,in);
  else if(backend==1) gemv_avx2(Q,scale,b,x,y,out,in);
  else gemv_scalar(Q,scale,b,x,y,out,in);
}

inline void projection_portable(const float* W,const float* b,const float* x,float* y,int out,int in) {
  // Same input order and single FMA as the frozen AVX512 register projection.
  std::copy(b,b+out,y);
  for(int i=0;i<in;++i) {
    if(x[i]==0.f)continue;
    for(int o=0;o<out;++o)y[o]=std::fma(x[i],W[static_cast<std::size_t>(i)*out+o],y[o]);
  }
}
}
