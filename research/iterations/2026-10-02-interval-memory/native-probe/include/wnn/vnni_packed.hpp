#pragma once
#include "wnn/vnni.hpp"
#include <cstring>
namespace wnn::vnni {
inline std::vector<std::int8_t> pack64(const std::int8_t* q,int out,int in) {
  validate_width(in);std::vector<std::int8_t> p(static_cast<std::size_t>((out+63)/64)*64*in,0);
  for(int o=0;o<out;++o)for(int i=0;i<in;++i){
    const std::size_t at=static_cast<std::size_t>(o/64)*64*in+(i/4)*256+(o%64)*4+i%4;
    p[at]=q[static_cast<std::size_t>(o)*in+i];
  }return p;
}
inline __mmask16 output_mask(int o,int out){return o+16<=out?0xffff: o>=out?0:static_cast<__mmask16>((1u<<(out-o))-1u);}
WNN_VNNI_TARGET inline __m512i correction(const std::int32_t* sums,int o,int out) {
  const __mmask16 mask=output_mask(o,out);if(!mask)return _mm512_setzero_si512();
  return _mm512_sub_epi32(_mm512_setzero_si512(),_mm512_slli_epi32(_mm512_maskz_loadu_epi32(mask,sums+o),7));
}
WNN_VNNI_TARGET inline void store_result(__m512i d,const float* scale,const float* b,float* y,float sx,int o,int out){
  const __mmask16 mask=output_mask(o,out);if(!mask)return;
  __m512 s=_mm512_mul_ps(_mm512_maskz_loadu_ps(mask,scale+o),_mm512_set1_ps(sx));
  __m512 z=_mm512_fmadd_ps(_mm512_cvtepi32_ps(d),s,_mm512_maskz_loadu_ps(mask,b+o));
  _mm512_mask_storeu_ps(y+o,mask,z);
}
WNN_VNNI_TARGET inline void gemv_packed(const std::int8_t* P,const float* scale,const std::int32_t* sums,const float* b,const float* x,float* y,int out,int in){
  alignas(64) std::uint8_t u[512];const float sx=quant_input(x,in,u);
  for(int o=0;o<out;o+=64){
    __m512i a0=correction(sums,o,out),a1=correction(sums,o+16,out),a2=correction(sums,o+32,out),a3=correction(sums,o+48,out);
    const std::int8_t* p=P+static_cast<std::size_t>(o)*in;
    for(int i=0;i<in;i+=4,p+=256){
      std::int32_t bits;std::memcpy(&bits,u+i,4);const __m512i v=_mm512_set1_epi32(bits);
      a0=_mm512_dpbusd_epi32(a0,v,_mm512_loadu_si512(p));
      a1=_mm512_dpbusd_epi32(a1,v,_mm512_loadu_si512(p+64));
      a2=_mm512_dpbusd_epi32(a2,v,_mm512_loadu_si512(p+128));
      a3=_mm512_dpbusd_epi32(a3,v,_mm512_loadu_si512(p+192));
    }
    store_result(a0,scale,b,y,sx,o,out);store_result(a1,scale,b,y,sx,o+16,out);store_result(a2,scale,b,y,sx,o+32,out);store_result(a3,scale,b,y,sx,o+48,out);
  }
}
}

namespace wnn::vnni {
template<int NV> WNN_VNNI_TARGET inline void projection_registers(const float* __restrict W,const float* __restrict b,const float* __restrict x,float* __restrict y,int in){
  __m512 a[NV];
#pragma GCC unroll 16
  for(int j=0;j<NV;++j)a[j]=_mm512_loadu_ps(b+16*j);
  for(int i=0;i<in;++i){
    const float xi=x[i];if(xi==0.f)continue;const __m512 v=_mm512_set1_ps(xi);const float* w=W+static_cast<std::size_t>(i)*NV*16;
#pragma GCC unroll 16
    for(int j=0;j<NV;++j)a[j]=_mm512_fmadd_ps(v,_mm512_loadu_ps(w+16*j),a[j]);
  }
#pragma GCC unroll 16
  for(int j=0;j<NV;++j)_mm512_storeu_ps(y+16*j,a[j]);
}
}
