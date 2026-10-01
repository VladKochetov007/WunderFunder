#pragma once
#include "wnn/vnni_packed.hpp"
namespace wnn::vnni {
WNN_VNNI_TARGET inline float quant_input_ymm(const float*x,int in,std::uint8_t*u){
 __m256 m=_mm256_setzero_ps();for(int i=0;i<in;i+=8){__m256 a=_mm256_andnot_ps(_mm256_set1_ps(-0.f),_mm256_loadu_ps(x+i));if(_mm256_movemask_ps(_mm256_cmp_ps(a,_mm256_set1_ps(INFINITY),_CMP_NLT_UQ)))throw std::invalid_argument("nonfinite dynamic int8 input");m=_mm256_max_ps(m,a);}
 __m128 a=_mm_max_ps(_mm256_castps256_ps128(m),_mm256_extractf128_ps(m,1));a=_mm_max_ps(a,_mm_movehl_ps(a,a));a=_mm_max_ss(a,_mm_shuffle_ps(a,a,1));float mx=_mm_cvtss_f32(a),s=mx>0?mx/127.f:1.f;
 for(int i=0;i<in;i+=8){__m256 z=_mm256_div_ps(_mm256_loadu_ps(x+i),_mm256_set1_ps(s));__m256i q=_mm256_cvtps_epi32(_mm256_round_ps(z,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));q=_mm256_add_epi32(q,_mm256_set1_epi32(128));_mm_storel_epi64(reinterpret_cast<__m128i*>(u+i),_mm256_cvtusepi32_epi8(q));}return s;
}
WNN_VNNI_TARGET inline __m256i mask8(int o,int out){return _mm256_setr_epi32(o<out?-1:0,o+1<out?-1:0,o+2<out?-1:0,o+3<out?-1:0,o+4<out?-1:0,o+5<out?-1:0,o+6<out?-1:0,o+7<out?-1:0);}
WNN_VNNI_TARGET inline void gemv_ymm(const std::int8_t*P,const float*scale,const std::int32_t*sums,const float*b,const float*x,float*y,int out,int in){
 alignas(64)std::uint8_t u[512];float sx=quant_input_ymm(x,in,u);
 for(int o=0;o<out;o+=64){__m256i a[8];
#pragma GCC unroll 8
  for(int j=0;j<8;++j){int at=o+8*j;a[j]=at>=out?_mm256_setzero_si256():_mm256_sub_epi32(_mm256_setzero_si256(),_mm256_slli_epi32(_mm256_maskload_epi32(sums+at,mask8(at,out)),7));}
  const std::int8_t*p=P+static_cast<std::size_t>(o)*in;
  for(int i=0;i<in;i+=4,p+=256){std::int32_t bits;std::memcpy(&bits,u+i,4);__m256i v=_mm256_set1_epi32(bits);
#pragma GCC unroll 8
   for(int j=0;j<8;++j)a[j]=_mm256_dpbusd_epi32(a[j],v,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+j*32)));
  }
#pragma GCC unroll 8
  for(int j=0;j<8;++j){int at=o+8*j;if(at<out){__m256i mask=mask8(at,out);__m256 s=_mm256_mul_ps(_mm256_maskload_ps(scale+at,mask),_mm256_set1_ps(sx));__m256 z=_mm256_fmadd_ps(_mm256_cvtepi32_ps(a[j]),s,_mm256_maskload_ps(b+at,mask));_mm256_maskstore_ps(y+at,mask,z);}}
 }
}
template<int NB> WNN_VNNI_TARGET inline void projection_ymm(const float*W,const float*b,const float*x,float*y,int in){
 for(int block=0;block<NB;++block){__m256 a[8];int base=block*64;
#pragma GCC unroll 8
  for(int j=0;j<8;++j)a[j]=_mm256_loadu_ps(b+base+j*8);
  for(int i=0;i<in;++i){float xi=x[i];if(xi==0.f)continue;__m256 v=_mm256_set1_ps(xi);const float*w=W+static_cast<std::size_t>(i)*NB*64+base;
#pragma GCC unroll 8
   for(int j=0;j<8;++j)a[j]=_mm256_fmadd_ps(v,_mm256_loadu_ps(w+8*j),a[j]);
  }
#pragma GCC unroll 8
  for(int j=0;j<8;++j)_mm256_storeu_ps(y+base+j*8,a[j]);
 }
}
}
