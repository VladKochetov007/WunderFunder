#pragma once
#include "wnn/vnni_packed.hpp"
namespace wnn::vnni {
inline std::vector<std::int8_t> pack128(const std::int8_t* q,int out,int in){
 validate_width(in);std::vector<std::int8_t> p(static_cast<std::size_t>((out+127)/128)*128*in,0);
 for(int o=0;o<out;++o)for(int i=0;i<in;++i)p[static_cast<std::size_t>(o/128)*128*in+(i/4)*512+(o%128)*4+i%4]=q[static_cast<std::size_t>(o)*in+i];return p;
}
WNN_VNNI_TARGET inline void gemv_block128_prequant(const std::int8_t* P,const float* scale,const std::int32_t* sums,const float* b,const std::uint8_t* u,float sx,float* y,int out,int in){
 for(int o=0;o<out;o+=128){
  __m512i a[8];
#pragma GCC unroll 8
  for(int j=0;j<8;++j)a[j]=correction(sums,o+16*j,out);
  const std::int8_t* p=P+static_cast<std::size_t>(o)*in;
  for(int i=0;i<in;i+=4,p+=512){std::int32_t bits;std::memcpy(&bits,u+i,4);const __m512i v=_mm512_set1_epi32(bits);
#pragma GCC unroll 8
   for(int j=0;j<8;++j)a[j]=_mm512_dpbusd_epi32(a[j],v,_mm512_loadu_si512(p+64*j));
  }
#pragma GCC unroll 8
  for(int j=0;j<8;++j)store_result(a[j],scale,b,y,sx,o+16*j,out);
 }
}
WNN_VNNI_TARGET inline void gemv_block128(const std::int8_t* P,const float* scale,const std::int32_t* sums,const float* b,const float* x,float* y,int out,int in){
 alignas(64)std::uint8_t u[1024];const float sx=quant_input(x,in,u);
 gemv_block128_prequant(P,scale,sums,b,u,sx,y,out,in);
}
}
