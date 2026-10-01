#include "wnn/quant_portable.hpp"
#include "wnn/kernels.hpp"
#include <iostream>
#include <limits>
#include <cstring>

// Deliberately independent literal nearest-even oracle on unit-scale half ties.
static int nearest(float x) {
  const int f=static_cast<int>(std::floor(x));
  const float d=x-f;
  return d<.5f?f:d>.5f?f+1:(f%2==0?f:f+1);
}
int main() {
  if(std::fegetround()!=FE_TONEAREST)return 1;
  int cases=0,dots=0;
  for(int n:{64,128,256,512})for(int base=-126;base<=125;++base)for(int side:{-1,0,1}) {
    std::vector<float>x(n,base+.5f);x[0]=127.f;x[1]=-127.f;
    if(side)for(int j=2;j<n;++j)x[j]=std::nextafter(x[j],side<0?-INFINITY:INFINITY);
    std::vector<std::int8_t>q(n),w(n);std::vector<std::uint8_t>u(n);
    float s=wnn::vnni::quant_input_scalar(x.data(),n,q.data());if(s!=1.f)return 2;
    for(int j=0;j<n;++j)if(q[j]!=nearest(x[j]))return 3;
    if(wnn::vnni::supported()) {
      float t=wnn::vnni::quant_input(x.data(),n,u.data());if(t!=s)return 4;
      for(int j=0;j<n;++j)if(int(u[j])-128!=q[j])return 5;
    }
    for(int pat=0;pat<4;++pat) {
      int sum=0;std::int64_t exact=0;
      for(int j=0;j<n;++j) {w[j]=pat==0?127:pat==1?-127:pat==2?(j%2?127:-127):((j*83)%255)-127;sum+=w[j];exact+=std::int64_t(q[j])*w[j];}
      if(wnn::vnni::dot_scalar(q.data(),w.data(),n)!=exact)return 6;
      if(wnn::vnni::avx2_supported()&&wnn::vnni::dot_avx2(q.data(),w.data(),n)!=exact)return 7;
      if(wnn::vnni::supported()&&wnn::vnni::dot(u.data(),w.data(),n,sum)!=exact)return 8;
      ++dots;
    }
    ++cases;
  }
  int tiny_disagreements=0;float tiny_scale=-1;
  if(wnn::vnni::supported()) {
    std::vector<float>x(64,0.f);x[0]=std::numeric_limits<float>::denorm_min();
    std::vector<std::int8_t>q(64);std::vector<std::uint8_t>u(64);
    tiny_scale=wnn::vnni::quant_input_scalar(x.data(),64,q.data());
    wnn::vnni::quant_input(x.data(),64,u.data());
    for(int j=0;j<64;++j)tiny_disagreements+=int(q[j])!=int(u[j])-128;
  }
  std::cout<<"{\"unit_scale_tie_cases\":"<<cases<<",\"dot_cases\":"<<dots<<",\"normal_range_status\":\"PASS\",\"subnormal_scale\":"<<tiny_scale<<",\"subnormal_quant_byte_disagreements\":"<<tiny_disagreements<<",\"vnni\":"<<wnn::vnni::supported()<<",\"avx2\":"<<wnn::vnni::avx2_supported()<<"}\n";
}
