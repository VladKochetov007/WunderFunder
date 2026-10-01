#include "wnn/quant_portable.hpp"
#include "wnn/kernels.hpp"
#include "wnn/model.hpp"
#include <iostream>
#include <random>
#include <cstring>
#include <limits>

int main() {
  using namespace wnn;
  if(std::fegetround()!=FE_TONEAREST)return 9;
  std::mt19937 rng(8233);std::uniform_real_distribution<float> unif(-3.f,3.f);
  int matrices=0,rows=0,quant_vectors=0;
  for(int in:{64,128,256,512})for(int out:{1,15,16,17,127,128,129,384,768})for(int trial=0;trial<8;++trial){
    std::vector<float> W(out*in),x(in),b(out),scale;
    for(auto& v:W)v=unif(rng);for(auto& v:x)v=unif(rng);for(auto&v:b)v=unif(rng);
    if(trial==0)std::fill(x.begin(),x.end(),0.f);
    if(trial==1) {std::fill(x.begin(),x.end(),-0.f);std::fill(W.begin(),W.end(),0.f);}
    if(trial==2) {for(int i=0;i<in;++i)x[i]=(i%2?127.f:-127.f);for(auto&v:W)v=127.f;}
    if(trial==3) {x[0]=127.f;for(int i=1;i<in;++i)x[i]=(i%127)-63.5f;}
    if(trial==4) {x[0]=-127.f;for(int i=1;i<in;++i)x[i]=std::nextafter((i%127)-63.5f,0.f);}
    kernels::Int8Vec Q;kernels::to_int8(W.data(),out,in,Q,scale);
    for(int o=0;o<out;++o) {
      float mx=0.f;for(int i=0;i<in;++i)mx=std::max(mx,std::fabs(W[o*in+i]));
      float s=mx>0?mx/127.f:1.f;if(std::memcmp(&s,&scale[o],4))return 1;
      for(int i=0;i<in;++i)if(Q[o*in+i]!=static_cast<int>(std::nearbyint(W[o*in+i]/s)))return 2;
    }
    auto P=vnni::pack128(Q.data(),out,in);std::vector<std::int32_t> sums(out);
    for(int o=0;o<out;++o)for(int i=0;i<in;++i)sums[o]+=Q[o*in+i];
    std::vector<std::int8_t> q(in);std::vector<std::uint8_t> u(in);
    float sx=vnni::quant_input_scalar(x.data(),in,q.data());
    if(vnni::supported()) {
      float sv=vnni::quant_input(x.data(),in,u.data());if(std::memcmp(&sx,&sv,4))return 3;
      for(int i=0;i<in;++i)if(int(u[i])-128!=q[i])return 4;
    }
    ++quant_vectors;
    std::vector<float> ref(out),got(out);
    for(int o=0;o<out;++o) {
      std::int64_t d=0;for(int i=0;i<in;++i)d+=std::int64_t(q[i])*Q[o*in+i];
      if(vnni::dot_scalar(q.data(),Q.data()+o*in,in)!=d)return 5;
      if(vnni::avx2_supported()&&vnni::dot_avx2(q.data(),Q.data()+o*in,in)!=d)return 6;
      if(vnni::supported()&&vnni::dot(u.data(),Q.data()+o*in,in,sums[o])!=d)return 7;
      const float s=scale[o]*sx;ref[o]=std::fma(static_cast<float>(d),s,b[o]);
    }
    for(int backend=0;backend<=2;++backend) {
      if(backend==1&&!vnni::avx2_supported())continue;if(backend==2&&!vnni::supported())continue;
      vnni::gemv_dispatch(backend,Q.data(),P.data(),scale.data(),sums.data(),b.data(),x.data(),got.data(),out,in);
      if(std::memcmp(ref.data(),got.data(),out*4)) {std::cerr<<"mismatch "<<backend<<" "<<in<<" "<<out<<" "<<trial<<"\n";return 8;}
    }
    ++matrices;rows+=out;
  }
  // Projection fallback preserves the exact input-ordered FMA recurrence.
  for(int out:{128,256}) {
    int in=211;std::vector<float> W(in*out),b(out),x(in),a(out),z(out);
    for(auto&v:W)v=unif(rng);for(auto&v:b)v=unif(rng);for(auto&v:x)v=unif(rng);
    for(int i=0;i<in;i+=3)x[i]=0.f;
    vnni::projection_portable(W.data(),b.data(),x.data(),a.data(),out,in);
    if(vnni::supported()) {
      if(out==128)vnni::projection_registers<8>(W.data(),b.data(),x.data(),z.data(),in);
      else vnni::projection_registers<16>(W.data(),b.data(),x.data(),z.data(),in);
      if(std::memcmp(a.data(),z.data(),out*4))return 10;
    }
  }
  // Input rejection is explicit and never routes to another candidate.
  for(float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    float x[64]={};x[5]=invalid;std::int8_t q[64];bool caught=false;
    try {vnni::quant_input_scalar(x,64,q);}catch(const std::invalid_argument&){caught=true;}if(!caught)return 11;
  }
  for(int multiplier:{1,2,31,63,64,65,127,128}) {
    float x[64]={};x[0]=std::numeric_limits<float>::denorm_min()*multiplier;
    std::int8_t q[64];std::uint8_t u[64];float ss=0,sv=0;bool scalar_reject=false,vnni_reject=false;
    try {ss=vnni::quant_input_scalar(x,64,q);}catch(const std::invalid_argument&){scalar_reject=true;}
    if(vnni::supported()) {
      try {sv=vnni::quant_input(x,64,u);}catch(const std::invalid_argument&){vnni_reject=true;}
      if(scalar_reject!=vnni_reject)return 12;
      if(!scalar_reject) {if(std::memcmp(&ss,&sv,4))return 13;for(int i=0;i<64;++i)if(int(u[i])-128!=q[i])return 14;}
    }
    if(!scalar_reject)return 15;
  }
  for(int sign:{-1,0,1}) {
    float x[64]={};float boundary=std::numeric_limits<float>::min()*127.f;
    x[0]=sign<0?std::nextafter(boundary,0.f):sign>0?std::nextafter(boundary,INFINITY):boundary;
    std::int8_t q[64];std::uint8_t u[64];float ss=0,sv=0;bool a=false,b=false;
    try {ss=vnni::quant_input_scalar(x,64,q);}catch(const std::invalid_argument&){a=true;}
    if(vnni::supported()) {
      try {sv=vnni::quant_input(x,64,u);}catch(const std::invalid_argument&){b=true;}
      if(a!=b)return 16;
      if(!a){if(std::memcmp(&ss,&sv,4))return 17;for(int i=0;i<64;++i)if(int(u[i])-128!=q[i])return 18;}
    }
    if(sign>=0&&a)return 19;
  }
  {
    const unsigned original=_mm_getcsr();const int rounding=std::fegetround();
    bool caught=false;std::fesetround(FE_UPWARD);
    try {Weights w;w.make_vnni("scalar");}catch(const std::invalid_argument& e){caught=std::string(e.what()).find("round-to-nearest")!=std::string::npos;}
    std::fesetround(rounding);_mm_setcsr(original);if(!caught)return 20;
    for(unsigned mode:{1u<<15,1u<<6}) {
      caught=false;_mm_setcsr(original|mode);
      try {Weights w;w.make_vnni("scalar");}catch(const std::invalid_argument& e){caught=std::string(e.what()).find("FTZ/DAZ")!=std::string::npos;}
      _mm_setcsr(original);if(!caught)return 21;
    }
  }
  std::cout<<"{\"status\":\"PASS\",\"matrices\":"<<matrices<<",\"output_rows\":"<<rows
           <<",\"quant_vectors\":"<<quant_vectors<<",\"int64_oracle\":true,\"gemv_bitexact\":true,\"projection_bitexact\":true,\"vnni\":"<<vnni::supported()<<",\"avx2\":"<<vnni::avx2_supported()<<"}\n";
}
