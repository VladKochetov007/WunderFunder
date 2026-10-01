#include "wnn/model.hpp"
#include <iostream>
#include <limits>
#include <cstring>
int main() {
  using namespace wnn;
  const float tiny=std::numeric_limits<float>::denorm_min();
  const float limit=std::numeric_limits<float>::min(), boundary=limit*127.f;
  const float values[]={0.f,tiny,63*tiny,64*tiny,128*tiny,190*tiny,limit,std::nextafter(boundary,0.f),boundary,std::nextafter(boundary,INFINITY),1.f,std::numeric_limits<float>::max(),INFINITY,NAN};
  int cases=0,rejects=0;
  for(float mx:values) {
    const float sx=mx>0.f?mx/127.f:1.f;
    const bool reject=!std::isfinite(mx)||!(sx>=limit)||!std::isfinite(sx);
    float x[64]={};x[0]=mx;x[1]=-mx;std::int8_t q[64]={};std::uint8_t u[64]={};bool rs=false,rv=false,ra=false;float ss=0,sv=0;
    try{ss=vnni::quant_input_scalar(x,64,q);}catch(const std::invalid_argument&){rs=true;}
    try{sv=vnni::quant_input(x,64,u);}catch(const std::invalid_argument&){rv=true;}
    std::int8_t weights[64];for(int i=0;i<64;++i)weights[i]=i%2?127:-127;
    const float scale=1.f,bias=0.f;float out=0;
    try{vnni::gemv_avx2(weights,&scale,&bias,x,&out,1,64);}catch(const std::invalid_argument&){ra=true;}
    if(rs!=reject||rv!=reject||ra!=reject)return 1;
    if(!reject){
      if(std::memcmp(&ss,&sv,4))return 2;
      for(int i=0;i<64;++i)if(int(q[i])!=int(u[i])-128)return 3;
    }
    ++cases;rejects+=reject;
  }
  const unsigned original=_mm_getcsr();const int rounding=std::fegetround();
  std::fesetround(FE_DOWNWARD);bool rejected=false;
  try{Weights w;w.make_vnni("scalar");}catch(const std::invalid_argument& e){rejected=std::string(e.what()).find("round-to-nearest")!=std::string::npos;}
  std::fesetround(rounding);_mm_setcsr(original);if(!rejected)return 4;
  for(unsigned mask:{1u<<15,1u<<6}) {
    _mm_setcsr(original|mask);rejected=false;
    try{Weights w;w.make_vnni("scalar");}catch(const std::invalid_argument& e){rejected=std::string(e.what()).find("FTZ/DAZ")!=std::string::npos;}
    _mm_setcsr(original);if(!rejected)return 5;
  }
  std::cout<<"{\"status\":\"PASS\",\"domain_cases\":"<<cases<<",\"identical_backend_rejections\":"<<rejects<<",\"normal_scale_boundary_below_at_above\":true,\"normal_range_quant_bytes_bitexact\":true,\"unsupported_environment_rejected\":true,\"initial_mxcsr\":"<<original<<",\"empirical_rows\":0}\n";
}
