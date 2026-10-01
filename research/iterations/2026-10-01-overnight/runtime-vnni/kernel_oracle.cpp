#include "wnn/kernels.hpp"
#include "wnn/vnni.hpp"
#include <cfenv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
int main(){
 if(!wnn::vnni::supported()) return 77;
 std::fesetround(FE_TONEAREST); std::mt19937 rng(87123);std::uniform_real_distribution<float>d(-3,3);int cases=0,rows=0;
 for(int in:{64,128,256,512}) for(int out:{1,3,4,5,17,128,384,768}) for(int mode=0;mode<5;++mode){
  std::vector<float> W(out*in),x(in),b(out),y(out);for(auto&v:W)v=d(rng);for(auto&v:x)v=d(rng);for(auto&v:b)v=d(rng);
  if(mode==0){std::fill(W.begin(),W.end(),0);std::fill(x.begin(),x.end(),0);}
  if(mode==1){for(int i=0;i<in;++i)x[i]=(i%2?-127:127);for(int i=0;i<out*in;++i)W[i]=i%3?-127:127;}
  if(mode==2){x[0]=127;for(int i=1;i<in;++i)x[i]=(i%4)-1.5f;}
  if(mode==3){std::fill(x.begin(),x.end(),0);x[0]=-0.f;}
  wnn::kernels::Int8Vec q;std::vector<float>s;wnn::kernels::to_int8(W.data(),out,in,q,s);std::vector<std::int32_t> sums(out,0);for(int o=0;o<out;++o)for(int i=0;i<in;++i)sums[o]+=q[o*in+i];
  alignas(64) std::uint8_t u[512];float sx=wnn::vnni::quant_input(x.data(),in,u);float mx=0;for(auto v:x)mx=std::max(mx,std::fabs(v));float ox=mx?mx/127.f:1.f;if(sx!=ox)return 1;
  for(int i=0;i<in;++i)if(int(u[i])!=128+std::lrint(x[i]/sx))return 2;
  wnn::vnni::gemv(q.data(),s.data(),sums.data(),b.data(),x.data(),y.data(),out,in);
  for(int o=0;o<out;++o){std::int64_t signed_dot=0,unsigned_dot=0;for(int i=0;i<in;++i){signed_dot+=(int(u[i])-128)*int(q[o*in+i]);unsigned_dot+=int(u[i])*int(q[o*in+i]);}
   if(signed_dot!=unsigned_dot-128*sums[o])return 3;
   if(wnn::vnni::dot(u,q.data()+o*in,in,sums[o])!=signed_dot)return 4;
   float oracle=std::fma(float(signed_dot),s[o]*sx,b[o]);if(std::memcmp(&oracle,&y[o],4))return 5;
  }++cases;rows+=out;
 }
 for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){float x[64]={};x[2]=bad;std::uint8_t u[64];bool threw=false;try{wnn::vnni::quant_input(x,64,u);}catch(const std::invalid_argument&){threw=true;}if(!threw)return 6;}
 for(int bad:{0,1,63,65,1024}){bool threw=false;try{wnn::vnni::validate_width(bad);}catch(const std::invalid_argument&){threw=true;}if(!threw)return 7;}
 std::cout<<"{\"status\":\"PASS\",\"matrix_cases\":"<<cases<<",\"output_rows\":"<<rows<<",\"exact_integer_oracle\":true,\"exact_float_rescale\":true,\"nonfinite_rejection\":true,\"width_rejection\":true}\n";
}
