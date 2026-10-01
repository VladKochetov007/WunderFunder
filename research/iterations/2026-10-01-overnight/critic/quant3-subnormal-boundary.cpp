#include "wnn/quant_portable.hpp"
#include <iostream>
#include <limits>
int main(){
  const float tiny=std::numeric_limits<float>::denorm_min();
  for(int units:{1,63,64,65,126,127,128,129,190,191,192,255,256,1000,10000}){
    float x[64]={};x[0]=tiny*units;std::int8_t q[64]={};std::uint8_t u[64]={};
    bool scalar_reject=false,vnni_reject=false;float scale=-1;
    try{scale=wnn::vnni::quant_input_scalar(x,64,q);}catch(const std::invalid_argument&){scalar_reject=true;}
    try{wnn::vnni::quant_input(x,64,u);}catch(const std::invalid_argument&){vnni_reject=true;}
    int disagree=0;for(int i=0;i<64;++i)disagree+=int(q[i])!=int(u[i])-128;
    std::cout<<"{\"maximum_subnormal_units\":"<<units<<",\"scale_units\":"<<scale/tiny<<",\"q_scalar\":"<<int(q[0])<<",\"q_vnni\":"<<int(u[0])-128<<",\"byte_mismatches\":"<<disagree<<",\"scalar_reject\":"<<scalar_reject<<",\"vnni_reject\":"<<vnni_reject<<"}\n";
  }
}
