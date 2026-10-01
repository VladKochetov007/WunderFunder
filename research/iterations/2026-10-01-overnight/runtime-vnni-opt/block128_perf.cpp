#include "wnn/kernels.hpp"
#include "wnn/vnni_block128.hpp"
#include <chrono>
#include <iostream>
#include <random>
volatile float sink=0;
int main(){std::mt19937 r(1);std::uniform_real_distribution<float>d(-1,1);std::cout<<"{\"shapes\":[";bool first=true;
 for(auto shape:{std::pair<int,int>{384,128},{768,256},{128,256},{256,512}}){int out=shape.first,in=shape.second;std::vector<float>w(out*in),x(in),b(out),y(out);for(auto&v:w)v=d(r);for(auto&v:x)v=d(r);wnn::kernels::Int8Vec q;std::vector<float>s;wnn::kernels::to_int8(w.data(),out,in,q,s);std::vector<int>sum(out,0);for(int o=0;o<out;++o)for(int i=0;i<in;++i)sum[o]+=q[o*in+i];auto p=wnn::vnni::pack128(q.data(),out,in);
 if(!first)std::cout<<",";first=false;std::cout<<"{\"out\":"<<out<<",\"in\":"<<in<<",\"microseconds\":[";
 for(int mode=0;mode<3;++mode){int n=20000;auto t=std::chrono::steady_clock::now();for(int z=0;z<n;++z){x[0]=(z%127)*.001f;if(mode==0)wnn::vnni::gemv(q.data(),s.data(),sum.data(),b.data(),x.data(),y.data(),out,in);if(mode==1)wnn::vnni::gemv_block128(p.data(),s.data(),sum.data(),b.data(),x.data(),y.data(),out,in);if(mode==2){std::uint8_t u[512];sink+=wnn::vnni::quant_input(x.data(),in,u);sink+=u[z%in];}else sink+=y[z%out];}double us=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/n*1e6;if(mode)std::cout<<",";std::cout<<us;}std::cout<<"]}";
 }std::cout<<"],\"modes\":[\"rowmajor\",\"packed128\",\"quant_only\"]}\n";}
