#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <random>
#include <vector>
#include "wnn/kernels.hpp"
#include "half512.hpp"
using wnn::kernels::Vec;
using wnn::kernels::HalfVec;
struct M {int in,out;Vec W,x,b,y;HalfVec half;};
int main(){
 std::mt19937 rng(781);std::normal_distribution<float> nd(0,.1);
 std::vector<M> mats;
 for(int h:{256,128,128,128}){
  for(int k=0;k<5;++k){M m; m.in=k==4?2*h:h;m.out=k==4?h:3*h;
   m.W.resize(m.in*m.out);m.x.resize(m.in);m.b.resize(m.out);m.y.resize(m.out);
   for(auto&v:m.W)v=nd(rng);for(auto&v:m.x)v=nd(rng);for(auto&v:m.b)v=nd(rng);
   m.half=wnn::kernels::to_half(m.W.data(),m.W.size());
   Vec a(m.out),b(m.out);wnn::kernels::gemv_half(m.half.data(),m.b.data(),m.x.data(),a.data(),m.out,m.in);
   runtime_opt::gemv_half512(m.half.data(),m.b.data(),m.x.data(),b.data(),m.out,m.in);
   if(std::memcmp(a.data(),b.data(),sizeof(float)*m.out)){fprintf(stderr,"BIT PARITY FAIL h%d k%d\n",h,k);return 2;}
   mats.push_back(std::move(m));
  }
 }
 volatile float sink=0;
 const int repeats=15000;
 printf("{\"bitwise_kernel_parity\":true,\"repeats\":%d,\"measurements\":[",repeats);
 for(int trial=0;trial<6;++trial){const bool wide=trial%2;
  auto run=[&](){for(auto&m:mats){if(wide)runtime_opt::gemv_half512(m.half.data(),m.b.data(),m.x.data(),m.y.data(),m.out,m.in);
   else wnn::kernels::gemv_half(m.half.data(),m.b.data(),m.x.data(),m.y.data(),m.out,m.in);sink=m.y[0];}};
  for(int i=0;i<500;++i)run();
  double c0=double(std::clock())/CLOCKS_PER_SEC;auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<repeats;++i)run();
  double cpu=double(std::clock())/CLOCKS_PER_SEC-c0;
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  printf("%s{\"variant\":\"%s\",\"cpu_us_per_4model_dense_set\":%.8f,\"wall_us\":%.8f}",trial?",":"",wide?"half512":"half256",cpu*1e6/repeats,wall*1e6/repeats);
 }
 printf("]}\n");return !std::isfinite(sink);
}
