#include "wnn/kernels.hpp"
#include "wnn/vnni_packed.hpp"
#include <cstring>
#include <random>
#include <iostream>
int main(){std::mt19937 r(43265);std::uniform_real_distribution<float>d(-5,5);int cases=0;
 for(int out:{128,256})for(int in:{1,17,112,211,323})for(int mode=0;mode<10;++mode){std::vector<float>w(in*out),x(in),b(out),a(out),z(out);for(auto&v:w)v=d(r);for(auto&v:x)v=d(r);for(auto&v:b)v=d(r);if(mode==0)std::fill(x.begin(),x.end(),0);if(mode==1)std::fill(x.begin(),x.end(),-0.f);if(mode==2)for(int i=0;i<in;++i)x[i]*=1e10f;
 wnn::kernels::saxpy_sparse(w.data(),b.data(),x.data(),a.data(),out,in);if(out==128)wnn::vnni::projection_registers<8>(w.data(),b.data(),x.data(),z.data(),in);else wnn::vnni::projection_registers<16>(w.data(),b.data(),x.data(),z.data(),in);if(std::memcmp(a.data(),z.data(),4*out)){std::cerr<<"FAIL "<<out<<" "<<in<<" "<<mode<<"\n";return 1;}++cases;}
 std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"original_projection_bitexact\":true}\n";}
