#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "wnn/kernels.hpp"

namespace wnn {
// SCALARPLE01. FP32 storage/ordered interpolation; double interval fractions.
struct ScalarEmbedding {
 bool enabled=false;
 std::array<int,44> counts{};
 std::vector<float> knots,table,projection;
 static constexpr float inverse_sqrt44=0.15075567364692688f;
 template<class Map> static ScalarEmbedding from(const Map& m,int hidden,int n_in) {
  ScalarEmbedding s;bool any=false;
  const std::array<std::string,5> keys={"scalar.knots","scalar.counts","scalar.raw_indices","scalar.table","scalar.projection"};
  for(const auto& item:m) if(item.first.rfind("scalar.",0)==0) {
   any=true;if(std::find(keys.begin(),keys.end(),item.first)==keys.end())throw std::invalid_argument("unknown scalar state");
  }
  if(!any)return s;
  if(hidden!=128 || n_in!=211)throw std::invalid_argument("scalar architecture");
  for(const auto& key:keys)if(m.find(key)==m.end())throw std::invalid_argument("missing scalar state");
  const auto get=[&](const std::string& key,const std::vector<int>& shape)->const auto& {
   const auto& t=m.at(key);if(t.shape!=shape)throw std::invalid_argument("scalar state shape");
   for(float x:t.data)if(!std::isfinite(x))throw std::invalid_argument("nonfinite scalar state");
   return t.data;
  };
  s.knots=get("scalar.knots",{44,16});s.table=get("scalar.table",{44,16,16});s.projection=get("scalar.projection",{128,16});
  const auto& c=get("scalar.counts",{44});const auto& ids=get("scalar.raw_indices",{44});
  for(int j=0;j<44;++j) {
   if(c[j]<1 || c[j]>16 || c[j]!=std::floor(c[j]))throw std::invalid_argument("scalar count");
   s.counts[j]=static_cast<int>(c[j]);if(ids[j]!=(j<22?j:j+30))throw std::invalid_argument("scalar raw indices");
   const float* k=s.knots.data()+j*16;
   for(int i=1;i<s.counts[j];++i)if(!(double(k[i])-double(k[i-1])>0))throw std::invalid_argument("scalar knot width");
   for(int i=s.counts[j];i<16;++i)if(k[i]!=k[s.counts[j]-1])throw std::invalid_argument("scalar padding");
  }
  s.enabled=true;return s;
 }
 void aggregate(const float* raw,float* out) const {
  if(!enabled)throw std::invalid_argument("scalar embedding disabled");
  std::fill(out,out+16,0.f);
  for(int j=0;j<44;++j) {
   float x=raw[j<22?j:j+30];if(!std::isfinite(x))throw std::invalid_argument("nonfinite scalar input");
   const float* k=knots.data()+j*16;const int n=counts[j];x=std::min(std::max(x,k[0]),k[n-1]);
   const int lo=std::clamp(int(std::upper_bound(k,k+n,x)-k)-1,0,n-1),hi=std::min(lo+1,n-1);
   const float fraction=lo==hi?0.f:static_cast<float>(std::clamp((double(x)-double(k[lo]))/(double(k[hi])-double(k[lo])),0.,1.));
   const float* a=table.data()+(j*16+lo)*16;const float* b=table.data()+(j*16+hi)*16;
   for(int r=0;r<16;++r) {
    // Volatile intermediates forbid FMA/reassociation across these three rounds.
    volatile float difference=b[r]-a[r];volatile float product=difference*fraction;
    const float value=a[r]+product;out[r]=out[r]+value;
   }
  }
  for(int r=0;r<16;++r)out[r]=out[r]*inverse_sqrt44;
 }
 void add(const float* raw,float* encoded) const {
  alignas(64) float value[16];aggregate(raw,value);
  for(int o=0;o<128;++o) {
   const float residual=kernels::dot(projection.data()+o*16,value,16);
   encoded[o]=encoded[o]+residual;
  }
 }
};
} // namespace wnn
