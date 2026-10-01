// Isolated FP32 causal minGRU. No checkpoint loader, features, quantization or training.
#include "kernels.hpp"
#include "activations.hpp"
#include <memory>
#include <limits>

using wnn::kernels::Vec;
using wnn::kernels::gemv;
using wnn::act::gelu_exact;
using wnn::act::sigmoid_exact;

static float positive_g(float v) { return v >= 0.f ? v + .5f : sigmoid_exact(v); }
struct Linear {
  int in, out;
  Vec w, b;
  Linear(const float*& p, int i, int o):in(i),out(o),w(p,p+i*o),b(p+i*o,p+i*o+o) { p += i*o+o; }
  void apply(const float* x, float* y) const { gemv(w.data(), b.data(), x, y, out, in); }
};
struct Model {
  int d,h,l,o;
  Vec mean,sd,state,norm,z,value,gate,candidate,head_input,head_hidden;
  Linear inp,head0,head2;
  std::vector<Linear> gates,candidates;
  // Flat layout: mean,std,inp.W,inp.b,head0.W,head0.b,head2.W,head2.b,
  // then each layer gate.W,gate.b,candidate.W,candidate.b.
  Model(int D,int H,int L,int O,const float* p):d(D),h(H),l(L),o(O),
    mean(p,p+D),sd(p+D,p+2*D),state(H*L,0.f),norm(D),z(H),value(H),gate(H),candidate(H),
    head_input(2*H),head_hidden(H),inp(p+=2*D,D,H),head0(p,2*H,H),head2(p,H,O) {
    for(int i=0;i<l;++i) { gates.emplace_back(p,h,h); candidates.emplace_back(p,h,h); }
  }
  void step(const float* x,float* y) {
    for(int j=0;j<d;++j) norm[j]=(x[j]-mean[j])/sd[j];
    inp.apply(norm.data(),z.data());
    for(int j=0;j<h;++j) z[j]=gelu_exact(z[j]);
    const float* current=z.data();
    for(int i=0;i<l;++i) {
      gates[i].apply(current,gate.data());
      candidates[i].apply(current,candidate.data());
      float* carry=state.data()+i*h;
      for(int j=0;j<h;++j) {
        float g=sigmoid_exact(gate[j]);
        carry[j]=(1.f-g)*carry[j]+g*positive_g(candidate[j]);
        value[j]=carry[j];
      }
      current=value.data();
    }
    std::copy(value.begin(),value.end(),head_input.begin());
    std::copy(z.begin(),z.end(),head_input.begin()+h);
    head0.apply(head_input.data(),head_hidden.data());
    for(int j=0;j<h;++j) head_hidden[j]=gelu_exact(head_hidden[j]);
    head2.apply(head_hidden.data(),y);
  }
};

extern "C" {
// Buffers are caller-owned, contiguous float32; handles have isolated mutable state.
void* min_create(int d,int h,int l,int o,const float* w,long long count) {
  if(d<=0||d>4096||h<=0||h>4096||l<=0||l>16||o<=0||o>1024||!w) return nullptr;
  long long expected=2LL*d+h*(1LL*d+1)+h*(2LL*h+1)+o*(1LL*h+1)+2LL*l*h*(1LL*h+1);
  if(count!=expected) return nullptr;
  for(long long j=0;j<count;++j) if(!std::isfinite(w[j])) return nullptr;
  for(int j=0;j<d;++j) if(w[d+j]<=0.f) return nullptr;
  try { return new Model(d,h,l,o,w); } catch(...) { return nullptr; }
}
void min_destroy(void* p) { delete static_cast<Model*>(p); }
int min_set_state(void* p,const float* state) {
  if(!p) return -1;
  auto& m=*static_cast<Model*>(p);
  if(!state) { std::fill(m.state.begin(),m.state.end(),0.f); return 0; }
  for(int j=0;j<m.h*m.l;++j) if(!std::isfinite(state[j])||state[j]<0.f) return -2;
  std::copy(state,state+m.h*m.l,m.state.begin()); return 0;
}
int min_get_state(void* p,float* state) {
  if(!p||!state) return -1;
  auto& m=*static_cast<Model*>(p); std::copy(m.state.begin(),m.state.end(),state); return 0;
}
int min_run(void* p,const float* x,int rows,float* y) {
  if(!p||!x||!y||rows<0) return -1;
  auto& m=*static_cast<Model*>(p);
  for(long long j=0;j<1LL*rows*m.d;++j) if(!std::isfinite(x[j])) return -2;
  for(int t=0;t<rows;++t) m.step(x+1LL*t*m.d,y+1LL*t*m.o);
  return 0;
}
int min_recur(const float* k,const float* v,const float* initial,int rows,int h,float* output) {
  if(!k||!v||!initial||!output||rows<=0||h<=0) return -1;
  for(int j=0;j<h;++j) if(!std::isfinite(initial[j])||initial[j]<0.f) return -2;
  for(long long j=0;j<1LL*rows*h;++j) if(!std::isfinite(k[j])||!std::isfinite(v[j])) return -2;
  for(int t=0;t<rows;++t) for(int j=0;j<h;++j) {
    long long at=1LL*t*h+j;
    float z=sigmoid_exact(k[at]),prev=t?output[at-h]:initial[j];
    output[at]=(1.f-z)*prev+z*positive_g(v[at]);
  }
  return 0;
}
}
