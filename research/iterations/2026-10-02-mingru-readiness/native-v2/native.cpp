// Generated-only reserve prototype. Private qualified-kernel copies, no production engine edits.
#include "wnn/model.hpp"
#include <memory>
#include <cstring>
using wnn::kernels::Vec;
static thread_local std::string last_error;
struct Dense {
 int in,out;Vec w,b,t;
 wnn::kernels::Int8Vec q;std::vector<float> scale;std::vector<int32_t> rowsum;std::vector<int8_t> packed;
 Dense(int i,int o,const float* W,const float* B):in(i),out(o),w(W,W+size_t(i)*o),b(B,B+o){}
 void quantize(){
  wnn::vnni::validate_width(in);
  for(int o=0;o<out;++o){float m=0;for(int i=0;i<in;++i)m=std::max(m,std::fabs(w[size_t(o)*in+i]));float s=m>0?m/127.f:1.f;
   if(!(s>=std::numeric_limits<float>::min())||!std::isfinite(s))throw std::invalid_argument("weight scale");}
  wnn::kernels::to_int8(w.data(),out,in,q,scale);rowsum.assign(out,0);
  for(int o=0;o<out;++o)for(int i=0;i<in;++i)rowsum[o]+=q[size_t(o)*in+i];
  packed=wnn::vnni::pack128(q.data(),out,in);
 }
 void apply(const float*x,float*y,bool quant,int backend)const{
  if(quant)wnn::vnni::gemv_dispatch(backend,q.data(),packed.data(),scale.data(),rowsum.data(),b.data(),x,y,out,in);
  else wnn::kernels::gemv(w.data(),b.data(),x,y,out,in);
 }
};
static Dense take(const float*&p,int i,int o){Dense v(i,o,p,p+size_t(i)*o);p+=size_t(i)*o+o;return v;}
static void environment(){
 if(std::fegetround()!=FE_TONEAREST||(_mm_getcsr()&((1u<<15)|(1u<<6))))throw std::invalid_argument("requires nearest-even, FTZ/DAZ off");
}
struct Minimal {
 int d,h,l,o,backend;bool quant;Vec mean,sd,state,z,logits,cat,head,input;
 Dense inp,head0,head2;std::vector<Dense> cells;wnn::EventFeatures events;
 Minimal(int D,int H,int L,int O,const float*p,bool Q,int B):d(D),h(H),l(L),o(O),backend(B),quant(Q),
  mean(p,p+D),sd(p+D,p+2*D),state(H*L,0),z(H),logits(2*H),cat(2*H),head(H),input(D),
  inp(take(p+=2*D,D,H)),head0(take(p,2*H,H)),head2(take(p,H,O)){
  environment();if(backend==2&&!wnn::vnni::supported())throw std::invalid_argument("VNNI unavailable");
  // Identical input normalization folding/projection order to the qualified GRU.
  for(int r=0;r<h;++r){double bias=inp.b[r];for(int c=0;c<d;++c){double v=double(inp.w[size_t(r)*d+c])/sd[c];inp.w[size_t(r)*d+c]=float(v);bias-=v*mean[c];}inp.b[r]=float(bias);}
  inp.t=wnn::kernels::transpose(inp.w.data(),h,d);
  for(int layer=0;layer<l;++layer){Dense g=take(p,h,h),v=take(p,h,h);Vec w,b;
   w.insert(w.end(),g.w.begin(),g.w.end());w.insert(w.end(),v.w.begin(),v.w.end());b.insert(b.end(),g.b.begin(),g.b.end());b.insert(b.end(),v.b.begin(),v.b.end());
   cells.emplace_back(h,2*h,w.data(),b.data());if(quant)cells.back().quantize();}
  if(quant)head0.quantize();
 }
 void reset(){std::fill(state.begin(),state.end(),0.f);events.reset();}
 void step(const float*x,float*y){
  if(backend==2)wnn::vnni::projection_registers<16>(inp.t.data(),inp.b.data(),x,z.data(),d);
  else wnn::vnni::projection_portable(inp.t.data(),inp.b.data(),x,z.data(),h,d);
  for(int j=0;j<h;++j)z[j]=wnn::act::gelu_exact(z[j]);const float* current=z.data();
  for(int layer=0;layer<l;++layer){cells[layer].apply(current,logits.data(),quant,backend);float* carry=state.data()+layer*h;
   for(int j=0;j<h;++j){const float k=wnn::act::sigmoid_exact(logits[j]),v=logits[h+j];const float gv=v>=0.f?v+.5f:wnn::act::sigmoid_exact(v);carry[j]=(1.f-k)*carry[j]+k*gv;}current=carry;}
  std::copy(current,current+h,cat.begin());std::copy(z.begin(),z.end(),cat.begin()+h);head0.apply(cat.data(),head.data(),quant,backend);
  for(int j=0;j<h;++j)head[j]=wnn::act::gelu_exact(head[j]);head2.apply(head.data(),y,false,backend);
 }
 void raw(const float*x,float*y){std::copy(x,x+112,input.begin());events.update(x,input.data()+112);step(input.data(),y);}
};
static wnn::Tensor tensor(const float*&p,int out,int in=0){size_t n=size_t(out)*(in?in:1);wnn::Tensor t; t.data.assign(p,p+n);t.shape=in?std::vector<int>{out,in}:std::vector<int>{out};p+=n;return t;}
static wnn::WeightMap gru_weights(int h,int l,const float*p){
 wnn::WeightMap m;m["mean"]=tensor(p,211);m["std"]=tensor(p,211);m["inp.weight"]=tensor(p,h,211);m["inp.bias"]=tensor(p,h);
 m["head.0.weight"]=tensor(p,h,2*h);m["head.0.bias"]=tensor(p,h);m["head.2.weight"]=tensor(p,2,h);m["head.2.bias"]=tensor(p,2);
 for(int j=0;j<l;++j){std::string s=std::to_string(j);m["gru.weight_ih_l"+s]=tensor(p,3*h,h);m["gru.bias_ih_l"+s]=tensor(p,3*h);m["gru.weight_hh_l"+s]=tensor(p,3*h,h);m["gru.bias_hh_l"+s]=tensor(p,3*h);}return m;
}
static void finite(const float*p,long long n){if(!p)throw std::invalid_argument("null");for(long long i=0;i<n;++i)if(!std::isfinite(p[i]))throw std::invalid_argument("nonfinite");}
extern "C" {
const char* prototype_error(){return last_error.c_str();}
int prototype_vnni(){return wnn::vnni::supported()?1:0;}
void* fast_create(int d,int h,int l,int o,const float*w,long long count,int quant,int backend){try{
 if(d!=211||h!=256||l!=2||(o!=2&&o!=23)||(backend!=0&&backend!=2))throw std::invalid_argument("fixed dimensions/backend");
 long long expected=2LL*d+h*(1LL*d+1)+h*(2LL*h+1)+o*(1LL*h+1)+2LL*l*h*(1LL*h+1);
 if(count!=expected)throw std::invalid_argument("weight count");finite(w,count);for(int j=0;j<d;++j)if(w[d+j]<=0)throw std::invalid_argument("std");return new Minimal(d,h,l,o,w,quant,backend);
 }catch(const std::exception&e){last_error=e.what();return nullptr;}}
void fast_destroy(void*p){delete static_cast<Minimal*>(p);}
int fast_reset(void*p,const float*s){try{auto&m=*static_cast<Minimal*>(p);if(s){finite(s,m.h*m.l);for(int j=0;j<m.h*m.l;++j)if(s[j]<0)return -2;std::copy(s,s+m.h*m.l,m.state.begin());m.events.reset();}else m.reset();return 0;}catch(const std::exception&e){last_error=e.what();return -2;}}
int fast_state(void*p,float*s){if(!p||!s)return -1;auto&m=*static_cast<Minimal*>(p);std::copy(m.state.begin(),m.state.end(),s);return 0;}
int fast_run(void*p,const float*x,int rows,float*y,int raw){try{auto&m=*static_cast<Minimal*>(p);if(rows<0)return -1;finite(x,1LL*rows*(raw?112:211));for(int i=0;i<rows;++i){if(raw)m.raw(x+112LL*i,y+1LL*i*m.o);else m.step(x+211LL*i,y+1LL*i*m.o);}return 0;}catch(const std::exception&e){last_error=e.what();return -3;}}
void* gru_create(int h,int l,const float*w,long long count){try{if((h!=128&&h!=256)||l<1||l>2)return nullptr;const long long n=422LL+h*212LL+h*(2LL*h+1)+2*(h+1LL)+l*(6LL*h*h+6*h);if(n!=count)return nullptr;finite(w,n);auto m=gru_weights(h,l,w);return new wnn::Predictor(wnn::Weights::from(m),true,false,false,true,"vnni");}catch(const std::exception&e){last_error=e.what();return nullptr;}}
void* gru_create_exact(int h,int l,const float*w,long long count){try{if((h!=128&&h!=256)||l<1||l>2)return nullptr;const long long n=422LL+h*212LL+h*(2LL*h+1)+2*(h+1LL)+l*(6LL*h*h+6*h);if(n!=count)return nullptr;finite(w,n);auto m=gru_weights(h,l,w);return new wnn::Predictor(wnn::Weights::from(m),false,false,false,true,"vnni");}catch(const std::exception&e){last_error=e.what();return nullptr;}}
void gru_destroy(void*p){delete static_cast<wnn::Predictor*>(p);}
void gru_reset(void*p){static_cast<wnn::Predictor*>(p)->reset();}
int gru_run(void*p,const float*x,int rows,float*y,int raw){try{finite(x,1LL*rows*(raw?112:211));auto&m=*static_cast<wnn::Predictor*>(p);for(int i=0;i<rows;++i){if(raw)m.step(x+112LL*i,y+2LL*i);else m.step_prebuilt(x+211LL*i,y+2LL*i);}return 0;}catch(const std::exception&e){last_error=e.what();return -3;}}
int build_features(const float*raw,int rows,float*x){try{finite(raw,112LL*rows);wnn::EventFeatures f;for(int i=0;i<rows;++i){std::copy(raw+112LL*i,raw+112LL*(i+1),x+211LL*i);f.update(raw+112LL*i,x+211LL*i+112);}return 0;}catch(...){return -1;}}
// Common generated whole4 serving loop, same independent member order policy and blend arithmetic.
int mixed_run(void*wide,int is_min,void*a,void*b,void*c,const float*x,int rows,float*y){try{
 finite(x,112LL*rows);wnn::Predictor*small[3]={static_cast<wnn::Predictor*>(a),static_cast<wnn::Predictor*>(b),static_cast<wnn::Predictor*>(c)};float out[8];int order[4]={2,1,0,3};bool reverse=false;
 for(int t=0;t<rows;++t){for(int k=0;k<4;++k){int member=order[reverse?3-k:k];if(member==0){if(is_min)static_cast<Minimal*>(wide)->raw(x+112LL*t,out);else static_cast<wnn::Predictor*>(wide)->step(x+112LL*t,out);}else small[member-1]->step(x+112LL*t,out+2*member);}reverse=!reverse;
 y[2LL*t]=y[2LL*t+1]=0;for(int m=0;m<4;++m)for(int j=0;j<2;++j)y[2LL*t+j]=std::fma(.25f,out[2*m+j],y[2LL*t+j]);}return 0;
 }catch(const std::exception&e){last_error=e.what();return -3;}}
}
