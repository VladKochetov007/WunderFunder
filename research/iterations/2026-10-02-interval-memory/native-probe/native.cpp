// Generated-only C ABI around the same qualified predictor used by current4.
#include "wnn/model.hpp"
#include <cmath>
#include <string>
static thread_local std::string error;
static wnn::Tensor take(const float*&p,int out,int in=0){std::size_t n=std::size_t(out)*(in?in:1);wnn::Tensor t;t.data.assign(p,p+n);t.shape=in?std::vector<int>{out,in}:std::vector<int>{out};p+=n;return t;}
static wnn::WeightMap weights(int h,int layers,const float*p){
 wnn::WeightMap m;m["mean"]=take(p,211);m["std"]=take(p,211);m["inp.weight"]=take(p,h,211);m["inp.bias"]=take(p,h);
 m["head.0.weight"]=take(p,h,2*h);m["head.0.bias"]=take(p,h);m["head.2.weight"]=take(p,2,h);m["head.2.bias"]=take(p,2);
 for(int l=0;l<layers;++l){std::string s=std::to_string(l);m["gru.weight_ih_l"+s]=take(p,3*h,h);m["gru.bias_ih_l"+s]=take(p,3*h);m["gru.weight_hh_l"+s]=take(p,3*h,h);m["gru.bias_hh_l"+s]=take(p,3*h);}return m;
}
extern "C" {
const char* probe_error(){return error.c_str();}
int probe_vnni(){return wnn::vnni::supported();}
void* probe_create(int h,int layers,const float*w,long long size,const float*extra,long long n_extra,int arm,int quant,int fast){try{
 if((h!=128&&h!=256)||layers<1||layers>2||arm< -1||arm>1)throw std::invalid_argument("shape/arm");
 long long n=422LL+h*212LL+h*(2LL*h+1)+2*(h+1LL)+layers*(6LL*h*h+6*h);if(size!=n)throw std::invalid_argument("ordinary weight count");
 for(long long i=0;i<size;++i)if(!std::isfinite(w[i]))throw std::invalid_argument("nonfinite weights");
 auto*p=new wnn::Predictor(wnn::Weights::from(weights(h,layers,w)),fast,false,false,quant,"vnni");
 if(arm>=0){if(!extra||n_extra!=51808){delete p;throw std::invalid_argument("extra weights");}for(long long i=0;i<n_extra;++i)if(!std::isfinite(extra[i])){delete p;throw std::invalid_argument("nonfinite extra");}p->configure_interval(extra,arm);}return p;
 }catch(const std::exception&e){error=e.what();return nullptr;}}
void probe_destroy(void*p){delete static_cast<wnn::Predictor*>(p);}
void probe_reset(void*p){static_cast<wnn::Predictor*>(p)->reset();}
int probe_run(void*p,const float*x,int rows,float*y,int raw){try{if(!p||rows<0)throw std::invalid_argument("input");auto&m=*static_cast<wnn::Predictor*>(p);for(int t=0;t<rows;++t){if(raw)m.step(x+112LL*t,y+2LL*t);else m.step_prebuilt(x+211LL*t,y+2LL*t);}return 0;}catch(const std::exception&e){error=e.what();return -1;}}
int probe_features(const float*raw,int rows,float*x){wnn::EventFeatures e;for(int t=0;t<rows;++t){std::copy(raw+112LL*t,raw+112LL*(t+1),x+211LL*t);e.update(raw+112LL*t,x+211LL*t+112);}return 0;}
int probe_mixed(void*w,void*a,void*b,void*c,const float*x,int rows,float*y){try{
 wnn::Predictor*p[4]={static_cast<wnn::Predictor*>(w),static_cast<wnn::Predictor*>(a),static_cast<wnn::Predictor*>(b),static_cast<wnn::Predictor*>(c)};int order[4]={2,1,0,3};float values[8];
 for(int t=0;t<rows;++t){for(int k=0;k<4;++k){int m=order[(t%2)?3-k:k];p[m]->step(x+112LL*t,values+2*m);}for(int j=0;j<2;++j){float v=0;for(int m=0;m<4;++m)v=std::fma(.25f,values[2*m+j],v);y[2LL*t+j]=v;}}return 0;
 }catch(const std::exception&e){error=e.what();return -1;}}
}
