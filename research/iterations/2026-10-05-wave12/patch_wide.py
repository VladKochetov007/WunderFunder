from pathlib import Path
D=Path(__file__).parent
p=D/'native-wide/include/wnn/vnni_packed.hpp'
s=p.read_text();s+='''
namespace wnn::vnni {
template<int NV> WNN_VNNI_TARGET inline void projection_strided(const float* __restrict W,const float* __restrict b,const float* __restrict x,float* __restrict y,int in,int stride){
  __m512 a[NV];
#pragma GCC unroll 16
  for(int j=0;j<NV;++j)a[j]=_mm512_loadu_ps(b+16*j);
  for(int i=0;i<in;++i){
    const float xi=x[i];if(xi==0.f)continue;const __m512 v=_mm512_set1_ps(xi);const float* w=W+static_cast<std::size_t>(i)*stride;
#pragma GCC unroll 16
    for(int j=0;j<NV;++j)a[j]=_mm512_fmadd_ps(v,_mm512_loadu_ps(w+16*j),a[j]);
  }
#pragma GCC unroll 16
  for(int j=0;j<NV;++j)_mm512_storeu_ps(y+16*j,a[j]);
}
WNN_VNNI_TARGET inline void projection_wide(const float* W,const float* b,const float* x,float* y,int in,int out){
  projection_strided<16>(W,b,x,y,in,out);
  if(out==384)projection_strided<8>(W+256,b+256,x,y+256,in,out);
  else if(out==512)projection_strided<16>(W+256,b+256,x,y+256,in,out);
  else throw std::invalid_argument("wide projection requires 384 or 512 outputs");
}
}
''';p.write_text(s)
p=D/'native-wide/include/wnn/model.hpp';s=p.read_text();needle='    else if(w_.has_vnni) vnni::projection_portable('
assert needle in s
s=s.replace(needle,'''    else if(w_.has_vnni && w_.quant_backend==2 && (H==384 || H==512)) vnni::projection_wide(A.at(w_.w_inp_t),A.at(w_.b_inp),x_.data(),z,w_.n_in,H);
'''+needle);p.write_text(s)
p=D/'kernel_oracle.cpp';s=p.read_text();needle='  // Input rejection is explicit and never routes to another candidate.'
s=s.replace(needle,'''  for(int out:{384,512}) {
    int in=211;std::vector<float> W(in*out),b(out),x(in),a(out),z(out);
    for(auto&v:W)v=unif(rng);for(auto&v:b)v=unif(rng);for(auto&v:x)v=unif(rng);
    for(int i=0;i<in;i+=3)x[i]=0.f;
    vnni::projection_portable(W.data(),b.data(),x.data(),a.data(),out,in);
    if(vnni::supported()) {
      vnni::projection_wide(W.data(),b.data(),x.data(),z.data(),in,out);
      if(std::memcmp(a.data(),z.data(),out*4))return 22;
    }
  }
'''+needle);(D/'kernel_oracle_wide.cpp').write_text(s)
