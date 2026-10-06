from pathlib import Path
import shutil
R=Path.home()/'WunderFunder-creative/research/iterations'
src=R/'2026-10-05-wave13/native-linear';dst=R/'2026-10-05-wave16/native-sparse'
assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('build'))
p=dst/'include/wnn/vnni_block128.hpp';s=p.read_text()
s=s.replace('WNN_VNNI_TARGET inline void gemv_block128(', 'WNN_VNNI_TARGET inline void gemv_block128_prequant(')
s=s.replace('const float* x,float* y,int out,int in){\n alignas(64)std::uint8_t u[1024];const float sx=quant_input(x,in,u);','const std::uint8_t* u,float sx,float* y,int out,int in){')
needle='\n}\n}'
assert s.endswith(needle+'\n')
s=s[:-len(needle+'\n')]+'''\n}
WNN_VNNI_TARGET inline void gemv_block128(const std::int8_t* P,const float* scale,const std::int32_t* sums,const float* b,const float* x,float* y,int out,int in){
 alignas(64)std::uint8_t u[1024];const float sx=quant_input(x,in,u);
 gemv_block128_prequant(P,scale,sums,b,u,sx,y,out,in);
}
}
''';p.write_text(s)
p=dst/'include/wnn/model.hpp';s=p.read_text()
s=s.replace('  Q8 q8_h1;','''  Q8 q8_h1;
  std::vector<int> block_widths;
  std::vector<std::vector<Q8>> block_ih, block_hh;''')
needle='    finish(q8_h1,2*hidden);has_vnni=true;'
assert needle in s
s=s.replace(needle,'''    finish(q8_h1,2*hidden);has_vnni=true;
    if(!block_widths.empty()) {
      auto extract=[&](const Q8& src,int width,int offset) {
        Q8 q; q.q.resize(3*width*width);q.scale.resize(3*width);
        for(int gate=0;gate<3;++gate)for(int j=0;j<width;++j) {
          const int source=gate*hidden+offset+j, target=gate*width+j;
          q.scale[target]=src.scale[source];
          std::copy_n(src.q.data()+source*hidden+offset,width,q.q.data()+target*width);
        }
        finish(q,width);return q;
      };
      for(std::size_t l=0;l<layers.size();++l) {
        std::vector<Q8> bi,bh;int offset=0;
        for(int width:block_widths) {
          bi.push_back(extract(q8_ih[l],width,offset));bh.push_back(extract(q8_hh[l],width,offset));offset+=width;
        }
        block_ih.push_back(std::move(bi));block_hh.push_back(std::move(bh));
      }
    }''')
needle='    if (w.layers.empty()) throw std::invalid_argument("no GRU layers");'
assert needle in s
s=s.replace(needle,needle+'''
    auto block_meta=m.find("meta.gru_blocks");
    if(block_meta!=m.end()) {
      int sum=0;
      for(float v:block_meta->second.data) {
        if(!std::isfinite(v) || v<64.f || v>1024.f || v!=float(int(v)) || int(v)%64)
          throw std::invalid_argument("GRU block widths must be positive multiples of 64");
        w.block_widths.push_back(int(v));sum+=int(v);
      }
      if(w.block_widths.size()<2 || sum!=w.hidden)throw std::invalid_argument("GRU block width sum mismatch");
      for(const auto& g:w.layers) {
        if(g.in!=w.hidden)throw std::invalid_argument("blocked GRU input width mismatch");
        int offset=0;
        for(int width:w.block_widths) {
          for(int gate=0;gate<3;++gate)for(int j=0;j<width;++j)for(int i=0;i<w.hidden;++i) {
            if(i>=offset && i<offset+width)continue;
            const int at=(gate*w.hidden+offset+j)*w.hidden+i;
            if(w.arena.at(g.w_ih)[at]!=0.f || w.arena.at(g.w_hh)[at]!=0.f)
              throw std::invalid_argument("nonzero weight outside declared GRU block");
          }
          offset+=width;
        }
      }
    }
''')
needle='''      if(w_.has_vnni) {
        vnni::gemv_dispatch'''
assert needle in s
s=s.replace(needle,'''      if(w_.has_vnni && !w_.block_widths.empty()) {
        blocked(w_.block_ih[l],A.at(g.b_ih),layer_in,gi_.data());
        blocked(w_.block_hh[l],A.at(g.b_hh),h,gh_.data());
      } else if(w_.has_vnni) {
        vnni::gemv_dispatch''',1)
needle=' private:\n  // Dense layer'
assert needle in s
s=s.replace(needle,''' private:
  // Quantize globally exactly once, preserving the dense engine's scale and signed dot.
  void blocked(const std::vector<Weights::Q8>& blocks,const float* bias,const float* x,float* y) const {
    alignas(64) std::uint8_t u[1024];alignas(64) std::int8_t q[1024];
    const int backend=w_.quant_backend, H=w_.hidden;
    const float sx=backend==2?vnni::quant_input(x,H,u):vnni::quant_input_scalar(x,H,q);
    int offset=0;
    for(std::size_t b=0;b<blocks.size();++b) {
      const int width=w_.block_widths[b];const auto& a=blocks[b];
      alignas(64) float bb[3072],yy[3072];
      for(int gate=0;gate<3;++gate)std::copy_n(bias+gate*H+offset,width,bb+gate*width);
      if(backend==2)vnni::gemv_block128_prequant(a.packed.data(),a.scale.data(),a.rowsum.data(),bb,u+offset,sx,yy,3*width,width);
      else for(int o=0;o<3*width;++o) {
        const auto* ww=a.q.data()+o*width;
        const auto dot=backend==1?vnni::dot_avx2(q+offset,ww,width):vnni::dot_scalar(q+offset,ww,width);
        const float scale=a.scale[o]*sx;yy[o]=std::fma(float(dot),scale,bb[o]);
      }
      for(int gate=0;gate<3;++gate)std::copy_n(yy+gate*width,width,y+gate*H+offset);
      offset+=width;
    }
  }
  // Dense layer''',1)
p.write_text(s)
print(dst,flush=True)
