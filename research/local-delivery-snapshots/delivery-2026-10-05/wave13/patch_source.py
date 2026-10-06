from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("linear_baseline"):
        from linear_baseline import LinearBaseline
        return LinearBaseline(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out)
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--linear-baseline", action="store_true")
    args = ap.parse_args()''');p.write_text(s)
p=D/'native-linear/include/wnn/model.hpp';s=p.read_text()
s=s.replace('  int hidden = 0, head_hidden = 0, n_in = 0;','''  int hidden = 0, head_hidden = 0, n_in = 0;
  bool linear_input = false, linear_head = false;
  std::size_t w_linear_head = 0, b_linear_head = 0;''')
s=s.replace('    w.feature_config = config;','''    auto meta = m.find("meta.linear_activations");
    if(meta != m.end()) {
      const auto& flags = meta->second.data;
      if(flags.size()!=2 || (flags[0]!=0.f && flags[0]!=1.f) || (flags[1]!=0.f && flags[1]!=1.f))
        throw std::invalid_argument("linear activation metadata requires two binary flags");
      w.linear_input = flags[0]==1.f; w.linear_head = flags[1]==1.f;
    }
    w.feature_config = config;''')
needle='    return w;'
assert s.count(needle)==1
s=s.replace(needle,'''    if(w.linear_head) {
      const Tensor& a=get("head.0.weight");const Tensor& ab=get("head.0.bias");
      const Tensor& b=get("head.2.weight");const Tensor& bb=get("head.2.bias");
      std::vector<float> fused(2*2*w.hidden), bias(2);
      for(int k=0;k<2;++k) {
        double v=bb.data[k];
        for(int j=0;j<w.head_hidden;++j)v+=double(b.data[k*w.head_hidden+j])*ab.data[j];
        bias[k]=float(v);
        for(int i=0;i<2*w.hidden;++i) {
          double sum=0.;
          for(int j=0;j<w.head_hidden;++j)sum+=double(b.data[k*w.head_hidden+j])*a.data[j*2*w.hidden+i];
          fused[k*2*w.hidden+i]=float(sum);
        }
      }
      w.w_linear_head=w.arena.push(fused.data(),fused.size());
      w.b_linear_head=w.arena.push(bias.data(),bias.size());
    }
'''+needle)
s=s.replace('    gelu(z, H);','    if(!w_.linear_input) gelu(z, H);')
needle='    std::copy(layer_in, layer_in + H, cat_.begin());'
s=s.replace(needle,needle+'''
    if(w_.linear_head) {
      kernels::gemv(A.at(w_.w_linear_head), A.at(w_.b_linear_head), cat_.data(), out, 2, 2*H);
      return;
    }''')
p.write_text(s)
