"""Materialize isolated CTX native fork from the frozen, validated base engine."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'critic/native_order_v2'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:80]
    return source.replace(old, new, 1)


def main():
    assert digest(BASE / 'include/wnn/model.hpp') == '05383c60cb0080d5f5a319e5016adb8124fb3665b098fbc3c123817f010d3798'
    assert digest(BASE / 'src/bindings.cpp') == '497bb9f003763be5c2ac58d7a57cf40d784dadfc096ce2573384f49890113e08'
    out = HERE / 'native_ctx'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    for name in ('include', 'src'):
        shutil.copytree(BASE / name, out / name)
    shutil.copy2(BASE / 'build.sh', out / 'build.sh')
    shutil.copy2(HERE / 'native_ctx.hpp', out / 'include/wnn/ctx_memory.hpp')
    path = out / 'include/wnn/model.hpp'
    source = path.read_text()
    source = source.replace('#include "wnn/transition_features.hpp"', '#include "wnn/ctx_memory.hpp"')
    source = source.replace('TransitionConfig', 'CtxConfig').replace('kTransitionWidth', 'kCtxWidth')
    source = source.replace('TransitionFeatures transition_;', 'CtxMemory ctx_;\n  std::array<float,kCtxRank> q_{};')
    source = source.replace('transition_(w_.feature_config)', 'ctx_(w_.feature_config)').replace('transition_.reset()', 'ctx_.reset()')
    source = replace_once(source, 'std::size_t w_extra = 0;', 'std::size_t w_extra = 0, w_projection = 0;')
    source = replace_once(source,
        '      w.w_extra=w.arena.push(extra.data.data(),extra.data.size());\n    }\n    return w;',
        '''      w.w_extra=w.arena.push(extra.data.data(),extra.data.size());
      if(w.hidden!=128) throw std::invalid_argument("CTX projection requires hidden128");
      const Tensor& projection=get("ctx_projection");
      if(projection.shape!=std::vector<int>{kCtxRank,128}) throw std::invalid_argument("CTX projection shape");
      for(int r=0;r<kCtxRank;++r) for(int c=0;c<128;++c)
        if(projection.data[r*128+c]!=ctx_projection_value(r,c)) throw std::invalid_argument("CTX fixed projection changed");
      w.w_projection=w.arena.push(projection.data.data(),projection.data.size());
    } else if(m.find("ctx_projection")!=m.end()) throw std::invalid_argument("projection without CTX head");
    return w;''')
    source = replace_once(source,
        '      const float* feature = transition_.update(row+48);',
        '''      kernels::gemv(A.at(w_.w_projection),zero_.data(),cat_.data(),q_.data(),kCtxRank,H);
      const float* feature = ctx_.update(q_.data(),x_[kRaw+66]>.5f);''')
    assert 'transition_' not in source and 'Transition' not in source
    path.write_text(source)
    path = out / 'src/bindings.cpp'
    source = path.read_text()
    begin = source.index('wnn::TransitionConfig to_config(')
    end = source.index('\nwnn::Weights to_weights(', begin)
    source = source[:begin] + '''wnn::CtxConfig to_config(const py::object& obj) {
  wnn::CtxConfig c;
  if(obj.is_none()) return c;
  py::dict d=py::cast<py::dict>(obj);
  if(py::cast<std::string>(d["schema"])!="ctxmem01-hadamard16-ema4-32-v1") throw std::invalid_argument("CTX schema");
  if(py::cast<std::string>(d["projection_sha256"])!="0ed6109334d68ec914d5c5e3da6b7f0ed898db05828ace54c76532a4c443f20b") throw std::invalid_argument("CTX projection metadata");
  if(py::cast<std::string>(d["lambda_fast_hex"])!="0x1.ae89f995ad3adp-1" ||
     py::cast<std::string>(d["lambda_slow_hex"])!="0x1.f50765b6e4540p-1") throw std::invalid_argument("CTX decay metadata");
  const std::string arm=py::cast<std::string>(d["arm"]);
  if(arm!="memory" && arm!="current") throw std::invalid_argument("CTX arm");
  c.enabled=true; c.current=arm=="current"; c.validate(); return c;
}
''' + source[end:]
    begin = source.index('  py::class_<wnn::TransitionFeatures>')
    end = source.index('  py::class_<wnn::Predictor>', begin)
    source = source[:begin] + '''  py::class_<wnn::CtxMemory>(m,"ContextMemory")
      .def(py::init([](const py::dict& cfg){return wnn::CtxMemory(to_config(cfg));}))
      .def("reset",&wnn::CtxMemory::reset)
      .def("update",[](wnn::CtxMemory& f,const Array& q,bool event){
        if(q.ndim()!=1 || q.shape(0)!=wnn::kCtxRank) throw std::invalid_argument("expected projected16");
        py::array_t<float> out(wnn::kCtxWidth); const float* got=f.update(q.data(),event);
        std::copy(got,got+wnn::kCtxWidth,out.mutable_data()); return out;
      });

''' + source[end:]
    assert 'Transition' not in source
    path.write_text(source)
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    receipt = {'base_directory': str(BASE), 'base_files': {str(p.relative_to(BASE)):digest(p) for p in sorted(BASE.rglob('*')) if p.is_file() and 'build' not in p.parts and p.suffix in ('.hpp','.cpp','.c','.sh')},
               'generated_files': {str(p.relative_to(out)):digest(p) for p in paths},
               'builder_sha256':digest(Path(__file__)), 'no_existing_source_modified':True}
    (HERE / 'native-source-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(paths),'receipt_sha256':digest(HERE/'native-source-provenance.json')}))


if __name__=='__main__':main()
