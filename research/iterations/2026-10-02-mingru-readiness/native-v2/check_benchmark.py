"""Generated weights only. Numerical checks first, one fixed paired benchmark; no kernel sweep."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,resource,sys,time
import numpy as np
import torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'reference'))
from torch_reference import MinimalRNN
from gru_reference import RNNModel
FP=C.POINTER(C.c_float)
def ptr(x):assert x.dtype==np.float32 and x.flags.c_contiguous;return x.ctypes.data_as(FP)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(m,kind):
 sd=m.state_dict();names=['mean','std','inp.weight','inp.bias','head.0.weight','head.0.bias','head.2.weight','head.2.bias']
 layers=m.layers if kind=='min' else m.gru.num_layers
 for j in range(layers):
  names += ([f'cells.{j}.{p}.{q}' for p in ('gate','candidate') for q in ('weight','bias')] if kind=='min' else [f'gru.{p}_l{j}' for p in ('weight_ih','bias_ih','weight_hh','bias_hh')])
 return np.concatenate([sd[k].detach().cpu().numpy().reshape(-1) for k in names]).astype('float32')
def libsetup(out):
 old=C.CDLL(str(out/'reference.so'));new=C.CDLL(str(out/'optimized.so'))
 old.min_create.argtypes=[C.c_int]*4+[FP,C.c_longlong];old.min_create.restype=C.c_void_p
 old.min_destroy.argtypes=[C.c_void_p];old.min_run.argtypes=[C.c_void_p,FP,C.c_int,FP]
 for name in ('min_set_state','min_get_state'):getattr(old,name).argtypes=[C.c_void_p,FP]
 new.fast_create.argtypes=[C.c_int]*4+[FP,C.c_longlong,C.c_int,C.c_int];new.fast_create.restype=C.c_void_p
 new.fast_destroy.argtypes=[C.c_void_p];new.fast_reset.argtypes=[C.c_void_p,FP];new.fast_state.argtypes=[C.c_void_p,FP];new.fast_run.argtypes=[C.c_void_p,FP,C.c_int,FP,C.c_int]
 new.gru_create.argtypes=[C.c_int,C.c_int,FP,C.c_longlong];new.gru_create.restype=C.c_void_p
 new.gru_create_exact.argtypes=new.gru_create.argtypes;new.gru_create_exact.restype=C.c_void_p
 new.gru_destroy.argtypes=[C.c_void_p];new.gru_reset.argtypes=[C.c_void_p];new.gru_run.argtypes=[C.c_void_p,FP,C.c_int,FP,C.c_int]
 new.build_features.argtypes=[FP,C.c_int,FP];new.prototype_error.restype=C.c_char_p
 new.mixed_run.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,FP,C.c_int,FP]
 assert new.prototype_vnni()==1,'VNNI CPU is required; no alternate backend benchmark'
 return old,new
class Min:
 def __init__(self,lib,m,kind='reference',quant=1,backend=2):
  self.lib,self.kind,self.o=lib,kind,m.head[2].out_features;w=pack(m,'min');self.flat=w
  self.handle=lib.min_create(211,256,2,self.o,ptr(w),len(w)) if kind=='reference' else lib.fast_create(211,256,2,self.o,ptr(w),len(w),quant,backend)
  assert self.handle,kind
 def reset(self,h=None):
  x=None if h is None else np.ascontiguousarray(h,dtype='float32');code=self.lib.min_set_state(self.handle,None if x is None else ptr(x)) if self.kind=='reference' else self.lib.fast_reset(self.handle,None if x is None else ptr(x));assert code==0
 def state(self):
  x=np.empty((2,1,256),np.float32);fn=self.lib.min_get_state if self.kind=='reference' else self.lib.fast_state;assert fn(self.handle,ptr(x))==0;return x
 def run(self,x,raw=False):
  x=np.ascontiguousarray(x,dtype='float32');y=np.empty((len(x),self.o),np.float32)
  code=self.lib.min_run(self.handle,ptr(x),len(x),ptr(y)) if self.kind=='reference' else self.lib.fast_run(self.handle,ptr(x),len(x),ptr(y),int(raw));assert code==0,(self.kind,code);return y
 def close(self):getattr(self.lib,'min_destroy' if self.kind=='reference' else 'fast_destroy')(self.handle)
class Gru:
 def __init__(self,lib,m,exact=False):
  self.lib=lib;w=pack(m,'gru');self.flat=w;create=lib.gru_create_exact if exact else lib.gru_create;self.handle=create(m.gru.hidden_size,m.gru.num_layers,ptr(w),len(w));assert self.handle,lib.prototype_error()
 def reset(self):self.lib.gru_reset(self.handle)
 def run(self,x,raw=False):
  x=np.ascontiguousarray(x,np.float32);y=np.empty((len(x),2),np.float32);assert self.lib.gru_run(self.handle,ptr(x),len(x),ptr(y),int(raw))==0;return y
 def close(self):self.lib.gru_destroy(self.handle)
def describe(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);d=a-b
 return dict(max_abs=float(abs(d).max()),RMS=float(np.sqrt(np.mean(d*d))),relative_L2=float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-15)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--cpu',type=int,required=True);a=ap.parse_args();out=a.output_dir.resolve()
 assert out.is_relative_to(P) and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={a.cpu}
 cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
 limits={k:(cg/k).read_text().strip() for k in ('cpu.max','memory.max','memory.swap.max')}
 assert limits['memory.swap.max']=='0' and int(limits['memory.max'])<=4*(1<<30) and int(limits['cpu.max'].split()[0])<=2*int(limits['cpu.max'].split()[1])
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(False);torch.manual_seed(2043)
 old,new=libsetup(out);start=time.monotonic();checks=[];drifts={}
 model=MinimalRNN(n_out=23).eval();model.mean.copy_(.2*torch.randn(211));model.std.copy_(.5+torch.rand(211))
 two=MinimalRNN(n_out=2).eval();two.load_state_dict({k:(v[:2].clone() if k.startswith('head.2.') else v.clone()) for k,v in model.state_dict().items()})
 shallow=RNNModel(211,256,1,0,2).eval();small=[RNNModel(211,128,2,0,2).eval() for _ in range(3)]
 x=torch.randn(1,2000,211);initial=torch.rand(2,1,256)*2+.1
 extreme=MinimalRNN(n_out=2).eval();extreme.load_state_dict(two.state_dict());levels=torch.tensor([-80.,-20.,0.,20.,80.]).repeat(52)[:256]
 with torch.no_grad():
  for cell in extreme.cells:cell.gate.weight.zero_();cell.gate.bias.copy_(levels);cell.candidate.weight.zero_();cell.candidate.bias.copy_(levels)
 rng=np.random.default_rng(9481);raw=np.empty((20000,112),np.float32);raw[0]=rng.normal(size=112)
 for t in range(1,len(raw)):
  change=rng.random(112)<.04;raw[t]=raw[t-1];raw[t,change]=rng.normal(size=int(change.sum()))
 engineered=np.empty((len(raw),211),np.float32);assert new.build_features(ptr(raw),len(raw),ptr(engineered))==0
 for name,m in [('min23',model),('min2',two),('min_extreme2',extreme),('shallow256',shallow),*[(f'anchor128_{i}',m) for i,m in enumerate(small)]]:np.savez(out/(name+'.npz'),**{k:v.detach().numpy() for k,v in m.state_dict().items()})
 np.savez(out/'generated-inputs.npz',x=x.numpy(),initial=initial.numpy(),raw=raw,engineered=engineered)
 inputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.glob('*.npz')};(out/'generated-before.json').write_text(json.dumps(inputs,indent=2)+'\n')
 def close(name,v,w,atol=2e-5,rtol=2e-5):
  assert np.isfinite(v).all() and np.isfinite(w).all();np.testing.assert_allclose(v,w,atol=atol,rtol=rtol,err_msg=name);checks.append(dict(name=name,atol=atol,rtol=rtol,**describe(v,w)))
 with torch.inference_mode():
  for label,m in [('23',model),('2',two),('extreme2',extreme)]:
   ref=Min(old,m);folded=Min(new,m,'optimized',0);q=Min(new,m,'optimized');scalar=Min(new,m,'optimized',1,0)
   for tag,h0 in [('zero',torch.zeros_like(initial)),('positive',initial)]:
    key=label+'_'+tag;rows=x[0].numpy();carry=h0.clone();pieces=[]
    for row in x.unbind(1):y,carry=m.step(row,carry);pieces.append(y)
    want=torch.stack(pieces,1)[0].numpy();tol=2e-4 if label=='extreme2' else 2e-5
    for obj in (ref,folded,q,scalar):obj.reset(h0.numpy())
    r=ref.run(rows);f=folded.run(rows);v=q.run(rows);end=q.state()
    close(key+'_reference_vs_Torch',r,want,tol);close(key+'_fused_folded_FP32_vs_reference',f,r,tol);close(key+'_FP32_state',folded.state(),carry.numpy(),tol)
    assert np.isfinite(v).all() and np.isfinite(end).all() and (end>=0).all();drifts[key]=dict(output=describe(v,r),state=describe(end,ref.state()))
    q.reset(h0.numpy());parts=[q.run(rows[lo:hi]) for lo,hi in zip([0,1,63,64,511,1999],[1,63,64,511,1999,2000])];close(key+'_INT8_split',np.concatenate(parts),v,0,0);close(key+'_INT8_split_state',q.state(),end,0,0)
    q.reset(h0.numpy());close(key+'_INT8_reset',q.run(rows),v,0,0)
    q.reset(h0.numpy());prefix=q.run(rows[:511]);q.reset(h0.numpy());changed=rows.copy();changed[511:]+=10;close(key+'_causal_prefix',q.run(changed)[:511],prefix,0,0)
    scalar.reset(h0.numpy());q.reset(h0.numpy());close(key+'_same_INT8_scalar_oracle',q.run(rows[:31]),scalar.run(rows[:31]),0,0)
    q.reset(h0.numpy());close(key+'_actual_initial_state',q.state(),h0.numpy(),0,0)
   for obj in (ref,folded,q,scalar):obj.close()
 # Long20000 streaming INT8 drift, including carried nonzero state; no accuracy threshold invented.
 ref=Min(old,two);fast=Min(new,two,'optimized');ref.reset(initial.numpy());fast.reset(initial.numpy());r=ref.run(engineered);v=fast.run(engineered)
 assert np.isfinite(v).all() and (fast.state()>=0).all();drifts['long20000']=dict(output=describe(v,r),state=describe(fast.state(),ref.state()))
 fast.reset();raw_out=fast.run(raw,True);fast.reset();close('raw_feature_wrapper_vs_prebuilt',raw_out,fast.run(engineered),0,0)
 # Benchmark exact same synthetic streams, affinity and process. Alternate paired execution order.
 g=Gru(new,shallow);g_exact=Gru(new,shallow,True);anchors=[Gru(new,m) for m in small];bench={k:[] for k in ('reference_min2000','INT8_min2000','qualified_shallow2000','exact_shallow2000','whole4_student','whole4_exact_shallow','whole4_min')};cpu_bench={k:[] for k in bench}
 rows=x[0].numpy()
 def bench_member(obj,data,rawflag=False):obj.reset();t=time.perf_counter();cpu=time.process_time();obj.run(data,rawflag);return time.perf_counter()-t,time.process_time()-cpu
 def mixed(wide,is_min):
  wide.reset()
  for q in anchors:q.reset()
  y=np.empty((len(raw),2),np.float32);t=time.perf_counter();cpu=time.process_time();code=new.mixed_run(wide.handle,int(is_min),*[q.handle for q in anchors],ptr(raw),len(raw),ptr(y));elapsed=time.perf_counter()-t;cpu_elapsed=time.process_time()-cpu
  assert code==0 and np.isfinite(y).all();return elapsed,cpu_elapsed,y
 for obj in (ref,fast,g,g_exact):bench_member(obj,rows)
 mixed(g,False);mixed(g_exact,False);mixed(fast,True)
 for rep in range(5):
  order=[('reference_min2000',ref),('INT8_min2000',fast),('qualified_shallow2000',g),('exact_shallow2000',g_exact)]
  for name,obj in (order if rep%2==0 else order[::-1]):wall,cpu=bench_member(obj,rows);bench[name].append(wall);cpu_bench[name].append(cpu)
  order=[('whole4_student',g,False),('whole4_exact_shallow',g_exact,False),('whole4_min',fast,True)]
  for name,obj,is_min in (order if rep%2==0 else order[::-1]):wall,cpu,y=mixed(obj,is_min);bench[name].append(wall);cpu_bench[name].append(cpu)
 median={k:float(np.median(v)) for k,v in bench.items()};cpu_median={k:float(np.median(v)) for k,v in cpu_bench.items()};delta=(median['whole4_min']-median['whole4_student'])/len(raw)
 result=dict(status='COMPLETE_GENERATED_MIN_NATIVE_READINESS',checks=checks,INT8_drift=drifts,timings_seconds=bench,process_CPU_seconds=cpu_bench,median_seconds=median,median_process_CPU_seconds=cpu_median,wall_minus_process_CPU_seconds={k:[w-c for w,c in zip(bench[k],cpu_bench[k])] for k in bench},
  descriptive_incremental=dict(seconds_per_raw_row=delta,whole4_ratio=median['whole4_min']/median['whole4_student'],additive_projection_minutes=44.53686675+delta*39400000/60,ratio_projection_minutes=44.53686675*median['whole4_min']/median['whole4_student']),
  benchmark=dict(member_rows=2000,whole4_rows=20000,repetitions=5,order='alternate paired order each repeat; reset every whole run',CPU=a.cpu,cgroup=limits,qualified_shallow='private qualified dynamic INT8/VNNI dense body, operational fast activations; generated H256x1 weights',equal_precision_shallow='same weights and INT8 kernels/state, exact sigmoid/tanh/erf activations; no tuned kernel or architecture change',min='FP32 carried states, exact sigmoid/positive_g and exact-erf GELU; fused2H gates/candidates, qualified INT8 kernels/head0; normalized input folding FP32',whole4='generated raw112, original event211 per member, matched three H128x2 anchors, same alternating member-order and float32 quarter accumulation'),
  limitations=['Random weights/input only: no trained quality, AMP parity or runtime qualification.','Additive and ratio projections assume transfer to the historical44.54min host-run; cache interaction, generated data and Python/ABI/process overhead can differ.','Full4 loops include native event construction and blending but exclude submission Python callback/loader, process startup and scorer; original FP32 reference has engineered211-only boundary.','INT8 drift includes input-normalization folding/reduction-order differences; FP32 folded control bounds this separate arithmetic change.','No kernel sweep or changed tolerance after failure.'],source_sha256=sha(__file__),generated_inventory_sha256=sha(out/'generated-before.json'),library_sha256={'reference':sha(out/'reference.so'),'optimized':sha(out/'optimized.so')},torch_version=torch.__version__,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-start,GPU=False,real_data=False,trained_checkpoints=False,forecast_fit=False)
 for name,rec in inputs.items():assert sha(out/name)==rec['sha256']
 for obj in (ref,fast,g,g_exact,*anchors):obj.close()
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','median_seconds','descriptive_incremental','seconds')}))
if __name__=='__main__':main()
