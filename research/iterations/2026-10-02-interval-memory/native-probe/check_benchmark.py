"""Generated-only parity then one fixed native cost probe; no real weights/data."""
import argparse,ctypes as C,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;I=P.parent;ROOT=I.parents[2]
sys.path[:0]=[str(I),str(ROOT/'src')]
from interval_memory import IntervalMemoryModel
from wnn.nn import RNNModel
FP=C.POINTER(C.c_float)
def ptr(x):assert x.dtype==np.float32 and x.flags.c_contiguous;return x.ctypes.data_as(FP)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ordinary(m):
 s=m.state_dict();names=['mean','std','inp.weight','inp.bias','head.0.weight','head.0.bias','head.2.weight','head.2.bias']
 for l in range(m.gru.num_layers):names.extend(f'gru.{k}_l{l}' for k in ('weight_ih','bias_ih','weight_hh','bias_hh'))
 return np.concatenate([(s[k][:2] if k.startswith('head.2.') else s[k]).detach().numpy().ravel() for k in names]).astype(np.float32)
def extra(m):
 s=m.state_dict();names=['interval_proj.weight','interval_proj.bias','event_gru.weight_ih_l0','event_gru.bias_ih_l0','event_gru.weight_hh_l0','event_gru.bias_hh_l0','extra_weight']
 return np.concatenate([s[k].detach().numpy().ravel() for k in names]).astype(np.float32)
class Engine:
 def __init__(self,lib,model,arm=-1,quant=True,fast=True,zero=False):
  self.lib=lib;w=ordinary(model);e=extra(model) if arm>=0 else None
  if zero:e=e.copy();e[-256*96:]=0
  self.handle=lib.probe_create(model.gru.hidden_size,model.gru.num_layers,ptr(w),len(w),None if e is None else ptr(e),0 if e is None else len(e),arm,int(quant),int(fast))
  assert self.handle,lib.probe_error()
 def reset(self):self.lib.probe_reset(self.handle)
 def run(self,x,raw=False):
  x=np.ascontiguousarray(x,np.float32);y=np.empty((len(x),2),np.float32)
  assert self.lib.probe_run(self.handle,ptr(x),len(x),ptr(y),int(raw))==0,self.lib.probe_error()
  assert np.isfinite(y).all();return y
 def close(self):self.lib.probe_destroy(self.handle)
def raw_stream(period,rows=5000):
 rng=np.random.default_rng(902+period);raw=np.empty((rows,112),np.float32);raw[0]=rng.normal(0,.3,112)
 for t in range(1,rows):
  raw[t]=raw[t-1];changed=rng.random(112)<.025;changed[[0,11]]=False
  raw[t,changed]=rng.normal(0,.3,int(changed.sum()))
  if t%period==0:raw[t,[0,11]]=rng.normal(0,.3,2)
 return raw
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--cpu',type=int,required=True);a=ap.parse_args();out=a.output_dir.resolve()
 assert out.is_relative_to(P) and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={a.cpu}
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_flush_denormal(False);torch.manual_seed(2026100240)
 lib=C.CDLL(str(out/'probe.so'));lib.probe_error.restype=C.c_char_p
 lib.probe_create.argtypes=[C.c_int,C.c_int,FP,C.c_longlong,FP,C.c_longlong,C.c_int,C.c_int,C.c_int];lib.probe_create.restype=C.c_void_p
 for f in ('probe_reset','probe_destroy'):getattr(lib,f).argtypes=[C.c_void_p]
 lib.probe_run.argtypes=[C.c_void_p,FP,C.c_int,FP,C.c_int];lib.probe_features.argtypes=[FP,C.c_int,FP]
 lib.probe_mixed.argtypes=[C.c_void_p]*4+[FP,C.c_int,FP]
 assert lib.probe_vnni()==1,'Fixed qualified VNNI path required; no fallback trial'
 start=time.monotonic();checks=[];drifts={};plain=RNNModel(211,256,1,0.,23).eval()
 with torch.no_grad():plain.mean.normal_(0,.05);plain.std.uniform_(.7,1.3)
 accumulated=IntervalMemoryModel('accumulated').eval();accumulated.load_plain_baseline(plain.state_dict())
 with torch.no_grad():accumulated.extra_weight.normal_(0,.02)
 instantaneous=IntervalMemoryModel('instantaneous').eval();instantaneous.load_state_dict(accumulated.state_dict(),strict=True)
 small=[RNNModel(211,128,2,0.,2).eval() for _ in range(3)]
 streams={str(p):raw_stream(p) for p in (16,2)}
 for name,m in [('plain',plain),('accumulated',accumulated),('instantaneous',instantaneous),*[(f'anchor{i}',m) for i,m in enumerate(small)]]:
  np.savez(out/(name+'.npz'),**{k:v.detach().numpy() for k,v in m.state_dict().items()})
 np.savez(out/'generated-raw.npz',**streams)
 inputs={p.name:sha(p) for p in out.glob('*.npz')}
 def close(name,x,y,atol=3e-5,rtol=1e-4):
  np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,err_msg=name)
  checks.append(dict(name=name,max_abs=float(abs(x-y).max()),RMSE=float(np.sqrt(np.mean((x.astype(float)-y)**2))),atol=atol,rtol=rtol))
 for label,m,arm in [('accumulated',accumulated,0),('instantaneous',instantaneous,1)]:
  exact=Engine(lib,m,arm,False,False);operational=Engine(lib,m,arm);zero=Engine(lib,m,arm,zero=True);base=Engine(lib,plain)
  raw=streams['16'][:257];features=np.empty((len(raw),211),np.float32);lib.probe_features(ptr(raw),len(raw),ptr(features))
  with torch.no_grad():expected=m(torch.from_numpy(features)[None])[0][0,:,:2].numpy()
  close(label+'_FP32_native_vs_Torch',exact.run(features),expected)
  operational.reset();q=operational.run(raw,True)
  drifts[label]=dict(max_abs=float(abs(q-expected).max()),RMSE=float(np.sqrt(np.mean((q-expected)**2))),meaning='Includes existing INT8/folding/fast-activation approximation; no quality claim')
  close(label+'_zero_extra_qualified_base',zero.run(raw,True),base.run(raw,True),0,0)
  operational.reset();parts=[operational.run(raw[l:r],True) for l,r in ((0,1),(1,17),(17,17),(17,133),(133,257))]
  close(label+'_chunk_state',np.concatenate(parts),q,0,0)
  operational.reset();close(label+'_reset',operational.run(raw,True),q,0,0)
  future=raw.copy();future[101:]+=7;operational.reset();close(label+'_future_prefix',operational.run(future,True)[:101],q[:101],0,0)
  # Quiet and every-row-event cases exercise closure semantics independently of timing workload.
  for period in (1,10000):
   r=raw_stream(period,97);f=np.empty((97,211),np.float32);lib.probe_features(ptr(r),97,ptr(f));exact.reset()
   with torch.no_grad():expected=m(torch.from_numpy(f)[None])[0][0,:,:2].numpy()
   close(label+'_event_period'+str(period),exact.run(r,True),expected)
  for obj in (exact,operational,zero,base):obj.close()
 engines={'plain':Engine(lib,plain),'accumulated':Engine(lib,accumulated,0),'instantaneous':Engine(lib,instantaneous,1)}
 anchors=[Engine(lib,m) for m in small];timings={};projections={}
 def timed(obj,raw,mixed):
  obj.reset()
  if mixed:
   for q in anchors:q.reset()
  y=np.empty((len(raw),2),np.float32);t=time.perf_counter();cpu=time.process_time()
  code=lib.probe_mixed(obj.handle,*[q.handle for q in anchors],ptr(raw),len(raw),ptr(y)) if mixed else lib.probe_run(obj.handle,ptr(raw),len(raw),ptr(y),1)
  wall=time.perf_counter()-t;cpu=time.process_time()-cpu
  assert code==0 and np.isfinite(y).all();return dict(wall=wall,CPU=cpu,checksum=float(y.astype(np.float64).sum()))
 for period,raw in streams.items():
  for obj in engines.values():timed(obj,raw[:256],True)
  timing={kind:{name:[] for name in engines} for kind in ('member','whole4')}
  for repeat in range(3):
   order=list(engines) if repeat%2==0 else list(reversed(engines))
   for name in order:
    for kind in ('member','whole4'):timing[kind][name].append(timed(engines[name],raw,kind=='whole4'))
  timings[period]=timing
  med={kind:{name:float(np.median([r['wall'] for r in rows])) for name,rows in arms.items()} for kind,arms in timing.items()}
  projections[period]=dict(median_seconds=med,whole4_ratios={name:med['whole4'][name]/med['whole4']['plain'] for name in ('accumulated','instantaneous')},
   indicative_minutes={name:44.53686675*med['whole4'][name]/med['whole4']['plain'] for name in ('accumulated','instantaneous')})
 for obj in [*engines.values(),*anchors]:obj.close()
 for name,h in inputs.items():assert sha(out/name)==h
 result=dict(status='COMPLETE_GENERATED_INTERVAL_NATIVE_PROBE',checks=checks,operational_drift=drifts,timings=timings,projections=projections,
  benchmark=dict(rows_per_trace=5000,repetitions=3,event_periods=[16,2],order='A/B/C then C/B/A then A/B/C; same raw traces and reset per trace',CPU=a.cpu,
    ordinary='Qualified dynamic INT8/VNNI GRU/head kernels; FP32 input projection; operational fast activations',
    interval='Identical ordinary trunk/head plus FP32 added projection,event-GRU,extra matrix; FP32 interval carry; same operational activation family',
    whole4='Same three generated h128x2 anchors, per-member raw event211, alternating member order and fixed quarter arithmetic'),
  source_sha256=sha(__file__),library_sha256=sha(out/'probe.so'),generated_sha256=inputs,seconds=time.monotonic()-start,
  real_data=False,trained_checkpoints=False,GPU=False,forecast_fits=0,metrics=0,
  limitations=['Generated traces/weights only; ratios and44.54min extrapolation are early cost evidence, not scorer qualification.',
    'Native loop includes raw features and whole4 arithmetic but excludes Python callback, loader, startup and final package.',
    'FP32 added matrices are a fixed conservative first implementation, no precision/kernel sweep; trained/native parity remains untested.',
    'Parent/H2 training and quality are separate; no automatic promotion or branch selection follows this receipt.'])
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','projections','seconds')}))
if __name__=='__main__':main()
