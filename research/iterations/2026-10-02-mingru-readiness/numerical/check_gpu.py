"""One fixed generated GPU AMP/gradient/full-prefix check. No forecast or dataset."""
import argparse,copy,gc,json,resource,sys,time
from pathlib import Path
import torch
from minimal_rnn import MinimalRNN,blocked_scan
from check_cpu import drift,hashes,literal_model
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'src'))
from wnn.nn import RNNModel

def grad_summary(names,a,b):
 out={}
 for group in ('inp','cells','head','gru','inputs'):
  selected=[i for i,n in enumerate(names) if n.split('.')[0]==group]
  if not selected:continue
  aa=torch.cat([a[i].detach().float().reshape(-1) for i in selected]);bb=torch.cat([b[i].detach().float().reshape(-1) for i in selected]);entry=drift(aa,bb)
  entry['cosine']=float(torch.nn.functional.cosine_similarity(aa.double()[None],bb.double()[None],eps=1e-30));out[group]=entry
 return out

def numerical_pair(model,x,target):
 names=['inputs',*[n for n,_ in model.named_parameters()]];outputs=[];grads=[];states=[]
 for amp in (False,True):
  xx=x.detach().clone().requires_grad_()
  with torch.autocast('cuda',dtype=torch.float16,enabled=amp):y,h=model(xx)
  loss=(y.float()-target).square().mean()
  gg=torch.autograd.grad(loss,[xx,*model.parameters()]);assert torch.isfinite(y).all() and all(torch.isfinite(g).all() for g in gg)
  outputs.append(y.detach().float());states.append(h.detach().float());grads.append([g.detach().cpu() for g in gg])
  del xx,y,h,loss,gg
 return dict(outputs=drift(outputs[1],outputs[0]),states=drift(states[1],states[0]),gradients=grad_summary(names,grads[1],grads[0])),outputs,states

@torch.no_grad()
def cpu_stream(model,x):
 cpu=copy.deepcopy(model).cpu().float().eval();rows=[];h=None
 for row in x.detach().cpu().float().unbind(1):
  y,h=cpu.step(row,h);rows.append(y)
 return torch.stack(rows,1),h

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 torch.set_num_threads(1);assert torch.cuda.is_available() and torch.cuda.device_count()==1
 torch.cuda.set_per_process_memory_fraction(.30);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.manual_seed(2048);torch.cuda.manual_seed_all(2048)
 started=time.perf_counter();result=dict(status='RUNNING',source_sha256=hashes(),real_data=False,forecast_fits=0,quality_queries=0,allocator_fraction=.30,TF32=False,AMP='FP16 projections; FP32 scan/state')
 try:
  # AMP does not turn the declared selected zero-branch derivative into a different rule.
  k=torch.zeros(1,1,1,device='cuda',requires_grad=True);v=torch.zeros_like(k,requires_grad=True)
  with torch.autocast('cuda',dtype=torch.float16):z,_=blocked_scan(k,v)
  gv=torch.autograd.grad(z.sum(),v)[0];torch.testing.assert_close(gv,torch.full_like(gv,.5),atol=1e-7,rtol=0)
  model=MinimalRNN(211,256,2,23).cuda().eval()
  with torch.no_grad():model.mean.copy_(torch.linspace(-.2,.2,211,device='cuda'));model.std.copy_(torch.linspace(.7,1.3,211,device='cuda'))
  x=torch.randn(2,2000,211,device='cuda');target=.2*torch.randn(2,2000,23,device='cuda')
  pair,outputs,states=numerical_pair(model,x,target);result['mingru_B2_T2000_AMP_vs_FP32_scan']=pair
  reference,reference_state=cpu_stream(model,x)
  result['mingru_B2_T2000_AMP_vs_FP32_step']=dict(outputs=drift(outputs[1].cpu(),reference),states=drift(states[1].cpu(),reference_state))
  result['mingru_B2_T2000_FP32_scan_vs_FP32_step']=dict(outputs=drift(outputs[0].cpu(),reference),states=drift(states[0].cpu(),reference_state))
  # Ordinary same-precision invariants keep fixed tolerances; AMP-vs-FP32 is reported,
  # not incorrectly treated as a bit-equality requirement or a forecasting veto.
  torch.testing.assert_close(outputs[0].cpu(),reference,atol=3e-5,rtol=3e-4)
  with torch.no_grad():
   pieces=[];h=None
   for lo,hi in ((0,1),(1,63),(63,129),(129,999),(999,2000)):
    with torch.autocast('cuda',dtype=torch.float16):p,h=model(x[:,lo:hi],h)
    pieces.append(p.float())
   chunk=torch.cat(pieces,1);result['AMP_external_chunk_drift']=dict(outputs=drift(chunk,outputs[1]),states=drift(h,states[1]))
   with torch.autocast('cuda',dtype=torch.float16):prefix,_=model(x[:,:512]);modified=x.clone();modified[:,512:]+=7;future,_=model(modified)
   # Same-shaped changed future is strict causal invariance. Shortened-prefix GEMMs
   # may choose different FP16 reduction algorithms, so their drift is descriptive.
   torch.testing.assert_close(future[:,:512].float(),outputs[1][:,:512],atol=0,rtol=0)
   result['AMP_short_prefix_drift']=drift(prefix.float(),outputs[1][:,:512])
  # A small actual211/256/2 model VJP also uses literal sequential recurrence onGPU.
  xx=x[:,:31].detach().clone().requires_grad_();yy,_=model(xx);gg=torch.autograd.grad(yy.float().square().mean(),[xx,*model.parameters()])
  xr=xx.detach().clone().requires_grad_();yr,_=literal_model(model,xr);gr=torch.autograd.grad(yr.float().square().mean(),[xr,*model.parameters()])
  for actual,ref in zip(gg,gr):torch.testing.assert_close(actual,ref,atol=3e-6,rtol=5e-4)
  result['FP32_GPU_literal_model_VJP_max_abs']=max(drift(aa,bb)['max_abs'] for aa,bb in zip(gg,gr))
  del xx,yy,gg,xr,yr,gr,outputs,states,reference,reference_state,pieces,chunk,h,p,prefix,modified,future
  # Existing cuDNN GRU: same generated inputs/target/shape, numerical context only.
  # cuDNN needs training mode for backward; dropout0 makes this the same function.
  gru=RNNModel(211,256,1,0.,23).cuda().train()
  with torch.no_grad():gru.mean.copy_(model.mean);gru.std.copy_(model.std)
  gru_pair,go,gs=numerical_pair(gru,x,target);result['shallow_GRU_B2_T2000_AMP_vs_FP32']=gru_pair
  del gru,go,gs,x,target;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  # One full20k forward/backward and fused AdamW update at B1. This tests possibility
  # at this small batch, NOT B32 feasibility or a full-sequence forecast recipe.
  full=torch.randn(1,20000,211,device='cuda');target=.2*torch.randn(1,20000,23,device='cuda')
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,fused=True);scaler=torch.amp.GradScaler('cuda')
  torch.cuda.synchronize();t0=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.float16):out,h=model(full)
  loss=(out.float()-target).square().mean();scaler.scale(loss).backward();scaler.unscale_(opt)
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  old=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=old
  torch.cuda.synchronize();result['B1_T20000_full_gradient']=dict(seconds=time.perf_counter()-t0,loss=float(loss.detach()),gradient_norm=float(norm),scale_before=old,scale_after=scaler.get_scale(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()),optimizer_steps=1,carry_detached=False)
  assert result['B1_T20000_full_gradient']['all_parameters_finite']
  result.update(status='PASS_GENERATED_NUMERICAL_INVARIANTS_WITH_DRIFT_REPORTED',AMP_drift_is_descriptive=True,not_qualified=['forecast quality','B32 full20k training','trained checkpoint native equivalence','submission runtime'])
 except Exception as e:result.update(status='FAIL_GENERATED_INVARIANT_OR_RESOURCE',error=repr(e));raise
 finally:
  result.update(seconds=time.perf_counter()-started,torch_version=torch.__version__,device=torch.cuda.get_device_name(0),peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
