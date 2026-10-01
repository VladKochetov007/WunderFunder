"""Separately allocation-gated, generated-only full resident-batch parent probe."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['control','ple'],required=True);ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
allocation=json.loads(a.allocation.read_text())
assert allocation.get('SCALARPLE_generated_GPU_probe_authorized') is True and allocation.get('SCALARPLE_generated_GPU_probe_v2_authorized') is True
assert not allocation.get('competition_inputs_authorized',False)
for rel,expected in allocation['sources'].items():assert sha(ROOT/rel)==expected,rel
fixture=P/'generated/gpu_fixture.npz';assert sha(fixture)==allocation['fixture_sha256']
assert not a.output.exists()
sys.path[:0]=[str(P),str(ROOT/'src')]
import numpy as np
import torch
from torch.nn import functional as F
from scalar_model import ScalarRNNModel
from gpu_initial_parity import check_initial_pair
from gpu_scalar_oracle import check_cuda_scalar
from gpu_step_audit import snapshot,step_audit,final_audit
from wnn.nn import RNNModel,weighted_mse_loss
from wnn.outcomes import N_SP_AUX,structure_labels,sequence_class
from wnn.features import FEATURE_NAMES
from wnn.resources import configure_torch
spec=importlib.util.spec_from_file_location('probe_ordinary_trainer',P.parent/'training/train_aux.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
torch.set_num_threads(1)
assert torch.cuda.is_available();device=torch.device('cuda:0')
total=torch.cuda.get_device_properties(0).total_memory
configure_torch(gpu_fraction=min(.9,5.5*(1<<30)/total),duty=1.)
torch.cuda.reset_peak_memory_stats()
scalar_cuda_check=check_cuda_scalar(P.parent/'critic/scalar-independent-forward-fixture')
torch.manual_seed(3)
with np.load(fixture,allow_pickle=False) as z:
 raw_cpu=torch.from_numpy(z['raw'].copy()).repeat(32,1,1)
 y_cpu=torch.from_numpy(z['targets'].copy())[torch.arange(32)%4]
 scored_cpu=torch.from_numpy(z['mask'].copy()).repeat(32,1);scored_cpu[:16]=False
 prob_cpu=torch.from_numpy(z['probability'].copy()).repeat(32,1)
 knots,counts=z['knots'].copy(),z['counts'].copy()
initial_parity=check_initial_pair(raw_cpu,knots,counts,trainer.build_inputs)
model=RNNModel(211,128,2,0.,n_out=23) if a.arm=='control' else ScalarRNNModel(211,knots=knots,counts=counts)
model=model.to(device)
trainer.fit_normalization(model,[(raw_cpu,y_cpu,scored_cpu,prob_cpu)],device)
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,fused=True)
# Same OneCycleLR family and parent update budget, no synthetic schedule search.
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,total_steps=15000,pct_start=.05,final_div_factor=100)
scaler=torch.amp.GradScaler(enabled=True);ema=trainer.FastEMA(model,.999)
x=raw_cpu.to(device);y=y_cpu.to(device);scored=scored_cpu.to(device);prob=prob_cpu.to(device)
# CudaPrefetcher normally retains the following raw batch; reserve actual same-shape tensors.
prefetch=(x.clone(),y.clone(),scored.clone(),prob.clone())
inputs=trainer.build_inputs(x)
steps=torch.arange(20000,device=device)[None].expand(32,-1);valid=steps>=99
main_valid=valid.float()*(prob.square()+.03)
main_valid=trainer.true_mask_weights(main_valid,valid,scored,.03,True)
since=torch.expm1(inputs[...,112+FEATURE_NAMES.index('since_i0px')].float())
main_valid=main_valid*(1/(1+since/32)).clamp_min(.25)
sp,branch,cls=structure_labels(y,sequence_class(y));nx_rows,nx_up,nx_ok=trainer.next_change_labels(y,valid)
nw=main_valid*nx_ok.float();state=teacher_state=None
records=[];completed_updates=0;start=time.monotonic()
scalar_before={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name.startswith('scalar.')}
for index,s in enumerate(range(0,20000,2000)):
 before=snapshot(model,optimizer)  # CPU-only readback before timed region; no RNG/weight operation.
 sl=slice(s,s+2000);torch.cuda.synchronize();wall=time.monotonic();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  teacher,teacher_state=ema.module(inputs[:,sl],teacher_state);teacher_state=trainer.detach_state(teacher_state)
 with torch.autocast('cuda',dtype=torch.float16):
  out,state=model(inputs[:,sl],state);state=trainer.detach_state(state)
  loss=weighted_mse_loss(out[...,:2],y[:,sl],main_valid[:,sl])
  o=out.float();wv=main_valid[:,sl].reshape(-1);ce=0.
  for lo,hi,label in [(2,2+N_SP_AUX,sp),(2+N_SP_AUX,5+N_SP_AUX,branch),(5+N_SP_AUX,9+N_SP_AUX,cls)]:
   ce=ce+F.cross_entropy(o[...,lo:hi].reshape(-1,hi-lo),label[:,sl].reshape(-1),reduction='none')
  loss=loss+.3*(ce*wv).sum()/wv.sum().clamp_min(1e-8)
  at=9+N_SP_AUX;weight=nw[:,sl]
  mse=(o[...,at]-(nx_rows[:,sl]-3)/2)**2
  bce=F.binary_cross_entropy_with_logits(o[...,at+1],nx_up[:,sl],reduction='none')
  loss=loss+.3*((mse+bce)*weight).sum()/weight.sum().clamp_min(1e-8)
  cw=main_valid[:,sl,None]
  loss=loss+(cw*(out[...,:2].float()-teacher[...,:2].float())**2).sum()/cw.sum().clamp_min(1e-8)
 optimizer.zero_grad(set_to_none=True);old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
 grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 scaler.step(optimizer);scaler.update();scheduler.step();ema.update_parameters(model)
 end.record();torch.cuda.synchronize()
 records.append({'step':index,'wall_seconds':time.monotonic()-wall,'device_ms':begin.elapsed_time(end),'loss':float(loss.detach()),'gradient_norm':float(grad_norm.detach()),'scale_before':old_scale,'scale_after':scaler.get_scale(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()})
 print(json.dumps({'diagnostic_step':records[-1]}),flush=True)
 audit,completed_updates=step_audit(before,model,optimizer,records[-1],completed_updates)
 records[-1]['audit']=audit
 print(json.dumps({'step_audit':audit,'step':index}),flush=True)
 assert audit['PASS'],f'v2 fixed AMP warmup/update audit failed: {audit}'
measured=records[3:]
final_state_audit=final_audit(model,ema,optimizer,completed_updates)
print(json.dumps({'final_state_audit':final_state_audit}),flush=True)
assert all(final_state_audit[key] for key in ['model_finite','EMA_finite','optimizer_finite','all_step_counters_completed_updates'])
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
scalar_updates={}
if a.arm=='ple':
 for name,value in scalar_before.items():
  changed=not torch.equal(value,model.state_dict()[name].detach().cpu());scalar_updates[name]=changed
  assert changed if name in ['scalar.table','scalar.projection'] else not changed
result={'arm':a.arm,'status':'PASS','generated_only':True,'competition_inputs_read':False,'fixture_sha256':sha(fixture),'source_sha256':sha(__file__),'allocation_sha256':sha(a.allocation),'actual_shapes':{'raw':list(x.shape),'inputs':list(inputs.shape),'targets':list(y.shape),'chunk':2000,'prefetched_raw':list(prefetch[0].shape)},'steps':records,'measured_steps':list(range(3,10)),'median_wall_seconds':float(np.median([r['wall_seconds'] for r in measured])),'median_device_ms':float(np.median([r['device_ms'] for r in measured])),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds':time.monotonic()-start,'torch_version':torch.__version__,'device_name':torch.cuda.get_device_name(),'backend_flags':{'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'float32_matmul_precision':torch.get_float32_matmul_precision()},'optimizer':'fused AdamW lr.001 wd.01','EMA_decay':.999,'teacher_weight':1.,'aux_struct':.3,'aux_next':.3,'normalization':'ordinary fit_normalization on generated resident batch','initial_AMP_parity':initial_parity,'scalar_CUDA_oracle':scalar_cuda_check,'scalar_state_changes':scalar_updates,'state_all_finite':True,'first_update_baseline_parity_claimed':False,'final_state_audit':final_state_audit,'warmup_skip_steps':[r['step'] for r in records if r['audit']['skipped_by_scaler']],'protocol_version':'v2 prospective warmup scaler checks'}
assert result['peak_reserved_bytes']<=5.5*(1<<30)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
