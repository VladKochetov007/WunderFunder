"""Generated CPU tests only; no competition files or metric routines."""
from pathlib import Path
import copy,hashlib,json,math,os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
sys.path[:0]=[str(P),str(ROOT/'src')]
import numpy as np
import torch
from scalar_model import ScalarRNNModel,ScalarAdapter,canonicalize_knots,validate_knots,_interpolate,_InterpolationSum,INV_SQRT44,RAW_INDICES
from wnn.nn import RNNModel
torch.set_num_threads(1)
q=np.tile(np.linspace(-2,2,16),(44,1));q[0]=0;q[1]=np.repeat([-1.,1.],8);q[2,5:9]=0
q[3,1]=np.nextafter(q[3,0],np.float64(np.inf))
k,c=canonicalize_knots(q);assert list(c[:3])==[1,2,13]
assert c[3]==15
generator=torch.Generator().manual_seed(987)
x=torch.randn((2,31,211),generator=generator)
torch.manual_seed(713);base=RNNModel(211,128,2,0.,n_out=23);rng_base=torch.get_rng_state().clone()
torch.manual_seed(713);candidate=ScalarRNNModel(211,knots=k,counts=c);rng_candidate=torch.get_rng_state().clone()
assert torch.equal(rng_base,rng_candidate)
for name,value in base.state_dict().items():assert torch.equal(value,candidate.state_dict()[name]),name
target=torch.randn((2,31,23),generator=generator)
yb,hb=base(x);yc,hc=candidate(x)
assert torch.equal(yb,yc) and torch.equal(hb,hc)
((yb-target)**2).sum().backward();((yc-target)**2).sum().backward()
for name,p in base.named_parameters():assert torch.equal(p.grad,dict(candidate.named_parameters())[name].grad),name
assert candidate.scalar.projection.grad.abs().sum()>0 and torch.count_nonzero(candidate.scalar.table.grad)==0
with torch.no_grad():candidate.scalar.projection.add_(candidate.scalar.projection.grad,alpha=-1e-4)
candidate.zero_grad();y,_=candidate(x);((y-target)**2).sum().backward()
assert torch.isfinite(candidate.scalar.table.grad).all() and candidate.scalar.table.grad.abs().sum()>0

# Dense eager autograd is separate from the custom backward; generated small tensors only.
raw=x.reshape(-1,211)[:,RAW_INDICES].contiguous();knots=torch.from_numpy(k)
table=torch.randn((44,16,16),generator=generator,requires_grad=True);dense=table.detach().clone().requires_grad_()
go=torch.randn((len(raw),16),generator=generator)
a=_InterpolationSum.apply(raw,knots,table,tuple(c.tolist()));b=_interpolate(raw,knots,dense,tuple(c.tolist()),False)[0]
assert torch.equal(a,b);(a*go).sum().backward();(b*go).sum().backward()
grad_error=float((table.grad-dense.grad).abs().max());assert torch.allclose(table.grad,dense.grad,atol=5e-6,rtol=2e-5),grad_error

# Literal NumPy oracle with independently written scalar interval search and rounding.
oracle_raw=np.zeros((49,112),np.float32);boundary=[]
for v in k[3]:boundary.extend([np.nextafter(v,np.float32(-np.inf)),v,np.nextafter(v,np.float32(np.inf))])
boundary.append(np.finfo(np.float32).max)
for j,idx in enumerate(RAW_INDICES):oracle_raw[:,idx]=np.roll(np.asarray(boundary,np.float32),j)
oracle_raw[0,RAW_INDICES[0]]=-np.finfo(np.float32).max
oracle_raw[1,RAW_INDICES[0]]=np.nextafter(np.float32(0),np.float32(1))
tab=candidate.scalar.table.detach().numpy();expected=np.zeros((49,16),np.float32)
for t in range(49):
 for j,idx in enumerate(RAW_INDICES):
  value=min(max(float(oracle_raw[t,idx]),float(k[j,0])),float(k[j,c[j]-1]));lo=0
  while lo+1<c[j] and float(k[j,lo+1])<=value:lo+=1
  hi=min(lo+1,int(c[j])-1)
  alpha=np.float32(0 if hi==lo else (value-float(k[j,lo]))/(float(k[j,hi])-float(k[j,lo])))
  for r in range(16):
   difference=np.float32(tab[j,hi,r]-tab[j,lo,r]);product=np.float32(difference*alpha)
   expected[t,r]=np.float32(expected[t,r]+np.float32(tab[j,lo,r]+product))
 expected[t]*=np.float32(INV_SQRT44)
actual=candidate.scalar.aggregated(torch.from_numpy(oracle_raw)).detach().numpy();assert np.array_equal(actual,expected)
rejected=0
for bad in [np.nan,np.inf,-np.inf]:
 xx=oracle_raw.copy();xx[0,0]=bad
 try:candidate.scalar.aggregated(torch.from_numpy(xx))
 except ValueError:rejected+=1
 else:raise AssertionError('nonfinite accepted')
for invalid in ['duplicate','count0','padding']:
 kk=k.copy();cc=c.copy()
 if invalid=='duplicate':kk[3,2]=kk[3,1]
 if invalid=='count0':cc[3]=0
 if invalid=='padding':kk[0,-1]=1
 try:validate_knots(kk,cc)
 except ValueError:rejected+=1
 else:raise AssertionError('invalid knots accepted')

# Full/checkpoint/prefix/chunk checks preserve all adapter state.
candidate.eval();full,hidden=candidate(x);first,state=candidate(x[:,:11]);last,hidden2=candidate(x[:,11:],state)
chunk_error=float((full-torch.cat([first,last],1)).detach().abs().max());assert chunk_error<2e-6
changed=x.clone();changed[:,11:]*=-7;future,_=candidate(changed);assert torch.equal(full[:,:11],future[:,:11])
clone=ScalarRNNModel(211,knots=k,counts=c);clone.load_state_dict(candidate.state_dict(),strict=True);clone.eval()
again,_=clone(x);assert torch.equal(full,again)
saved=copy.deepcopy(clone.scalar.state_dict())
for name,param in clone.named_parameters():
 if name.startswith(('inp','gru','scalar')):param.requires_grad_(False)
optimizer=torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad],lr=.0005)
optimizer.zero_grad();out,_=clone(x);out.square().mean().backward();optimizer.step()
for key,value in saved.items():assert torch.equal(value,clone.scalar.state_dict()[key])
bad_state=copy.deepcopy(candidate.state_dict());bad_state['scalar.counts'][3]-=1
try:clone.load_state_dict(bad_state,strict=True)
except RuntimeError:rejected+=1
else:raise AssertionError('changed immutable knots/counts accepted')

art=P/'generated';art.mkdir(exist_ok=True)
weights={key:value.detach().cpu().float().numpy() for key,value in candidate.state_dict().items()}
np.savez(art/'model.npz',**weights);np.save(art/'raw.npy',oracle_raw);np.save(art/'scalar_features.npy',expected)
from wnn.features import event_features_torch
native_raw=torch.randn((1,641,112),generator=generator)*.5
native_inputs=torch.cat([native_raw,event_features_torch(native_raw)],dim=-1)
with torch.no_grad():native_expected=candidate(native_inputs)[0][0,:,:2].numpy()
np.save(art/'stream_raw.npy',native_raw[0].numpy());np.save(art/'stream_expected.npy',native_expected)
np.savez(art/'knots.npz',knots=k,counts=c)
result={'status':'PASS','competition_inputs_read':False,'GPU_used':False,'generated_only':True,'ordinary_state_and_global_RNG_exact':True,'zero_adapter_output_hidden_base_gradients_exact':True,'projection_then_table_gradients_nonzero':True,'dense_custom_gradient_max_abs':grad_error,'literal_scalar_oracle_rows':49,'literal_scalar_oracle_exact':True,'rejected_cases':rejected,'chunk_max_abs':chunk_error,'prefix_future_invariance_exact':True,'checkpoint_roundtrip_exact':True,'H2_scalar_frozen_exact':True,'torch_version':torch.__version__,'scalar_source_sha256':hashlib.sha256((P/'scalar_model.py').read_bytes()).hexdigest(),'spec_sha256':hashlib.sha256((P/'spec.md').read_bytes()).hexdigest()}
(P/'cpu-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
