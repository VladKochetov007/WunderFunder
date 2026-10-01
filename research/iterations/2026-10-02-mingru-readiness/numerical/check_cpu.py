"""Generated CPU recurrence/gradient checks; long-retention drift is descriptive."""
import argparse,copy,hashlib,json,time,resource
from pathlib import Path
import torch
from minimal_rnn import MinimalRNN,blocked_scan,log_g
HERE=Path(__file__).resolve().parent

def literal(k,v,h):
 rows=[]
 for t in range(k.shape[1]):
  z=torch.sigmoid(k[:,t]);candidate=torch.where(v[:,t]>=0,v[:,t]+.5,torch.sigmoid(v[:,t]))
  h=(1-z)*h+z*candidate;rows.append(h)
 return torch.stack(rows,1),h

def literal_model(m,x,h=None):
 z=torch.nn.functional.gelu(torch.nn.functional.linear((x-m.mean)/m.std,m.inp.weight,m.inp.bias))
 value=z;states=[]
 for i,cell in enumerate(m.cells):
  k=torch.nn.functional.linear(value,cell.gate.weight,cell.gate.bias)
  v=torch.nn.functional.linear(value,cell.candidate.weight,cell.candidate.bias)
  initial=torch.zeros_like(k[:,0]) if h is None else h[i]
  value,last=literal(k,v,initial);states.append(last)
 return m.head(torch.cat((value,z),-1)),torch.stack(states)

def drift(a,b):
 a=a.detach().double();b=b.detach().double();d=a-b
 return dict(max_abs=float(d.abs().max()),rmse=float(d.square().mean().sqrt()),reference_rms=float(b.square().mean().sqrt()),relative_l2=float(d.norm()/b.norm().clamp_min(1e-30)))
def hashes():
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 torch.set_num_threads(1);torch.manual_seed(2047);started=time.perf_counter();checks=[];result=dict(status='RUNNING',source_sha256=hashes(),real_data=False,GPU=False)
 def close(name,x,y,atol=2e-5,rtol=2e-4):
  d=drift(x,y);d.update(name=name,atol=atol,rtol=rtol);checks.append(d)
  torch.testing.assert_close(x.double(),y.double(),atol=atol,rtol=rtol);assert torch.isfinite(x).all()
 try:
  # Literal derivative at selected x>=0 branch: sigmoid(0)*g'(0)=.5.
  k=torch.zeros(1,1,1,requires_grad=True);v=torch.zeros_like(k,requires_grad=True);h=torch.zeros(1,1,requires_grad=True)
  y,_=blocked_scan(k,v,h);g=torch.autograd.grad(y.sum(),(k,v,h))
  for name,actual,want in zip(('gate','candidate','h0'),g,(.125,.5,.5)):close('scalar_zero_'+name,actual,torch.full_like(actual,want),1e-7,0)
  x=torch.tensor([-80.,-1.,-1e-7,0.,1e-7,1.,80.],requires_grad=True);gx=torch.exp(log_g(x));dg=torch.autograd.grad(gx.sum(),x)[0]
  expected=torch.where(x>=0,torch.ones_like(x),torch.sigmoid(x)*(1-torch.sigmoid(x)))
  close('selected_branch_derivatives',dg,expected,2e-6,2e-5)
  for kind in ('zero','positive'):
   k=torch.randn(2,131,7,requires_grad=True);v=torch.randn(2,131,7);v[:,::11]=0;v.requires_grad_()
   h=(torch.zeros(2,7) if kind=='zero' else torch.rand(2,7)+.1).requires_grad_();weight=torch.randn_like(k)
   y,last=blocked_scan(k,v,h);kd,vd,hd=[t.detach().double().requires_grad_() for t in (k,v,h)]
   ref,end=literal(kd,vd,hd);close('cell_'+kind,y,ref)
   got=torch.autograd.grad((y*weight).sum(),(k,v,h));want=torch.autograd.grad((ref*weight.double()).sum(),(kd,vd,hd))
   for label,xx,yy in zip(('k','v','h0'),got,want):close('VJP_'+kind+'_'+label,xx,yy)
   pieces=[];carry=h.detach()
   for lo,hi in ((0,1),(1,63),(63,65),(65,131)):
    part,carry=blocked_scan(k.detach()[:,lo:hi],v.detach()[:,lo:hi],carry);pieces.append(part)
   close('chunk_'+kind,torch.cat(pieces,1),y.detach());close('chunk_state_'+kind,carry,last.detach())
  small=MinimalRNN(11,8,2,23);reference=copy.deepcopy(small).double()
  x=torch.randn(2,41,11,requires_grad=True);h=torch.rand(2,2,8,requires_grad=True);coef=torch.randn(2,41,23)
  y,end=small(x,h);xd=x.detach().double().requires_grad_();hd=h.detach().double().requires_grad_();ref,_=literal_model(reference,xd,hd)
  close('full_model_output',y,ref)
  got=torch.autograd.grad((y*coef).sum(),[x,h,*small.parameters()]);want=torch.autograd.grad((ref*coef.double()).sum(),[xd,hd,*reference.parameters()])
  for i,(xx,yy) in enumerate(zip(got,want)):close('full_model_VJP_'+str(i),xx,yy,3e-5,3e-4)
  # Existing slow-decay discrepancy remains exposed, not relabeled a bug fix.
  retention=[]
  with torch.no_grad():
   for gate in (-20.,-17.,-8.):
    k=torch.full((1,20000,1),gate);v=torch.full_like(k,-17.);h=torch.full((1,1),20.)
    scan,last=blocked_scan(k,v,h);p32,e32=literal(k,v,h);p64,e64=literal(k.double(),v.double(),h.double())
    assert torch.isfinite(scan).all() and (scan>=0).all()
    retention.append(dict(gate=gate,rows=20000,scan_final=float(last.item()),fp32_final=float(e32.item()),fp64_final=float(e64.item()),scan_vs_fp32=drift(scan,p32),scan_vs_fp64=drift(scan,p64),interpretation='descriptive coefficient/accumulation rounding; no equality veto'))
  result.update(status='PASS_GENERATED_CPU',checks=checks,long_retention=retention)
 except Exception as e:
  result.update(status='FAIL_GENERATED_CPU',checks=checks,error=repr(e));raise
 finally:
  result.update(seconds=time.perf_counter()-started,torch_version=torch.__version__,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
