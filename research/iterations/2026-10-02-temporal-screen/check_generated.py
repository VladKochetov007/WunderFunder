"""Generated CPU only: fixed forward partition, gradient cuts and outer update counts."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import torch
from wnn.nn import RNNModel
from temporal import SegmentedRNN

def objective(p,y,w):return ((p-y).square()*w).sum()/w.sum().clamp_min(1e-8)
def literal(model,x,h,detach):
 parts=[]
 for s in range(0,x.shape[1],2):
  if s and detach:h=h.detach()
  o,h=model(x[:,s:s+2],h);parts.append(o)
 return torch.cat(parts,1),h
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 torch.set_num_threads(1);torch.manual_seed(7102)
 state=torch.get_rng_state();ordinary=RNNModel(5,7,1,0.,2).double();after=torch.get_rng_state()
 torch.set_rng_state(state);control=SegmentedRNN(5,7,1,0.,2,gradient_horizon=8,segment=2).double();assert torch.equal(after,torch.get_rng_state())
 torch.set_rng_state(state);candidate=SegmentedRNN(5,7,1,0.,2,gradient_horizon=2,segment=2).double();assert torch.equal(after,torch.get_rng_state())
 assert list(candidate.state_dict())==list(ordinary.state_dict()) and all(torch.equal(v,ordinary.state_dict()[k]) for k,v in candidate.state_dict().items())
 checks=[]
 def close(name,x,y,atol=1e-10,rtol=1e-8):
  torch.testing.assert_close(x,y,atol=atol,rtol=rtol);checks.append(dict(name=name,max_abs=float((x-y).abs().max())))
 x=torch.randn(3,8,5,dtype=torch.float64);h=torch.randn(1,3,7,dtype=torch.float64);y=torch.randn(3,8,2,dtype=torch.float64)
 w=torch.rand_like(y);w[:,:2]*=.001;w[:,2:4]*=30.;w[:,4:6]=0
 cp,ch=control(x,h);pp,ph=candidate(x,h);assert torch.equal(cp,pp) and torch.equal(ch,ph)
 for name,model,cut in [('control',control,False),('candidate',candidate,True)]:
  inp=x.clone().requires_grad_();hi=h.clone().requires_grad_();p,hn=model(inp,hi)
  loss=objective(p,y,w);g=torch.autograd.grad(loss,[inp,hi,*model.parameters()])
  ref=copy.deepcopy(ordinary);xr=x.clone().requires_grad_();hr=h.clone().requires_grad_();pr,eh=literal(ref,xr,hr,cut)
  # Independent accumulation: every piece uses the SAME full8 denominator.
  ref_loss=sum(((pr[:,s:s+2]-y[:,s:s+2]).square()*w[:,s:s+2]).sum()/w.sum() for s in range(0,8,2))
  gr=torch.autograd.grad(ref_loss,[xr,hr,*ref.parameters()])
  close(name+'_full_denominator_loss',loss,ref_loss)
  for i,(u,v) in enumerate(zip(g,gr)):close(name+'_gradient_'+str(i),u,v)
  close(name+'_state',hn,eh)
 # The wrong micro-normalized loss is detectable on deliberately unequal masses.
 wrong=sum(objective(cp[:,s:s+2],y[:,s:s+2],w[:,s:s+2]) for s in range(0,8,2))/4
 assert abs(float(wrong-objective(cp,y,w)))>1e-4
 # Truncation changes credit only: terminal loss has no gradient before last cut.
 grads=[]
 for model in (candidate,control):
  xx=x.clone().requires_grad_();out,_=model(xx,h);g=torch.autograd.grad(out[:,-1].square().sum(),xx)[0];grads.append(g)
 assert torch.count_nonzero(grads[0][:,:6])==0 and grads[1][:,:6].abs().max()>1e-10
 # Control segmented graph agrees with unsegmented BPTT on the toy CPU recurrence.
 ordinary.load_state_dict(control.state_dict(),strict=True);xx=x.clone().requires_grad_();yy,_=ordinary(xx,h)
 xc=x.clone().requires_grad_();cc,_=control(xc,h)
 close('unsegmented_control_forward',cc,yy)
 gg=torch.autograd.grad(objective(cc,y,w),[xc,*control.parameters()]);rr=torch.autograd.grad(objective(yy,y,w),[xx,*ordinary.parameters()])
 for i,(u,v) in enumerate(zip(gg,rr)):close('unsegmented_control_gradient_'+str(i),u,v)
 # Same two macro updates, schedules and EMA clocks, irrespective of detach arm.
 clocks=[]
 for original in (candidate,control):
  m=copy.deepcopy(original);optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1/(step+1));ema=copy.deepcopy(m);carry=None;steps=0
  for unused in range(2):
   optimizer.zero_grad(set_to_none=True);out,carry=m(x,carry);carry=carry.detach();objective(out,y,w).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step();scheduler.step()
   with torch.no_grad():
    for dest,src in zip(ema.parameters(),m.parameters()):dest.lerp_(src,.001)
   steps+=1
  clocks.append(dict(optimizer_steps=steps,scheduler_steps=scheduler.last_epoch,EMA_updates=steps,lr=scheduler.get_last_lr()))
 assert clocks[0]==clocks[1] and clocks[0]['optimizer_steps']==2 and clocks[0]['scheduler_steps']==2
 result=dict(status='PASS_GENERATED_TEMPORAL_CPU',checks=checks,exact_same_forward_and_state=True,constructor_rng_identical=True,only_credit_cut_differs=True,common_denominator_checked=True,terminal_gradient_cut_checked=True,clocks=clocks,real_data=False,GPU=False,forecast_fit=False,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},limitations='Generated float64 CPU invariants; no AMP/resource/forecast qualification; zero extra serving operators.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
