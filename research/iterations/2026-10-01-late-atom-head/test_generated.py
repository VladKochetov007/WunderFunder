"""Generated CPU checks only; no dataset/checkpoint read and no GPU use."""
import copy,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import torch
from wnn.nn import RNNModel,weighted_pooled_corr_loss
from model import LateAtom,corrected_decode
def main():
 torch.set_num_threads(1);torch.manual_seed(734)
 atoms=torch.tensor([[-.5,-.5],[.5,-1.5],[2.,-.5],[-1.5,2.]])
 baseline=RNNModel(7,8,1,0,23).eval();before=torch.get_rng_state().clone()
 candidate=LateAtom(copy.deepcopy(baseline),atoms);assert torch.equal(before,torch.get_rng_state())
 control=LateAtom(copy.deepcopy(baseline),atoms)
 assert all(torch.equal(a,b) for a,b in zip(candidate.state_dict().values(),control.state_dict().values()))
 frozen={k:v.clone() for k,v in baseline.state_dict().items()};h0=h1=None
 for length in (11,7):
  x=torch.randn(3,length,7);p,h0=baseline(x,h0);q,_,h1=candidate(x,h1)
  assert torch.equal(p[...,:2],q) and torch.equal(h0,h1)
 assert not any(p.requires_grad for p in candidate.baseline.parameters())
 # At gain0 the correction's gain derivative exists and is nonzero.
 x=torch.randn(3,29,7);labels=torch.randint(0,4,(3,29));y=atoms[labels];valid=torch.ones(3,29,dtype=torch.bool)
 for model,coef in ((candidate,.5),(control,0.)):
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
  for step in range(3):
   opt.zero_grad(set_to_none=True);pred,logits,_=model(x);main=weighted_pooled_corr_loss(pred,y,valid)
   w=y.abs().mean(-1);ce=(torch.nn.functional.cross_entropy(logits.reshape(-1,4),labels.reshape(-1),reduction='none').reshape_as(w)*w).sum()/w.sum()
   (main+coef*ce).backward()
   assert torch.isfinite(model.residual.gain.grad) and model.residual.gain.grad.abs()>0
   if step==0:
    grads=[p.grad for p in model.residual.mlp.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert (any(torch.count_nonzero(g)>0 for g in grads) if coef else all(torch.count_nonzero(g)==0 for g in grads))
   opt.step()
   assert all(torch.equal(v,frozen[k]) for k,v in model.baseline.state_dict().items())
  assert model.residual.gain!=0
 # Full wrapper strict reload plus causal chunk/prefix consistency with nonzero gain.
 clone=LateAtom(copy.deepcopy(baseline),atoms);clone.load_state_dict(candidate.state_dict(),strict=True)
 full=candidate(x)[0];assert torch.equal(full,clone(x)[0]);h=None;parts=[]
 for lo,hi in ((0,8),(8,19),(19,29)):
  p,_,h=candidate(x[:,lo:hi],h);parts.append(p)
 assert torch.allclose(torch.cat(parts,1),full,atol=2e-7,rtol=1e-6)
 assert torch.allclose(candidate(x[:,:13])[0],full[:,:13],atol=2e-7,rtol=1e-6)
 bad=candidate.state_dict();bad.pop('baseline.inp.weight')
 try:clone.load_state_dict(bad,strict=True)
 except RuntimeError:pass
 else:raise AssertionError('partial state accepted')
 # Weighted CE q proportional P*a; corrected moment matches independent literal oracle.
 g=torch.tensor([[-1.,-1.],[1.,-3.],[4.,-2.],[-3.,2.]])
 c=g.clamp(-2,2);P=torch.tensor([.1,.2,.3,.4]);a=c.abs().mean(-1);q=P*a;q/=q.sum()
 want=torch.tensor([sum(P[k]*abs(c[k,j])*c[k,j] for k in range(4))/sum(P[k]*abs(c[k,j]) for k in range(4)) for j in range(2)])
 assert torch.allclose(corrected_decode(q.log()[None],g)[0],want,atol=2e-7,rtol=1e-6)
 print(json.dumps(dict(status='PASS_GENERATED_CPU_ONLY',actual_data=False,GPU=False,checks=['paired_initialization_and_RNG','zero_gain_original_forward_and_carried_state','nonzero_gain_derivative','CE0_vs_CEhalf_initial_branch_gradients','frozen_all_original_tensors_after_updates','strict_reload_rejects_partial','prefix_chunk_causality','weighted_CE_decode_oracle'],zero_identity_scope='numeric exact; universal signed-zero byte preservation not asserted')))
if __name__=='__main__':main()
