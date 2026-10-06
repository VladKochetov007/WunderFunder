"""Generated CPU tensors only: analytical posterior correction and exact old parity."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import torch
from torch.utils.data import DataLoader,TensorDataset
from model import ConsistentAtomRNNModel,consistent_decode
from wnn.atom_head import AtomRNNModel
from train import skip_evaluation,AuditedSampler

def main():
 torch.set_num_threads(1);torch.manual_seed(241)
 # Exactly two atoms; CE minimizer q=(9/17,8/17), metric optimum=(1/5,-19/14).
 g=torch.tensor([[-1.,-.5],[4.,-2.]],dtype=torch.float64)
 p=torch.tensor([[.75,.25]],dtype=torch.float64);c=g.clamp(-2,2);a=c.abs().mean(-1)
 q=p*a;q=q/q.sum(-1,keepdim=True)
 assert torch.allclose(q,torch.tensor([[9/17,8/17]],dtype=torch.float64),atol=1e-15,rtol=0)
 got=consistent_decode(q,g,torch.zeros(1,2,dtype=torch.float64))
 expected=torch.tensor([[1/5,-19/14]],dtype=torch.float64)
 assert torch.allclose(got,expected,atol=1e-14,rtol=0),(got,expected)
 old=(q@(g*g.abs()))/(q@g.abs());assert not torch.allclose(old,expected)
 # CE optimum itself has zero gradient in logits under P*a weighting.
 logq=q.log().requires_grad_();ce=-(p*a*logq.log_softmax(-1)).sum();ce.backward()
 assert logq.grad.abs().max()<1e-15
 # Zero atom contributes zero and zero target support returns regression exactly.
 za=torch.tensor([[0.,0.],[2.,0.]],dtype=torch.float64)
 reg=torch.tensor([[.31,-.73]],dtype=torch.float64)
 z=consistent_decode(torch.tensor([[.9,.1]],dtype=torch.float64),za,reg)
 assert torch.equal(z,torch.tensor([[2.,-.73]],dtype=torch.float64))
 assert torch.equal(consistent_decode(torch.tensor([[1.,0.]],dtype=torch.float64),za,reg),reg)
 # Each decoder initializes exactly the original parameter/buffer bytes and RNG.
 def construct(cls,**kw):
  torch.manual_seed(73);m=cls(7,8,2,0.,23,(17,21),g.float(),**kw);return m,torch.random.get_rng_state()
 ref,rng=construct(AtomRNNModel);original,rng1=construct(ConsistentAtomRNNModel,atom_decoder='original');corrected,rng2=construct(ConsistentAtomRNNModel,atom_decoder='corrected')
 assert torch.equal(rng,rng1) and torch.equal(rng,rng2)
 for m in (original,corrected):
  assert list(ref.state_dict())==list(m.state_dict())
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items())
 ref.eval();original.eval();corrected.eval();hr=ho=hc=None
 for n in (11,7):
  x=torch.randn(2,n,7);r,hr=ref(x,hr);o,ho=original(x,ho);c0,hc=corrected(x,hc)
  assert torch.equal(r,o) and torch.equal(hr,ho)
  assert torch.equal(hr,hc) and torch.equal(r[...,2:],c0[...,2:])
 # No gradient disconnection: new readout trains logits plus original mixture/backbone.
 corrected.zero_grad();loss=corrected(torch.randn(2,13,7))[0][...,:2].square().sum();loss.backward()
 for prefix in ('inp.','gru.','atom_head.','head.','mix_logit'):
  assert any(v.grad is not None and v.grad.abs().sum()>0 for k,v in corrected.named_parameters() if k.startswith(prefix)),prefix
 # Empty nonshuffled loader CPU seed effect equals skipped evaluation exactly.
 ds=TensorDataset(torch.arange(20));dl=DataLoader(ds,batch_size=4,shuffle=False,num_workers=0)
 torch.manual_seed(81);list(dl);want_rng=torch.random.get_rng_state()
 torch.manual_seed(81);skip_evaluation(None,dl,4000,'cpu');assert torch.equal(want_rng,torch.random.get_rng_state())
 # Auditing preserves complete and partial sampler order and RNG without eager reads.
 for count in (3,20):
  torch.manual_seed(91);s=torch.utils.data.RandomSampler(ds);it=iter(s);v=[next(it) for _ in range(count)];rng=torch.random.get_rng_state()
  torch.manual_seed(91);s2=AuditedSampler(torch.utils.data.RandomSampler(ds));it2=iter(s2);v2=[next(it2) for _ in range(count)];assert v==v2 and torch.equal(rng,torch.random.get_rng_state())
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'model.py',HERE/'train.py',Path(__file__),ROOT/'src/wnn/atom_head.py']}
 print(json.dumps(dict(status='PASS_GENERATED_CPU_ONLY',two_atom_q=q.tolist(),corrected=got.tolist(),analytic=expected.tolist(),original=old.tolist(),checks=['weighted_CE_stationary','twoatom_clipped_metric_optimum','zero_atom_and_zero_coordinate_fallback','identical_initial_state_and_RNG','exact_original_forward_and_carried_state','unchanged_aux_logits_and_backbone','corrected_gradient_connected','skipped_evaluation_RNG','lazy_sampler_order_RNG'],source_sha256=sources,data_reads=0,metrics=0,GPU=False)))
if __name__=='__main__':main()
