"""Generated-only factory RNG, strict warmstart forecast/norm, H2 freeze checks."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'research/iterations/2026-10-02-interval-memory')]
import torch
from train_warm import make_model,strict_initialize,READOUT,sha
from interval_memory import IntervalMemoryModel
from wnn.nn import RNNModel
def main():
 torch.set_num_threads(1);config={'hidden':256,'layers':1,'dropout':0};checks=[]
 for stage in ('parent','H2'):
  torch.manual_seed(3);reference=IntervalMemoryModel('instantaneous',23);after=torch.get_rng_state().clone();expected_order=torch.randperm(257)
  torch.manual_seed(3);ordinary=make_model(config,211,23)
  assert torch.equal(torch.get_rng_state(),after);assert torch.equal(torch.randperm(257),expected_order)
  assert set(ordinary.state_dict())=={'mean','std','inp.weight','inp.bias','gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
  assert all(torch.equal(v,reference.state_dict()[k]) for k,v in ordinary.state_dict().items())
  source=RNNModel(211,256,1,0.,n_out=23);source.mean.copy_(torch.linspace(-.2,.2,211));source.std.copy_(torch.linspace(.7,1.3,211))
  rng=torch.get_rng_state().clone();r=strict_initialize(ordinary,source.state_dict(),head=stage=='H2');assert torch.equal(torch.get_rng_state(),rng)
  assert r['generated_CPU_initial_forecast_exact'] and r['normalization_preserved']
  expected=READOUT if stage=='H2' else set(dict(source.named_parameters()))
  assert {n for n,p in ordinary.named_parameters() if p.requires_grad}==expected
  if stage=='H2':
   frozen={n:p.detach().clone() for n,p in ordinary.named_parameters() if not p.requires_grad}
   opt=torch.optim.AdamW([p for p in ordinary.parameters() if p.requires_grad],lr=.0005)
   out,_=ordinary(torch.zeros(1,4,211));out.square().sum().backward();opt.step()
   assert all(torch.equal(p,dict(ordinary.named_parameters())[n]) for n,p in frozen.items())
  checks.append(dict(stage=stage,exact_interval_constructor_RNG=True,same_sampler_probe=True,exact_plain_parameters=True,strict_load_and_initial_forecasts=True,parity_CPU_RNG_preserved=True,trainable_roles_exact=True))
 print(json.dumps(dict(status='PASS_GENERATED_ORDINARY_WARM',checks=checks,adapter_sha256=sha(HERE/'train_warm.py'),test_sha256=sha(Path(__file__)),data_reads=0,forecast_fits=0)))
if __name__=='__main__':main()
