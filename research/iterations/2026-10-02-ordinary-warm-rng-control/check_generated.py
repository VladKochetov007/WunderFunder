"""Generated-only W/control factory RNG, strict load and frozen-role checks."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import torch
from train_warm import make_model,strict_initialize,READOUT,sha
from wnn.nn import RNNModel
W=ROOT/'research/iterations/2026-10-02-ordinary-warm/train_warm.py'
def main():
 torch.set_num_threads(1)
 assert sha(W)=='f351139fa40b4a65cc834ba7373d82c32ce653e1d98436953929f711155a321e'
 sp=importlib.util.spec_from_file_location('warm_reference_factory',W);warm=importlib.util.module_from_spec(sp);sp.loader.exec_module(warm)
 def function_ast(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
 for name in ('make_model','strict_initialize'):assert function_ast(HERE/'train_warm.py',name)==function_ast(W,name)
 config={'hidden':256,'layers':1,'dropout':0}
 torch.manual_seed(19);source=RNNModel(211,256,1,0.,n_out=23)
 source.mean.copy_(torch.linspace(-.2,.2,211));source.std.copy_(torch.linspace(.7,1.3,211))
 def setup(factory,initialize):
  torch.manual_seed(3);model=factory(config,211,23);before=torch.get_rng_state().clone()
  parity=initialize(model,source.state_dict(),head=True);assert torch.equal(before,torch.get_rng_state())
  assert parity['generated_CPU_initial_forecast_exact'] and parity['normalization_preserved']
  loader=torch.utils.data.DataLoader(torch.arange(257),batch_size=16,shuffle=True,num_workers=0,drop_last=True)
  orders=[]
  for _ in range(2):
   orders.append(torch.cat(list(loader)));torch.empty((),dtype=torch.int64).random_()
  return model,before,orders,torch.get_rng_state().clone()
 reference,wrng,worder,wend=setup(warm.make_model,warm.strict_initialize)
 control,crng,corder,cend=setup(make_model,strict_initialize)
 assert torch.equal(wrng,crng) and torch.equal(wend,cend)
 assert all(torch.equal(a,b) for a,b in zip(worder,corder))
 assert all(torch.equal(v,control.state_dict()[k]) for k,v in reference.state_dict().items())
 assert {n for n,p in control.named_parameters() if p.requires_grad}==READOUT
 before={k:v.clone() for k,v in control.state_dict().items()}
 optimizer=torch.optim.AdamW([p for p in control.parameters() if p.requires_grad],lr=.0005)
 out,_=control(torch.linspace(-.2,.2,4*211).reshape(1,4,211));out.square().mean().backward();optimizer.step()
 frozen=set(before)-READOUT
 assert all(torch.equal(v,control.state_dict()[k]) for k,v in before.items() if k in frozen)
 assert any(not torch.equal(before[k],control.state_dict()[k]) for k in READOUT)
 print(json.dumps(dict(status='PASS_GENERATED_ORDINARY_WARM_RNG',factory_and_initializer_AST_identical=True,
  strict_load_initial_forecast_and_normalization_exact=True,initializer_preserves_CPU_RNG=True,
  full_CPU_RNG_matches_W=True,two_epoch_generated_sampler_order_matches_W=True,
  readout_roles_exact=True,frozen_trunk_and_norm_unchanged_after_optimizer=True,readout_changed=True,
  adapter_sha256=sha(HERE/'train_warm.py'),test_sha256=sha(Path(__file__)),data_reads=0,forecast_fits=0,metrics=0)))
if __name__=='__main__':main()
