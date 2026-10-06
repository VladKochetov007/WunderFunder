"""Frozen ordinary trainer, one extra parent epoch; sham draws pair I's shuffle."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
READOUT={'head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class Parser(argparse.ArgumentParser):
 def parse_args(self,*a,**kw):
  args=super().parse_args(*a,**kw);resolved=vars(args).copy()
  if args.fast:
   for k in ('sync_free','fused_adam','pin_memory','prop_cache'):resolved[k]=True
  job=next(j for j in json.loads((HERE/'commands.json').read_text())['jobs'] if j['name']==args.name)
  if resolved!=job['resolved_args']:self.error('Exact registered103-argument warm recipe required')
  path=ROOT/'artifacts/runs'/args.name
  if not args.name.startswith('oct02_ow_') or path.exists() or path.is_symlink():self.error('Fresh ordinary warm run required')
  return args

def make_model(config,n_in,n_out):
 import torch
 from wnn.nn import RNNModel
 assert n_in==211 and n_out==23 and config['hidden']==256 and config['layers']==1 and config['dropout']==0
 model=RNNModel(211,256,1,0.,n_out=23)
 # Exact I constructor order after the ordinary superclass; these modules are
 # discarded immediately and never enter optimizer/state/forward/serving.
 torch.nn.Linear(256,32)
 torch.nn.GRU(33,64,batch_first=True)
 return model

def strict_initialize(model,state,head=False):
 import torch
 from wnn.nn import RNNModel
 model.load_state_dict(state,strict=True)
 assert all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items())
 if head:
  for n,p in model.named_parameters():p.requires_grad_(not n.startswith(('inp.','gru.')))
  assert {n for n,p in model.named_parameters() if p.requires_grad}==READOUT
 with torch.random.fork_rng(devices=[]),torch.no_grad():
  plain=RNNModel(211,256,1,0.,n_out=23).cpu().eval();plain.load_state_dict(state,strict=True)
  probe=copy.deepcopy(model).cpu().eval();x=torch.linspace(-.25,.25,211*19,dtype=torch.float32).reshape(1,19,211)
  x[...,178]=0;x[:,[3,7,18],178]=1
  expected,h=plain(x);got,carry=probe(x)
  assert torch.equal(expected,got) and torch.equal(h,carry)
 return dict(strict_full_state_loaded=True,ordinary_tensors_exact=True,normalization_preserved=True,
             generated_CPU_initial_forecast_exact=True,generated_output_sha256=state_sha({'output':got}))

def initialize_model(model,ck,args):
 import torch
 plan=json.loads((HERE/'commands.json').read_text());job=next(j for j in plan['jobs'] if j['name']==args.name)
 run,_,ep=args.init.partition('@ep');path=ROOT/'artifacts/runs'/run/f'model_ep{int(ep)}.pt'
 assert ck['n_in']==211 and ck['epoch']==int(ep) and ck['readout'] is None and ck['readout_tilted'] is False
 if job['stage']=='parent':
  binding=plan['plain_parents'][str(args.valid_fold)]
  assert str(path.relative_to(ROOT))==binding['checkpoint'] and sha(path)==binding['sha256']
  assert ck['args']==binding['resolved_args'] and args.freeze=='' and args.epochs==1 and ep=='3'
 else:
  parent=next(j for j in plan['jobs'] if j['name']==run)
  assert job['stage']=='H2' and args.freeze=='inp,gru' and args.epochs==2 and ep=='0'
  assert parent['stage']=='parent' and parent['fold']==args.valid_fold and ck['args']==parent['resolved_args']
 parity=strict_initialize(model,ck['model'],head=job['stage']=='H2')
 if job['stage']=='parent':assert all(p.requires_grad for p in model.parameters())
 receipt=dict(run=args.name,stage=job['stage'],initial_checkpoint=str(path.relative_to(ROOT)),initial_checkpoint_sha256=sha(path),
  initial_model_state_sha256=state_sha(model.state_dict()),normalization_sha256=state_sha({'mean':model.mean,'std':model.std}),
  parity=parity,trainer_source_sha256=TRAIN_SHA,adapter_source_sha256=sha(Path(__file__)),
  training_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
  rng_pairing='Discarded Linear256to32 then GRU33to64 CPU constructors match interval initialization RNG; no parameters retained',
  cpu_rng_state_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),forecast_data_reads=0,metrics=0)
 with (ROOT/'artifacts/runs'/args.name/'initial-state.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('ORDINARY_WARM_INITIAL',json.dumps(receipt),flush=True)

def load_trainer():
 assert sha(TRAINER)==TRAIN_SHA
 source=TRAINER.read_text();needle='        model.load_state_dict(init["model"])';assert source.count(needle)==1
 source=source.replace(needle,'        initialize_model(model, init, args)')
 module=types.ModuleType('ordinary_warm_frozen_trainer');module.__file__=str(TRAINER)
 exec(compile(source,str(TRAINER),'exec'),module.__dict__)
 module.argparse=types.SimpleNamespace(**{**vars(argparse),'ArgumentParser':Parser})
 module.make_model=make_model;module.initialize_model=initialize_model
 return module
if __name__=='__main__':load_trainer().main()
