"""Isolated frozen-trainer adapter: explicit plain warmstart, strict own-parent H2."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
MODEL_SHA='7ef61881baf1ad0befe229ea087c958f9aad6aa65240c0b3d8580dd90015d241'
FREEZE='inp,gru,interval_proj,event_gru'
READOUT={'extra_weight','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class Parser(argparse.ArgumentParser):
 def parse_args(self,*a,**kw):
  self.add_argument('--interval-arm',choices=['accumulated','instantaneous'],required=True)
  args=super().parse_args(*a,**kw)
  args.interval_source_sha256=MODEL_SHA;args.interval_trainer_source_sha256=TRAIN_SHA
  plan=json.loads((HERE/'commands.json').read_text());job=next(j for j in plan['jobs'] if j['name']==args.name)
  # --fast resolved by base.main after parsing; mirror only for argument checking.
  resolved=vars(args).copy()
  if args.fast:
   for k in ('sync_free','fused_adam','pin_memory','prop_cache'):resolved[k]=True
  if resolved!=job['resolved_args']:self.error('Only exact registered interval recipe is accepted')
  if not args.name.startswith('oct02_im_') or (ROOT/'artifacts/runs'/args.name).exists():self.error('Fresh interval run directory required')
  return args

def make_model(config,n_in,n_out):
 assert n_in==211 and n_out==23 and config['hidden']==256 and config['layers']==1 and config['dropout']==0
 assert config['interval_source_sha256']==MODEL_SHA and config['interval_trainer_source_sha256']==TRAIN_SHA
 assert sha(HERE/'interval_memory.py')==MODEL_SHA
 from interval_memory import IntervalMemoryModel
 return IntervalMemoryModel(config['interval_arm'],n_out=n_out)


def transplant_plain(model,state):
 """Generated-only CPU parity on actual warmstart weights; training RNG unchanged."""
 import torch
 from wnn.nn import RNNModel
 model.load_plain_baseline(state)
 assert all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items())
 assert torch.count_nonzero(model.extra_weight)==0
 # CPU-only constructor/dummy inputs consume no CUDA RNG; fork restores CPU RNG.
 with torch.random.fork_rng(devices=[]),torch.no_grad():
  plain=RNNModel(211,256,1,0.,n_out=23).cpu().eval();plain.load_state_dict(state,strict=True)
  probe=copy.deepcopy(model).cpu().eval()
  x=torch.linspace(-.25,.25,211*19,dtype=torch.float32).reshape(1,19,211)
  x[...,178]=0;x[:,[3,7,18],178]=1
  original,h=plain(x);got,carry=probe(x)
  assert torch.equal(original,got) and torch.equal(h,carry[0]),'Warmstart changed initial forecasts'
 return dict(generated_CPU_initial_forecast_exact=True,generated_output_sha256=state_sha({'output':got}),ordinary_tensors_exact=True,normalization_preserved=True,extra_readout_zero=True)


def load_own_parent(model,state):
 model.load_state_dict(state,strict=True);model.freeze_trunk()
 assert {n for n,p in model.named_parameters() if p.requires_grad}==READOUT
 assert all(__import__('torch').equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in state.items())
 return dict(strict_full_state_loaded=True,H2_trainable=sorted(READOUT),H2_freeze_prefixes=FREEZE)


def initialize_model(model,ck,args):
 import torch
 plan=json.loads((HERE/'commands.json').read_text());job=next(j for j in plan['jobs'] if j['name']==args.name)
 run,_,ep=args.init.partition('@ep');path=ROOT/'artifacts/runs'/run/f'model_ep{int(ep)}.pt'
 assert ck['n_in']==211 and ck['epoch']==int(ep) and ck['readout'] is None and ck['readout_tilted'] is False
 if job['stage']=='parent':
  binding=plan['plain_parents'][str(args.valid_fold)]
  assert str(path.relative_to(ROOT))==binding['checkpoint'] and sha(path)==binding['sha256']
  assert ck['args']==binding['resolved_args'] and args.freeze=='' and args.epochs==1 and ep=='3'
  parity=transplant_plain(model,ck['model']);assert all(p.requires_grad for p in model.parameters())
 else:
  assert job['stage']=='H2' and args.freeze==FREEZE and args.epochs==2 and ep=='0'
  parent=next(j for j in plan['jobs'] if j['name']==run)
  assert parent['stage']=='parent' and parent['fold']==args.valid_fold and parent['resolved_args']['interval_arm']==args.interval_arm
  assert ck['args']==parent['resolved_args']
  parity=load_own_parent(model,ck['model'])
 receipt=dict(run=args.name,stage=job['stage'],arm=args.interval_arm,initial_checkpoint=str(path.relative_to(ROOT)),initial_checkpoint_sha256=sha(path),initial_model_state_sha256=state_sha(model.state_dict()),normalization_sha256=state_sha({'mean':model.mean,'std':model.std}),parity=parity,model_source_sha256=MODEL_SHA,trainer_source_sha256=TRAIN_SHA,training_parameters=[n for n,p in model.named_parameters() if p.requires_grad],forecast_data_reads=0,metrics=0)
 with (ROOT/'artifacts/runs'/args.name/'initial-state.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('INTERVAL_INITIAL',json.dumps(receipt),flush=True)


def generated_source():
 assert sha(TRAINER)==TRAIN_SHA
 text=TRAINER.read_text();needle='        model.load_state_dict(init["model"])'
 assert text.count(needle)==1
 return text.replace(needle,'        initialize_model(model, init, args)')

def load_trainer():
 assert sha(HERE/'interval_memory.py')==MODEL_SHA
 module=types.ModuleType('interval_fixed_recipe_trainer');module.__file__=str(TRAINER)
 exec(compile(generated_source(),str(TRAINER),'exec'),module.__dict__)
 module.argparse=types.SimpleNamespace(**{**vars(argparse),'ArgumentParser':Parser})
 module.make_model=make_model;module.initialize_model=initialize_model
 return module

if __name__=='__main__':load_trainer().main()
