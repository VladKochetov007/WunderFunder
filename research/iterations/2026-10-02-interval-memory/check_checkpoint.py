"""CPU metadata/state validation only; no dataset reads or inference."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FREEZE='inp,gru,interval_proj,event_gru'
SHAPES={'mean':(211,),'std':(211,),'inp.weight':(256,211),'inp.bias':(256,),
 'gru.weight_ih_l0':(768,256),'gru.weight_hh_l0':(768,256),'gru.bias_ih_l0':(768,),'gru.bias_hh_l0':(768,),
 'head.0.weight':(256,512),'head.0.bias':(256,),'head.2.weight':(23,256),'head.2.bias':(23,),
 'interval_proj.weight':(32,256),'interval_proj.bias':(32,),
 'event_gru.weight_ih_l0':(192,33),'event_gru.weight_hh_l0':(192,64),'event_gru.bias_ih_l0':(192,),'event_gru.bias_hh_l0':(192,),
 'extra_weight':(256,96)}
READOUT={'extra_weight','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('--commands',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert a.output.resolve().is_relative_to(HERE) and not a.output.exists() and not a.output.is_symlink()
 plan=json.loads(a.commands.read_text());job=next(j for j in plan['jobs'] if j['name']==a.job)
 assert job['stage'] in ('parent','H2') and len(job['resolved_args'])==106
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 torch.set_num_threads(1)
 def same(x,y):return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
 def load(path,expected,ep):
  before=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False)
  assert ck['args']==expected and ck['n_in']==211 and ck['readout'] is None and ck['readout_tilted'] is False
  if ep is not None:assert ck['epoch']==ep
  sd=ck['model'];assert set(sd)==set(SHAPES)
  for k,shape in SHAPES.items():assert tuple(sd[k].shape)==shape and sd[k].dtype==torch.float32 and torch.isfinite(sd[k]).all(),k
  assert (sd['std']>0).all() and sha(path)==before
  return sd,dict(path=str(path.relative_to(ROOT)),sha256=before,epoch=ep)
 path=ROOT/job['endpoint'];epoch=0 if job['stage']=='parent' else 1;args=job['resolved_args']
 assert args['epochs']==epoch+1
 sd,record=load(path,args,epoch);alias,alias_record=load(path.parent/'model.pt',args,None)
 assert all(same(v,alias[k]) for k,v in sd.items())
 history=json.loads((path.parent/'history.json').read_text());assert len(history)==epoch+1
 assert all(h['epoch']==i and h['evaluation_skipped'] is True and h['metric_placeholder_not_a_readout'] is True for i,h in enumerate(history))
 initial_path=path.parent/'initial-state.json';initial=json.loads(initial_path.read_text())
 assert initial['arm']==job['arm'] and initial['run']==a.job and initial['stage']==job['stage']
 initrun,_,initep=args['init'].partition('@ep');initpath=ROOT/'artifacts/runs'/initrun/f'model_ep{int(initep)}.pt'
 assert initial['initial_checkpoint']==str(initpath.relative_to(ROOT)) and initial['initial_checkpoint_sha256']==sha(initpath)
 ck=torch.load(initpath,map_location='cpu',weights_only=False);before=ck['model'];frozen=[]
 if job['stage']=='parent':
  binding=plan['plain_parents'][str(job['fold'])]
  assert sha(initpath)==binding['sha256'] and ck['epoch']==3 and ck['args']==binding['resolved_args']
  assert args['freeze']=='' and args['eval_ema'] and args['ema_teacher']==1 and args['distill_alpha']==.5
  assert args['distill']==f'h256_f{job["fold"]},h256s4_f{job["fold"]}' and args['lr']==.0005
  assert initial['parity']['generated_CPU_initial_forecast_exact'] and initial['parity']['ordinary_tensors_exact'] and initial['parity']['normalization_preserved'] and initial['parity']['extra_readout_zero']
  assert set(initial['training_parameters'])==set(SHAPES)-{'mean','std'}
  assert same(sd['mean'],before['mean']) and same(sd['std'],before['std'])
 else:
  parent=next(j for j in plan['jobs'] if j['name']==initrun)
  assert parent['stage']=='parent' and parent['arm']==job['arm'] and parent['fold']==job['fold'] and initep=='0'
  assert ck['args']==parent['resolved_args'] and ck['epoch']==0 and args['freeze']==FREEZE
  assert set(initial['training_parameters'])==READOUT and initial['parity']['strict_full_state_loaded']
  frozen=sorted(set(SHAPES)-READOUT)
  assert all(same(sd[k],before[k]) for k in frozen)
 changed=[k for k in READOUT if k in before and not same(sd[k],before[k])]
 assert changed,'Readout tensors did not adapt'
 result=dict(status='PASS_INTERVAL_FIXED_ENDPOINT_AND_LINEAGE',job=a.job,stage=job['stage'],arm=job['arm'],fold=job['fold'],checkpoint=record,model_pt=alias_record,model_pt_state_byte_equals_endpoint=True,all106_args_match=True,byte_identical_frozen_tensors=frozen,normalization_matches_parent=True,changed_readout_tensors=changed,initial_state_receipt_sha256=sha(initial_path),commands_sha256=sha(a.commands),validator_sha256=sha(Path(__file__)),data_or_targets_read=False,model_forward_calls=0,metrics=0)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
