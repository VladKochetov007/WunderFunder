"""Original-parent RNG-control H2 endpoint/lineage; CPU tensor integrity only."""
from pathlib import Path
import argparse,hashlib,json,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SHAPES={'mean':(211,),'std':(211,),'inp.weight':(256,211),'inp.bias':(256,),
 'gru.weight_ih_l0':(768,256),'gru.weight_hh_l0':(768,256),'gru.bias_ih_l0':(768,),'gru.bias_hh_l0':(768,),
 'head.0.weight':(256,512),'head.0.bias':(256,),'head.2.weight':(23,256),'head.2.bias':(23,)}
READOUT={'head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('--commands',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert a.output.resolve().is_relative_to(HERE) and not a.output.exists() and not a.output.is_symlink()
 plan=json.loads(a.commands.read_text());job=next(j for j in plan['jobs'] if j['name']==a.job)
 assert job['stage']=='H2' and len(job['resolved_args'])==103
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
 path=ROOT/job['endpoint'];epoch=0 if job['stage']=='parent' else 1;args=job['resolved_args'];assert args['epochs']==epoch+1
 sd,record=load(path,args,epoch);alias,alias_record=load(path.parent/'model.pt',args,None)
 assert all(same(v,alias[k]) for k,v in sd.items())
 history=json.loads((path.parent/'history.json').read_text());assert len(history)==epoch+1
 assert all(h['epoch']==i and h['evaluation_skipped'] is True and h['metric_placeholder_not_a_readout'] is True for i,h in enumerate(history))
 ip=path.parent/'initial-state.json';initial=json.loads(ip.read_text())
 assert initial['run']==a.job and initial['stage']==job['stage'] and initial['adapter_source_sha256']==sha(HERE/'train_warm.py')
 initrun,_,ep=args['init'].partition('@ep');initpath=ROOT/'artifacts/runs'/initrun/f'model_ep{int(ep)}.pt'
 assert initial['initial_checkpoint']==str(initpath.relative_to(ROOT)) and initial['initial_checkpoint_sha256']==sha(initpath)
 ck=torch.load(initpath,map_location='cpu',weights_only=False);before=ck['model'];assert set(before)==set(SHAPES)
 assert all(initial['parity'][k] is True for k in ('strict_full_state_loaded','ordinary_tensors_exact','normalization_preserved','generated_CPU_initial_forecast_exact'))
 assert same(sd['mean'],before['mean']) and same(sd['std'],before['std'])
 frozen=[]
 b=plan['plain_parents'][str(job['fold'])]
 assert ep=='3' and sha(initpath)==b['sha256'] and ck['epoch']==3 and ck['args']==b['resolved_args']
 assert args['freeze']=='inp,gru' and set(initial['training_parameters'])==READOUT
 frozen=sorted(set(SHAPES)-READOUT);assert all(same(sd[k],before[k]) for k in frozen)
 wr=job['warm_initial_receipt'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256']
 warm=json.loads(wp.read_text())
 assert initial['cpu_rng_matches_warm_H2'] is True and initial['cpu_rng_state_sha256']==warm['cpu_rng_state_sha256']
 assert initial['warm_H2_initial_receipt_path']==wr['path'] and initial['warm_H2_initial_receipt_sha256']==wr['sha256']
 changed=[k for k in READOUT if not same(sd[k],before[k])];assert changed
 result=dict(status='PASS_ORDINARY_WARM_RNG_FIXED_ENDPOINT_AND_LINEAGE',job=a.job,stage=job['stage'],fold=job['fold'],checkpoint=record,model_pt=alias_record,
  model_pt_state_byte_equals_endpoint=True,all103_args_match=True,byte_identical_frozen_tensors=frozen,normalization_matches_parent=True,changed_readout_tensors=changed,cpu_rng_matches_warm_H2=True,
  initial_state_receipt_sha256=sha(ip),commands_sha256=sha(a.commands),validator_sha256=sha(Path(__file__)),data_or_targets_read=False,model_forward_calls=0,metrics=0)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
