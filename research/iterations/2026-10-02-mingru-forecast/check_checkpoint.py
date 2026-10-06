"""CPU-only fixed endpoint/strict shape/frozen-backbone check. No model forward."""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
import torch
from train_minimal import load_trainer,ARCH_SHA
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--commands',type=Path,default=HERE/'commands.json');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1)
 out=a.output.resolve();assert out.is_relative_to(HERE) and not out.exists();d=json.loads(a.commands.read_text())
 job=next(j for j in d['jobs'] if j['name']==a.run and j['stage'] in ('parent','H2'));path=ROOT/job['endpoint'];ck=torch.load(path,map_location='cpu',weights_only=False);epoch=3 if job['stage']=='parent' else 1
 expected=job['resolved_args']|dict(architecture_source_sha256=ARCH_SHA,recurrent_state='positive FP32 tensor [layers,batch,hidden]')
 assert ck['args']==expected and ck['epoch']==epoch and ck['n_in']==211 and ck.get('readout') is None
 assert sha(HERE/'minimal_rnn.py')==ARCH_SHA and all(torch.isfinite(v).all() for v in ck['model'].values())
 # Construction/strict load verifies shapes; there is no dataset access or inference.
 trainer=load_trainer();model=trainer.make_model(dict(ck['args']),211,23);model.load_state_dict(ck['model'],strict=True)
 alias=path.with_name('model.pt');end=torch.load(alias,map_location='cpu',weights_only=False)
 assert end['args']==ck['args'] and end['n_in']==211 and set(end['model'])==set(ck['model']) and all(equal(v,end['model'][k]) for k,v in ck['model'].items())
 frozen=[];parent=None
 if job['stage']=='H2':
  p=ROOT/f'artifacts/runs/oct02_mingru_s3_f{job["fold"]}/model_ep3.pt';pc=torch.load(p,map_location='cpu',weights_only=False)
  assert ck['args']['init']==p.parent.name+'@ep3' and pc['epoch']==3 and pc['args']['valid_fold']==job['fold'] and pc['args']['architecture_source_sha256']==ARCH_SHA
  frozen=[k for k in ck['model'] if k.startswith(('inp.','cells.')) or k in ('mean','std')]
  assert frozen and all(equal(ck['model'][k],pc['model'][k]) for k in frozen)
  assert any(not equal(v,pc['model'][k]) for k,v in ck['model'].items() if k.startswith('head.'))
  parent=dict(path=str(p),sha256=sha(p))
 result=dict(status='PASS_FIXED_ENDPOINT_FINITE_LINEAGE',run=a.run,stage=job['stage'],fold=job['fold'],epoch=epoch,n_in=211,checkpoint=dict(path=str(path),sha256=sha(path)),model_pt_sha256=sha(alias),model_pt_state_byte_equal=True,all_resolved_args_equal=True,architecture_source_sha256=ARCH_SHA,strict_state_shapes=True,all_state_finite=True,parent=parent,byte_identical_frozen_tensors=frozen,commands_sha256=sha(a.commands),validator_sha256=sha(__file__),inference=False,metrics=0)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
