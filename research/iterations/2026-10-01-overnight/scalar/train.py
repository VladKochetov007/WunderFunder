"""Allocation-gated fixed SCALARPLE train wrapper; unchanged ordinary trainer loop.

This file is preparation only until an external root allocation binds its source,
exact base argv, approved train-only knot fit and (for H2) exact parent checkpoint.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
 alloc=json.loads(a.allocation.read_text())
 assert alloc.get('SCALARPLE_forecast_fit_authorized') is True
 for rel,expected in alloc['sources'].items():assert sha(ROOT/rel)==expected,rel
 from training_recipe import verify_job,enforce_cutoff,verify_input_snapshot
 job=alloc['runs'][a.run];verify_job(job,a.run);enforce_cutoff(job);verify_input_snapshot(alloc)
 argv=job['base_argv'];arm=job['arm'];stage=job['stage']
 assert arm in ['control','ple'] and stage in ['parent','head']
 assert argv[argv.index('--name')+1]==a.run and a.run.startswith(('on_ple_','on_plectl_'))
 assert '--profile-steps' not in argv and '--parity-dump' not in argv and '--max-batches' not in argv
 assert argv[argv.index('--epochs')+1]==('4' if stage=='parent' else '2')
 assert argv[argv.index('--hidden')+1]=='128' and argv[argv.index('--chunk')+1]=='2000'
 assert argv[argv.index('--batch')+1]==('32' if stage=='parent' else '16')
 if stage=='parent':assert '--init' not in argv and '--freeze' not in argv and '--eval-ema' in argv
 else:
  assert '--init' in argv and '--freeze' in argv and '--eval-ema' not in argv
  assert argv[argv.index('--freeze')+1]==('inp,gru,scalar' if arm=='ple' else 'inp,gru')
  parent=job['parent'];assert sha(ROOT/parent['path'])==parent['sha256']
  import torch
  ck=torch.load(ROOT/parent['path'],map_location='cpu',weights_only=False)
  assert ck['epoch']==3 and ck['args']['valid_fold']==job['fold'] and ck['args']['seed']==job['seed'] and ck['args']['scalar_arm']==arm
  assert ck['args']['scalar_stage']=='parent' and ck['args']['name']==job['parent_run']
  init=argv[argv.index('--init')+1];assert init.endswith('@ep3')
  assert str(Path('artifacts/runs')/init[:-4]/'model_ep3.pt')==parent['path']
 sys.path[:0]=[str(P),str(ROOT/'src')]
 import numpy as np
 from scalar_model import ScalarRNNModel,SCHEMA,validate_knots
 fit=None
 if arm=='ple':
  fitted=alloc['approved_scalar_fit'];fit=json.loads((ROOT/fitted['receipt_path']).read_text())
  assert sha(ROOT/fitted['receipt_path'])==fitted['receipt_sha256']
  assert fit['status']=='PASS' and fit['targets_read'] is False and fit['validation_chains_read'] is False
  if stage=='head':assert ck['args']['scalar_fit_receipt_sha256']==fitted['receipt_sha256']
  assert sha(ROOT/fit['knots_path'])==fit['knots_file_sha256']
  with np.load(ROOT/fit['knots_path'],allow_pickle=False) as z:k=z['knots'].copy();c=z['counts'].copy()
  assert k.dtype==np.float32 and c.dtype==np.int64
  validate_knots(k,c)
  assert hashlib.sha256(k.tobytes()).hexdigest()==fit['knots_array_sha256']
  assert hashlib.sha256(c.tobytes()).hexdigest()==fit['counts_array_sha256']
 spec=importlib.util.spec_from_file_location('scalar_ordinary_train',P.parent/'training/train_aux.py')
 base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 original_make=base.make_model
 def make_model(args,n_in,n_out):
  args['scalar_arm']=arm;args['scalar_stage']=stage;args['scalar_allocation_sha256']=sha(a.allocation);args['scalar_training_plan_sha256']=alloc['training_plan_sha256'];args['scalar_parent_sha256']=job.get('parent',{}).get('sha256')
  if arm=='control':return original_make(args,n_in,n_out)
  args['scalar_schema']=SCHEMA;args['scalar_fit_receipt_sha256']=fitted['receipt_sha256']
  args['scalar_knots_array_sha256']=fit['knots_array_sha256'];args['scalar_counts_array_sha256']=fit['counts_array_sha256']
  assert args['enc']=='linear' and args['arch']=='gru' and args['cell']=='gru'
  return ScalarRNNModel(n_in,args['hidden'],args['layers'],args['dropout'],n_out,knots=k,counts=c)
 base.make_model=make_model
 # New construction consumes only its own table generator. Ordinary loop/data RNG,
 # fixed epoch saving, skipped-evaluation RNG draw, loss, EMA and optimizer unchanged.
 sys.argv=[str(P/'train.py'),*argv];base.main()
if __name__=='__main__':main()
