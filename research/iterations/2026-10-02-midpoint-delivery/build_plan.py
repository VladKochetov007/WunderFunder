"""Resolve two conditional final fits from frozen small metadata; no Torch/data."""
from pathlib import Path
import argparse,ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
F=ROOT/'research/iterations/2026-10-01-continuation/student/final'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse(TRAINER.read_text())):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   kw={k.arg:({'int':int,'float':float,'str':str}[k.value.id] if k.arg=='type' else ast.literal_eval(k.value)) for k in n.keywords}
   p.add_argument(*[ast.literal_eval(x) for x in n.args],**kw)
 return p
def resolve(p,argv):
 a=vars(p.parse_args(argv))
 if a['fast']:
  for k in ('sync_free','fused_adam','pin_memory','prop_cache'):a[k]=True
 return a
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 assert sha(TRAINER)=='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
 assert (HERE/'train_warm.py').read_bytes()==(W/'train_warm.py').read_bytes()
 final=json.loads((F/'commands.json').read_text());warm=json.loads((W/'commands.json').read_text());inspection=json.loads((F/'fit-inspection.json').read_text())
 p=parser();defaults=resolve(p,['--name','none']);rel=str(HERE.relative_to(ROOT));jobs=[];proof=[]
 names=['oct02_ow_final_s3','oct02_ow_final_s3h2']
 for i,name in enumerate(names):
  old=final['jobs'][i];a=old['resolved_args'].copy();a['name']=name
  if i==0:a.update(epochs=1,lr=.0005,init='cont_sh256_final_s3@ep3')
  else:a['init']=names[0]+'@ep0'
  argv=[]
  for k,v in a.items():
   if k!='name' and v==defaults[k]:continue
   flag='--'+k.replace('_','-')
   if isinstance(v,bool):assert v;argv.append(flag)
   else:argv.extend([flag,str(v)])
  assert len(a)==103 and resolve(p,argv)==a
  changes={k:[old['resolved_args'][k],v] for k,v in a.items() if old['resolved_args'][k]!=v}
  assert set(changes)==({'name','epochs','lr','init'} if i==0 else {'name','init'})
  w=next(j for j in warm['jobs'] if j['stage']==old['stage'] and j['fold']==0)['resolved_args']
  wdelta={k:[w[k],v] for k,v in a.items() if w[k]!=v}
  assert set(wdelta)==({'name','init','distill','valid_fold','include_heldout_valid','eval_source'} if i==0 else {'name','init','valid_fold','include_heldout_valid','eval_source'})
  assert a['valid_fold']==-1 and a['include_heldout_valid'] and a['eval_source']=='holdout' and a['seed']==3
  epoch=0 if i==0 else 1;out=f'artifacts/runs/{name}';check=f'{rel}/execution-01/{name}.check.json'
  jobs.append(dict(name=name,stage=old['stage'],fold=-1,argv=['.venv/bin/python',rel+'/train_warm.py',*argv],resolved_args=a,
   endpoint=out+f'/model_ep{epoch}.pt',dependency=a['init'],timeout_seconds=1800 if i==0 else 900,
   post_argv=['.venv/bin/python',rel+'/check_checkpoint.py',name,'--commands',rel+'/commands.json','--output',check],
   require_absent=[out,check],required_outputs=[out+f'/model_ep{epoch}.pt',out+'/model.pt',out+'/history.json',out+'/initial-state.json',check]))
  proof.append(dict(name=name,argument_count=103,from_original_final=changes,from_W_fold0=wdelta))
 parent=inspection['checkpoints']['parent'];assert parent['path']=='artifacts/runs/cont_sh256_final_s3/model_ep3.pt' and parent['epoch']==3
 assert parent['args']==final['jobs'][0]['resolved_args']
 source_paths=[TRAINER,HERE/'train_warm.py',HERE/'check_checkpoint.py',HERE/'run_queue.py',Path(__file__),
  W/'train_warm.py',W/'commands.json',F/'commands.json',F/'fit-inspection.json',F/'inputs.json',
  ROOT/'research/iterations/2026-10-01-context-replay/run.py',*sorted((ROOT/'src/wnn').glob('*.py'))]
 plan=dict(status='CONDITIONAL_SOURCE_ONLY_TWO_FINAL_FITS',execution_authorized=False,counts=dict(fits=2,OOF=0,metrics=0),
  jobs=jobs,plain_parents={'-1':dict(checkpoint=parent['path'],sha256=parent['sha256'],resolved_args=parent['args'])},
  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},environment=warm['environment'],
  success_status='COMPLETED_2_ORDINARY_WARM_FINAL_FITS_NO_METRICS',execution_directory=rel+'/execution-01',
  teacher_provenance='Use same accepted historical fin256_s3/s4 train/valid cache bytes as original final parent. Producer epochs unknown; do not assert ep3 cache lineage. Fresh identity binding is root allocation prerequisite.',
  required_root_decision='Fixed midpoint seed3/seed4 screen gate passed; finalseed3 fixed. No allocation created by this builder.')
 write(HERE/'commands.json',plan);write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ONLY_2X103_ARGS',commands_sha256=sha(HERE/'commands.json'),jobs=proof,adapter_byte_identical_W=True,data_reads=0,model_imports=0))
 print(json.dumps(dict(status=plan['status'],commands_sha256=sha(HERE/'commands.json'),fits=2,OOF=0)))
if __name__=='__main__':main()
