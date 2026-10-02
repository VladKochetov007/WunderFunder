"""Exact103-arg ordinary counterpart to the already fixed interval warm recipe."""
from pathlib import Path
import argparse,ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
I=ROOT/'research/iterations/2026-10-02-interval-memory'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
I_SHA='e4deda87a9520aad4ec0c3a95886bfddacbaa4e11a4bdc36ed838f2d0de3c663'
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
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def main():
 assert sha(TRAINER)==TRAIN_SHA and sha(I/'commands.json')==I_SHA
 original=json.loads((I/'commands.json').read_text());p=parser();defaults=resolve(p,['--name','none']);rel=str(HERE.relative_to(ROOT));jobs=[];proof=[]
 extras={'interval_arm','interval_source_sha256','interval_trainer_source_sha256'}
 for fold in (0,1):
  parent=f'oct02_ow_s3_f{fold}';head=f'oct02_ow_s3h2_f{fold}'
  for stage,name,ep in [('parent',parent,0),('H2',head,1)]:
   old=next(j for j in original['jobs'] if j['stage']==stage and j['fold']==fold and j['arm']=='instantaneous')
   a={k:v for k,v in old['resolved_args'].items() if k not in extras};a['name']=name
   if stage=='H2':a.update(init=parent+'@ep0',freeze='inp,gru')
   argv=[]
   for k,v in a.items():
    if k!='name' and v==defaults[k]:continue
    flag='--'+k.replace('_','-')
    if isinstance(v,bool):assert v;argv.append(flag)
    else:argv.extend([flag,str(v)])
   assert len(a)==103 and resolve(p,argv)==a
   delta={k:dict(interval=old['resolved_args'][k],ordinary=a[k]) for k in a if a[k]!=old['resolved_args'][k]}
   assert set(delta)==({'name'} if stage=='parent' else {'name','init','freeze'})
   assert a['epochs']==ep+1 and a['lr']==.0005
   if stage=='parent':assert a['init']==f'cont_sh256_s3_f{fold}@ep3' and not a['freeze']
   jobs.append(dict(name=name,stage=stage,fold=fold,argv=['.venv/bin/python',rel+'/train_warm.py',*argv],resolved_args=a,
    expected_n_in=211,endpoint=f'artifacts/runs/{name}/model_ep{ep}.pt',dependency=a['init'],wall_limit_seconds=1200 if stage=='parent' else 900,
    interval_reference=old['name'],argument_changes=delta))
   proof.append(dict(name=name,resolved_arguments=103,exact=True,removed_metadata=sorted(extras),changes=delta))
  out=f'{rel}/cache/pred_{head}_valid.npy'
  jobs.append(dict(name=head+'_OOF',stage='OOF',fold=fold,argv=['.venv/bin/python',rel+'/predict_fixed.py',head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],dependency=head,wall_limit_seconds=600,
   require_absent=[out,out.replace('.npy','.npy.partial'),out.replace('.npy','.inputs.json')],required_outputs=[out,out.replace('.npy','.inputs.json')]))
 source_paths=[TRAINER,I/'commands.json',I/'interval_memory.py',HERE/'train_warm.py',HERE/'predict_fixed.py',HERE/'check_checkpoint.py',HERE/'check_generated.py',HERE/'protocol.md',Path(__file__),*sorted((ROOT/'src/wnn').glob('*.py'))]
 plan=dict(status='SOURCE_ONLY_ORDINARY_WARM_CONTROL',execution_authorized=False,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_paths},
  counts=dict(fits=4,OOF=2),jobs=jobs,plain_parents=original['plain_parents'],environment=original['environment'],estimated_minutes=[12,22],allocation=None,
  rng_pairing='Exact discarded interval extra-module initialization draws before both parent/H2 data-loader iterations; ordinary model state remains12keys.',
  scientific_contrast='Ordinary one-epoch warmstart versus current student4; I instantaneous versus ordinary warm separates added architecture from same extra epoch, initialization and data order.')
 write(HERE/'commands.json',plan)
 write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ONLY_103_ARGUMENTS',commands_sha256=sha(HERE/'commands.json'),interval_commands_sha256=I_SHA,jobs=proof,forecast_fits=0,data_reads=0,quality_reads=0))
 print(json.dumps(dict(status=plan['status'],commands_sha256=sha(HERE/'commands.json'),fits=4,OOF=2)))
if __name__=='__main__':main()
