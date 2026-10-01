"""AST-only fixed command derivation; no Torch, dataset or model access."""
import argparse,ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
REFERENCE=ROOT/'research/iterations/2026-10-01-continuation/student/commands.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse(TRAINER.read_text())):
  if not (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'):continue
  args=[ast.literal_eval(x) for x in n.args];kw={}
  for x in n.keywords:
   kw[x.arg]={'int':int,'float':float,'str':str}[x.value.id] if x.arg=='type' else ast.literal_eval(x.value)
  if '--arch' in args:kw['choices']=[*kw['choices'],'mingru']
  p.add_argument(*args,**kw)
 return p
def resolve(p,argv):
 a=vars(p.parse_args(argv))
 if a['fast']:
  for k in ('sync_free','fused_adam','pin_memory','prop_cache'):a[k]=True
 return a
def setarg(argv,flag,value):
 if flag in argv:argv[argv.index(flag)+1]=str(value)
 else:argv.extend([flag,str(value)])
def main():
 assert sha(TRAINER)=='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
 ref=json.loads(REFERENCE.read_text());p=parser();jobs=[]
 for fold in (0,1):
  parent=f'oct02_mingru_s3_f{fold}';head=f'oct02_mingru_s3h2_f{fold}'
  for stage,name in (('parent',parent),('H2',head)):
   old=next(j for j in ref['jobs'] if j['stage']==stage and j['fold']==fold);argv=old['argv'].copy();argv[1]=str((HERE/'train_minimal.py').relative_to(ROOT))
   for flag,value in [('--name',name),('--arch','mingru'),('--hidden',256),('--layers',2),('--batch',32 if stage=='parent' else 16),('--chunk',2000),('--epochs',4 if stage=='parent' else 2)]:setarg(argv,flag,value)
   if stage=='H2':setarg(argv,'--init',parent+'@ep3');setarg(argv,'--freeze','inp,cells')
   args=resolve(p,argv[2:]);diff={k:[v,args[k]] for k,v in old['resolved_args'].items() if v!=args[k]}
   assert set(diff)==({'name','arch','layers'} if stage=='parent' else {'name','arch','layers','init','freeze'})
   endpoint=f'artifacts/runs/{name}/model_ep{3 if stage=="parent" else 1}.pt'
   jobs.append(dict(name=name,stage=stage,fold=fold,argv=argv,resolved_args=args,reference_run=old['name'],argument_changes=diff,endpoint=endpoint,require_absent=[f'artifacts/runs/{name}'],required_outputs=[endpoint,f'artifacts/runs/{name}/model.pt',f'artifacts/runs/{name}/history.json'],suggested_timeout_seconds=6000 if stage=='parent' else 1800))
  dest=str((HERE/'cache'/f'pred_{head}_valid.npy').relative_to(ROOT))
  jobs.append(dict(name=head+'_OOF',stage='OOF',fold=fold,run=head,argv=['.venv/bin/python',str((HERE/'predict_minimal.py').relative_to(ROOT)),head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',str((HERE/'cache').relative_to(ROOT))],endpoint=dest,require_absent=[dest,dest.replace('.npy','.npy.partial'),dest.replace('.npy','.inputs.json')],required_outputs=[dest,dest.replace('.npy','.inputs.json')],suggested_timeout_seconds=900))
 sources=[TRAINER,REFERENCE,*[HERE/n for n in ('minimal_rnn.py','train_minimal.py','predict_minimal.py','check_checkpoint.py','build_plan.py')],*[ROOT/'src/wnn'/f'{n}.py' for n in ('nn','features','data','outcomes','resources','splits','propensity','metric','gmetric','event_attn')]]
 result=dict(status='SOURCE_ONLY_NOT_ALLOCATED',execution_authorized=False,reference_commands=str(REFERENCE.relative_to(ROOT)),reference_sha256=sha(REFERENCE),architecture_sha256=sha(HERE/'minimal_rnn.py'),source_sha256={str(s.relative_to(ROOT)):sha(s) for s in sources},counts=dict(fits=4,OOF=2,WP=0),jobs=jobs,environment=ref['environment'],requirements=['Root allocation and input identity inventory before execution','Native runtime gate pending','Checkpoint validation after each fit and before dependent job','Parent EMA fixed ep3; H2 fixed ep1; no model/epoch selection'],comparison='Replace current shallow256 member with minGRU256x2 at fixed quarter weight; other3 anchors unchanged',RNG_caveat='Architecture changes initialization draws and thus may alter normalization sample/shuffle relative to old GRU; this is recipe utility, not parameter-matched causal isolation.')
 (HERE/'commands.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS_SOURCE_ARGUMENT_DERIVATION_ONLY','jobs':len(jobs),'commands_sha256':sha(HERE/'commands.json')}))
if __name__=='__main__':main()
