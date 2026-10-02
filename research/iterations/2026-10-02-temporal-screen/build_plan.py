"""AST-only paired fixed-clock command derivation; no Torch or data access."""
import argparse, ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
REFERENCE=ROOT/'research/iterations/2026-10-01-continuation/student/commands.json'
TEMPLATE=ROOT/'research/iterations/2026-10-02-distill07/commands.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse(TRAINER.read_text())):
  if not (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'):continue
  args=[ast.literal_eval(x) for x in n.args];kw={}
  for x in n.keywords:kw[x.arg]={'int':int,'float':float,'str':str}[x.value.id] if x.arg=='type' else ast.literal_eval(x.value)
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
 ref=json.loads(REFERENCE.read_text());p=parser();jobs=[];proof=[]
 for fold in (0,1):
  # Fixed AB/BA order only balances execution order; no outcome-selected branch.
  for arm in (('control','candidate') if fold==0 else ('candidate','control')):
   parent=f'oct02_temporal_{arm}_s3_f{fold}';head=f'oct02_temporal_{arm}_s3h2_f{fold}'
   horizon=500 if arm=='candidate' else 2000
   for stage,name in (('parent',parent),('H2',head)):
    old=next(j for j in ref['jobs'] if j['stage']==stage and j['fold']==fold)
    argv=old['argv'].copy();setarg(argv,'--name',name)
    if stage=='parent':argv[1]=str((HERE/'train_temporal.py').relative_to(ROOT))
    else:setarg(argv,'--init',parent+'@ep3')
    args=resolve(p,argv[2:]);diff={k:[v,args[k]] for k,v in old['resolved_args'].items() if v!=args[k]}
    assert set(diff)==({'name'} if stage=='parent' else {'name','init'})
    assert len(args)==103 and args['arch']=='gru' and args['hidden']==256 and args['layers']==1
    assert args['chunk']==2000 and args['batch']==(32 if stage=='parent' else 16)
    if stage=='parent':
     argv.extend(['--gradient-horizon',str(horizon)])
     args.update(gradient_horizon=horizon,temporal_source_sha256=sha(HERE/'temporal.py'))
    epoch=3 if stage=='parent' else 1;run=f'artifacts/runs/{name}';endpoint=f'{run}/model_ep{epoch}.pt'
    job=dict(name=name,stage=stage,arm=arm,fold=fold,argv=argv,resolved_args=args,expected_n_in=211,endpoint=endpoint,
      reference_job=old['name'],argument_changes=diff,dependency=None if stage=='parent' else parent+'@ep3',
      require_absent=[run],required_outputs=[*[f'{run}/model_ep{e}.pt' for e in range(epoch+1)],f'{run}/model.pt',f'{run}/history.json'],wall_limit_seconds=3600 if stage=='parent' else 1200)
    jobs.append(job);proof.append(dict(job=name,common_argument_count=103,reference_differences=diff,
       added_training_metadata={k:args[k] for k in ('gradient_horizon','temporal_source_sha256')} if stage=='parent' else {},
       architecture_unchanged=True,ordinary_H2=stage=='H2'))
   dest=str((HERE/'cache'/f'pred_{head}_valid.npy').relative_to(ROOT))
   jobs.append(dict(name=head+'_OOF',stage='OOF',arm=arm,fold=fold,run=head,
    argv=['.venv/bin/python',str((HERE/'predict_fixed.py').relative_to(ROOT)),head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',str((HERE/'cache').relative_to(ROOT))],
    endpoint=dest,require_absent=[dest,dest.replace('.npy','.npy.partial'),dest.replace('.npy','.inputs.json')],
    required_outputs=[dest,dest.replace('.npy','.inputs.json')],dependency=head,wall_limit_seconds=900))
 for fold in (0,1):
  for stage in ('parent','H2'):
   pair=[j for j in jobs if j['fold']==fold and j['stage']==stage]
   a,b=[j['resolved_args'] for j in pair];delta={k for k in a if a[k]!=b[k]}
   assert delta==({'name','gradient_horizon'} if stage=='parent' else {'name','init'})
 sources={k:v for k,v in json.loads(TEMPLATE.read_text())['source_sha256'].items() if '/2026-10-02-distill07/' not in k}
 extra=[REFERENCE,TEMPLATE,ROOT/'research/iterations/2026-10-01-context-replay/run.py',*[HERE/n for n in ('temporal.py','train_temporal.py','predict_fixed.py','check_checkpoint.py','build_plan.py','run_queue.py','check_generated.py')]]
 sources.update({str(s.relative_to(ROOT)):sha(s) for s in extra})
 result=dict(status='SOURCE_ONLY_FIXED_TEMPORAL_PAIR',execution_authorized=False,reference_commands=str(REFERENCE.relative_to(ROOT)),reference_sha256=sha(REFERENCE),source_sha256=sources,
  scientific_change='Candidate: detach recurrent graph every500 rows within fixed2000 update. Control: connected graph across same four500 forwards. Identical full2000 objectives, optimizer/scheduler/EMA clocks; no carry reset.',
  counts=dict(fits=8,OOF=4,WP=0),jobs=jobs,environment=ref['environment'],
  execution_order='Fold0 control then candidate; fold1 candidate then control; each parent,H2,OOF. One root-allocated GPU lane.',
  primary='Fixed candidate4 vs current_student4; candidate4 vs fresh segmented control4 distinguishes gradient-horizon effect; no blend search.',
  serving='Ordinary unsegmented GRU256x1 with exact original parameter keys; no temporal adapter at H2 or prediction.',
  requirements=['Root allocation/input identities before execution','Parent EMA fixedep3; ordinary H2 fixedep1; no epoch selection','Root supplies 2.5h absolute deadline and resource inventory; no automatic retry'])
 (HERE/'commands.json').write_text(json.dumps(result,indent=2)+'\n')
 report=dict(status='PASS_AST_ARGUMENT_PROOF_NO_DATA',commands_sha256=sha(HERE/'commands.json'),fits=8,OOF=4,proof=proof,
  paired_parent_differences=['name','gradient_horizon'],paired_H2_differences=['name','init'],
  hidden_architecture_changes=False,ordinary_argument_count=103,parent_saved_argument_count=105,models_loaded=0,arrays_read=0)
 (HERE/'argument-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','commands_sha256','fits','OOF')}))
if __name__=='__main__':main()
