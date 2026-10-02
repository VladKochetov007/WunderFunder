"""Build a conditional seed4 confirmation from actual archived args; source/JSON only."""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REF=ROOT/'research/iterations/2026-10-01-continuation/student-seed4'
TEMPLATE=ROOT/'research/iterations/2026-10-02-distill07'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAINER_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse(TRAINER.read_text())):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   args=[ast.literal_eval(x) for x in n.args];kw={}
   for k in n.keywords:kw[k.arg]={'int':int,'float':float,'str':str}[k.value.id] if k.arg=='type' else ast.literal_eval(k.value)
   p.add_argument(*args,**kw)
 return p

def resolve(p,argv):
 a=vars(p.parse_args(argv))
 if a['fast']:
  for k in ('sync_free','fused_adam','pin_memory','prop_cache'):a[k]=True
 return a

def main():
 assert sha(TRAINER)==TRAINER_SHA
 reference=json.loads((REF/'commands.json').read_text());template=json.loads((TEMPLATE/'commands.json').read_text())
 inspection=json.loads((REF/'fit-inspection-recovery.json').read_text());bindings=json.loads((REF/'bindings-v2-01.json').read_text())
 assert inspection['all_model_states_finite'] and inspection['all103args_match'] and inspection['epoch_history_metrics_skipped']
 assert inspection['commands_sha256']==sha(REF/'commands.json')
 p=parser();jobs=[];proof=[];controls={};rel=str(HERE.relative_to(ROOT))
 for fold in (0,1):
  parent=f'oct02_d07_sh256_s4_f{fold}';head=f'oct02_d07_sh256_s4h2_f{fold}'
  for stage,oldname,name,epoch in [('parent',f'cont_sh256_s4_f{fold}',parent,3),('H2',f'cont_sh256_s4h2_f{fold}',head,1)]:
   old=next(j for j in reference['jobs'] if j['name']==oldname);a=dict(old['resolved_args']);argv=list(old['argv'])
   assert argv[:2]==['.venv/bin/python',str(TRAINER.relative_to(ROOT))]
   assert resolve(p,argv[2:])==a and a['seed']==4 and len(a)==103
   a['name']=name;argv[argv.index('--name')+1]=name
   if stage=='parent':
    assert a['distill_alpha']==.5 and a['distill']==f'h256_f{fold},h256s4_f{fold}' and not a['init']
    a['distill_alpha']=.7
    if '--distill-alpha' in argv:argv[argv.index('--distill-alpha')+1]='0.7'
    else:argv += ['--distill-alpha','0.7']
   else:
    assert a['init']==f'cont_sh256_s4_f{fold}@ep3' and not a['distill'] and a['distill_alpha']==.5
    a['init']=parent+'@ep3';argv[argv.index('--init')+1]=a['init']
   assert resolve(p,argv[2:])==a and a['epochs']==epoch+1 and a['valid_fold']==fold
   diff={k:dict(old=v,new=a[k]) for k,v in old['resolved_args'].items() if a[k]!=v}
   assert set(diff)==({'name','distill_alpha'} if stage=='parent' else {'name','init'})
   t=next(j for j in template['jobs'] if j['stage']==stage and j['fold']==fold)
   template_diff={k:dict(seed3=v,seed4=a[k]) for k,v in t['resolved_args'].items() if a[k]!=v}
   assert set(template_diff)==({'name','seed'} if stage=='parent' else {'name','seed','init'})
   j=dict(name=name,stage=stage,fold=fold,argv=argv,resolved_args=a,expected_n_in=211,
          endpoint=f'artifacts/runs/{name}/model_ep{epoch}.pt',reference_job=oldname,
          argument_changes=diff,dependency=parent+'@ep3' if stage=='H2' else None,
          wall_limit_seconds=3600 if stage=='parent' else 1200)
   jobs.append(j);proof.append(dict(name=name,arguments=103,resolved_exact=True,changes=diff,seed3_template_changes=template_diff))
  oo=f'{rel}/cache/pred_{head}_valid.npy'
  jobs.append(dict(name=head+'_OOF',stage='OOF',fold=fold,argv=['.venv/bin/python',rel+'/predict_fixed.py',head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],require_absent=[oo,oo.replace('.npy','.npy.partial'),oo.replace('.npy','.inputs.json')],required_outputs=[oo,oo.replace('.npy','.inputs.json')],dependency=head,wall_limit_seconds=900))
  oldhead=f'cont_sh256_s4h2_f{fold}';oi=inspection['OOF'][oldhead];ob=bindings['predictions'][oldhead]
  assert oi['checkpoint_sha256']==ob['checkpoint_sha256']==inspection['paired_folds'][str(fold)]['H2_epoch1_sha256']
  assert oi['producer_reported_prediction_sha256']==ob['sha256'] and oi['receipt_sha256']==ob['receipt_sha256']
  controls[str(fold)]=dict(run=oldhead,checkpoint=f'artifacts/runs/{oldhead}/model_ep1.pt',checkpoint_sha256=ob['checkpoint_sha256'],parent_epoch3_sha256=inspection['paired_folds'][str(fold)]['parent_epoch3_sha256'],prediction=ob,source_inspection=str((REF/'fit-inspection-recovery.json').relative_to(ROOT)),fresh_control_fit=False)
 assert len(jobs)==6
 template_predict=(TEMPLATE/'predict_fixed.py').read_text();actual_predict=(HERE/'predict_fixed.py').read_text()
 assert actual_predict==template_predict.replace('oct02_d07_sh256_s3h2_f','oct02_d07_sh256_s4h2_f')
 for name in ('check_checkpoint.py','run_queue.py'):assert (HERE/name).read_bytes()==(TEMPLATE/name).read_bytes()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 sources=[TRAINER,ROOT/'scripts/train_rnn.py',*sorted((ROOT/'src/wnn').glob('*.py')),*sorted(HERE.glob('*.py')),HERE/'protocol.md',REF/'commands.json',REF/'fit-inspection-recovery.json',REF/'bindings-v2-01.json',TEMPLATE/'commands.json',TEMPLATE/'predict_fixed.py',TEMPLATE/'check_checkpoint.py',TEMPLATE/'run_queue.py']
 result=dict(status='CONDITIONAL_SOURCE_ONLY_DISTILL07_SEED4',execution_authorized=False,
  activation_condition='Root judges fixed seed3 alpha.7 outcome supports independent seed4 budget, then issues separate immutable allocation. No automatic launch.',
  reference_commands=str((REF/'commands.json').relative_to(ROOT)),reference_sha256=sha(REF/'commands.json'),
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
  scientific_change='Only seed4 parent distill_alpha .5→.7; names/init remapped. Original h256_fF+h256s4_fF teachers unchanged; H2 recipe unchanged.',
  counts=dict(fits=4,OOF=2,fresh_control_fits=0),jobs=jobs,environment=template['environment'],
  existing_controls=controls,control_inspection_sha256=sha(REF/'fit-inspection-recovery.json'),control_bindings_sha256=sha(REF/'bindings-v2-01.json'),
  primary='Replace only shallow256 slot with .7 seed4 vs existing .5 seed4 in same equal-quarter ensemble with on_ctl128s3/on_ctl128s4/on_ctlc500; exposed pure-chain folds0/1, exact same scored rows.',
  secondary='Operational comparison to current .5 seed3 incumbent separately; root must freeze grader before scoring.',
  estimated_minutes=49,allocation=None,production_final_fit=False)
 write(HERE/'commands.json',result)
 write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ARGUMENT_PROOF_ONLY',method='AST-extracted argparse declarations only. No trainer import, Torch, arrays, checkpoint reads, fits, OOF or metrics.',trainer_sha256=TRAINER_SHA,reference_sha256=sha(REF/'commands.json'),commands_sha256=sha(HERE/'commands.json'),jobs=proof,predictor_exact_template_except_s4_guard=True,validator_byte_identical_to_template=True,runner_byte_identical_to_template=True,existing_control_inspection_all103args=True,existing_control_receipt_checkpoint_hashes_match_bindings=True,scientific_processes_started=0))
 print(json.dumps(dict(status=result['status'],commands_sha256=sha(HERE/'commands.json'),source_files=len(sources),jobs=len(jobs),fits=4,OOF=2,allocation=False)))
if __name__=='__main__':main()
