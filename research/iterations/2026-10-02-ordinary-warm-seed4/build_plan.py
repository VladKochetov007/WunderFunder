"""Conditional ordinary-warm seed4, archived103args/parents/control provenance only."""
from pathlib import Path
import argparse,ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
C=ROOT/'research/iterations/2026-10-01-continuation/student-seed4'
D=ROOT/'research/iterations/2026-10-02-distill07-seed4'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
W_SHA='bd13fb2d96fa898fc1bfd2bf7ba1eae75c742ea805f6633e54ba6adf4dbc1494'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
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
def main():
 assert sha(TRAINER)==TRAIN_SHA and sha(W/'commands.json')==W_SHA
 for name in ('train_warm.py','check_checkpoint.py'):assert (HERE/name).read_bytes()==(W/name).read_bytes()
 assert (HERE/'predict_fixed.py').read_text()==(W/'predict_fixed.py').read_text().replace('oct02_ow_s3h2_f','oct02_ow_s4h2_f')
 assert (HERE/'check_generated.py').read_text()==(W/'check_generated.py').read_text().replace('torch.manual_seed(3)','torch.manual_seed(4)')
 old=json.loads((C/'commands.json').read_text());warm=json.loads((W/'commands.json').read_text());reserve=json.loads((D/'commands.json').read_text())
 inspection=json.loads((C/'fit-inspection-recovery.json').read_text());bindings=json.loads((C/'bindings-v2-01.json').read_text())
 assert inspection['commands_sha256']==sha(C/'commands.json') and inspection['all103args_match'] and inspection['all_model_states_finite']
 p=parser();defaults=resolve(p,['--name','none']);jobs=[];proof=[];parents={};rel=str(HERE.relative_to(ROOT))
 for fold in (0,1):
  parent=f'oct02_ow_s4_f{fold}';head=f'oct02_ow_s4h2_f{fold}';oldparent=f'cont_sh256_s4_f{fold}'
  original_parent=next(j for j in old['jobs'] if j['name']==oldparent)
  parenthash=inspection['paired_folds'][str(fold)]['parent_epoch3_sha256']
  assert parenthash==reserve['existing_controls'][str(fold)]['parent_epoch3_sha256']
  parents[str(fold)]=dict(checkpoint=f'artifacts/runs/{oldparent}/model_ep3.pt',sha256=parenthash,resolved_args=original_parent['resolved_args'],inspection_path=str((C/'fit-inspection-recovery.json').relative_to(ROOT)),inspection_sha256=sha(C/'fit-inspection-recovery.json'))
  for stage,name,oldname,ep in [('parent',parent,oldparent,0),('H2',head,f'cont_sh256_s4h2_f{fold}',1)]:
   reference=next(j for j in old['jobs'] if j['name']==oldname);a=dict(reference['resolved_args']);a['name']=name
   assert len(a)==103 and a['seed']==4 and a['distill_alpha']==.5
   if stage=='parent':a.update(epochs=1,lr=.0005,init=oldparent+'@ep3');assert a['distill']==f'h256_f{fold},h256s4_f{fold}'
   else:a.update(init=parent+'@ep0');assert a['freeze']=='inp,gru' and a['epochs']==2
   argv=[]
   for k,v in a.items():
    if k!='name' and v==defaults[k]:continue
    flag='--'+k.replace('_','-')
    if isinstance(v,bool):assert v;argv.append(flag)
    else:argv.extend([flag,str(v)])
   assert resolve(p,argv)==a
   delta={k:dict(original=v,warm=a[k]) for k,v in reference['resolved_args'].items() if v!=a[k]}
   assert set(delta)==({'name','epochs','lr','init'} if stage=='parent' else {'name','init'})
   s3=next(j for j in warm['jobs'] if j['stage']==stage and j['fold']==fold)
   across={k:dict(seed3=v,seed4=a[k]) for k,v in s3['resolved_args'].items() if v!=a[k]}
   assert set(across)=={'name','seed','init'}
   jobs.append(dict(name=name,stage=stage,fold=fold,argv=['.venv/bin/python',rel+'/train_warm.py',*argv],resolved_args=a,expected_n_in=211,
    endpoint=f'artifacts/runs/{name}/model_ep{ep}.pt',reference_job=oldname,dependency=a['init'],wall_limit_seconds=1200 if stage=='parent' else 900,argument_changes=delta))
   proof.append(dict(name=name,resolved_arguments=103,exact=True,original_seed4_changes=delta,seed3_warm_changes=across))
  out=f'{rel}/cache/pred_{head}_valid.npy'
  jobs.append(dict(name=head+'_OOF',stage='OOF',fold=fold,argv=['.venv/bin/python',rel+'/predict_fixed.py',head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],dependency=head,wall_limit_seconds=600,
   require_absent=[out,out.replace('.npy','.npy.partial'),out.replace('.npy','.inputs.json')],required_outputs=[out,out.replace('.npy','.inputs.json')]))
 selected={f'{name}_f{fold}' for fold in (0,1) for name in ('cont_sh256_s4h2','cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')}
 predictions={name:bindings['predictions'][name] for name in sorted(selected)}
 for fold in (0,1):
  name=f'cont_sh256_s4h2_f{fold}';r=predictions[name];i=inspection['OOF'][name]
  assert r==reserve['existing_controls'][str(fold)]['prediction']
  assert r['checkpoint_sha256']==i['checkpoint_sha256']==inspection['paired_folds'][str(fold)]['H2_epoch1_sha256']
  assert r['sha256']==i['producer_reported_prediction_sha256'] and r['receipt_sha256']==i['receipt_sha256']
 baseline=dict(status='SOURCE_ONLY_REUSED_SEED4_COMPARATOR_BINDINGS',primary_control=['cont_sh256_s4h2','on_ctl128s3','on_ctl128s4','on_ctlc500'],
  secondary_incumbent=['cont_sh256_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500'],weights=[.25]*4,predictions=predictions,data=bindings['data'],
  source_bindings=dict(path=str((C/'bindings-v2-01.json').relative_to(ROOT)),sha256=sha(C/'bindings-v2-01.json')),
  fit_inspection=dict(path=str((C/'fit-inspection-recovery.json').relative_to(ROOT)),sha256=sha(C/'fit-inspection-recovery.json')),
  note='Metadata copied from completed inspection/bindings; no forecast arrays or targets read, no scores inferred. Fixed member order and pooled clipping must be preserved. No fresh baseline fits.')
 write(HERE/'baseline-cache-manifest.json',baseline)
 sources=[TRAINER,*sorted((ROOT/'src/wnn').glob('*.py')),*sorted(HERE.glob('*.py')),HERE/'protocol.md',HERE/'baseline-cache-manifest.json',
  W/'commands.json',W/'train_warm.py',W/'predict_fixed.py',W/'check_checkpoint.py',W/'check_generated.py',C/'commands.json',C/'fit-inspection-recovery.json',C/'bindings-v2-01.json',D/'commands.json']
 result=dict(status='CONDITIONAL_SOURCE_ONLY_ORDINARY_WARM_SEED4',execution_authorized=False,allocation=None,
  activation_condition='Root may allocate only after seed3 ordinary-warm fixed readout is useful enough to spend replication budget; no automatic activation.',
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},counts=dict(fits=4,OOF=2),jobs=jobs,plain_parents=parents,environment=warm['environment'],estimated_minutes=[12,22],
  baseline_cache_manifest=dict(path=str((HERE/'baseline-cache-manifest.json').relative_to(ROOT)),sha256=sha(HERE/'baseline-cache-manifest.json')),
  primary='seed4 ordinary_warm4 minus ORIGINAL seed4four; same three quarter anchors',secondary='seed4 ordinary_warm4 minus current seed3 student4 separately',
  no_final_training=True,rng_pairing='Same discarded Linear256to32 + GRU33to64 constructors as W both stages, now seed4; ordinary state stays12keys.')
 write(HERE/'commands.json',result)
 write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ONLY_SEED4_WARM_103_ARGUMENTS',commands_sha256=sha(HERE/'commands.json'),jobs=proof,adapter_and_validator_byte_identical_to_W=True,
  predictor_only_seed4_name_guard=True,generated_test_only_seed4_rng=True,baseline_provenance_matches_two_independent_preserved_metadata_documents=True,empirical_executions=0))
 print(json.dumps(dict(status=result['status'],commands_sha256=sha(HERE/'commands.json'),baseline_manifest_sha256=sha(HERE/'baseline-cache-manifest.json'),fits=4,OOF=2)))
if __name__=='__main__':main()
