"""Paired one-parent-epoch + H2 protocol from archived ordinary seed3 arguments."""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REFERENCE=ROOT/'research/iterations/2026-10-01-continuation/student'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
MODEL_SHA='7ef61881baf1ad0befe229ea087c958f9aad6aa65240c0b3d8580dd90015d241'
FREEZE='inp,gru,interval_proj,event_gru'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse(TRAINER.read_text())):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   kw={k.arg:({'int':int,'float':float,'str':str}[k.value.id] if k.arg=='type' else ast.literal_eval(k.value)) for k in n.keywords}
   p.add_argument(*[ast.literal_eval(x) for x in n.args],**kw)
 p.add_argument('--interval-arm',choices=['accumulated','instantaneous'],required=True)
 return p

def resolve(p,argv):
 a=vars(p.parse_args(argv))
 if a['fast']:
  for k in ('sync_free','fused_adam','pin_memory','prop_cache'):a[k]=True
 a.update(interval_source_sha256=MODEL_SHA,interval_trainer_source_sha256=TRAIN_SHA)
 return a

def main():
 assert sha(TRAINER)==TRAIN_SHA and sha(HERE/'interval_memory.py')==MODEL_SHA
 source=json.loads((REFERENCE/'commands.json').read_text());p=parser();defaults=resolve(p,['--name','none','--interval-arm','accumulated'])
 jobs=[];proof=[];parents={};rel=str(HERE.relative_to(ROOT))
 for fold in (0,1):
  oldparent=next(j for j in source['jobs'] if j['name']==f'cont_sh256_s3_f{fold}')
  inspection=json.loads((REFERENCE/f'fit-inspection-f{fold}.json').read_text())['checkpoints']['parent']
  assert inspection['checkpoint']==oldparent['endpoint'] and inspection['epoch']==3
  parents[str(fold)]=dict(checkpoint=inspection['checkpoint'],sha256=inspection['sha256'],resolved_args=oldparent['resolved_args'],inspection_path=str((REFERENCE/f'fit-inspection-f{fold}.json').relative_to(ROOT)),inspection_sha256=sha(REFERENCE/f'fit-inspection-f{fold}.json'))
  for arm in ('instantaneous','accumulated'):
   parent=f'oct02_im_{arm}_s3_f{fold}';head=f'oct02_im_{arm}_s3h2_f{fold}'
   for stage,oldname,name,ep in [('parent',f'cont_sh256_s3_f{fold}',parent,0),('H2',f'cont_sh256_s3h2_f{fold}',head,1)]:
    old=next(j for j in source['jobs'] if j['name']==oldname);a={**old['resolved_args'],'name':name,'interval_arm':arm,'interval_source_sha256':MODEL_SHA,'interval_trainer_source_sha256':TRAIN_SHA}
    if stage=='parent':a.update(epochs=1,lr=.0005,init=oldname+'@ep3')
    else:a.update(init=parent+'@ep0',freeze=FREEZE)
    argv=[]
    for k,v in a.items():
     if k in ('interval_source_sha256','interval_trainer_source_sha256'):continue
     if k not in ('name','interval_arm') and v==defaults[k]:continue
     flag='--'+k.replace('_','-')
     if isinstance(v,bool):assert v;argv.append(flag)
     else:argv.extend([flag,str(v)])
    assert resolve(p,argv)==a and len(a)==106
    delta={k:dict(old=v,new=a[k]) for k,v in old['resolved_args'].items() if a[k]!=v}
    assert set(delta)==({'name','epochs','lr','init'} if stage=='parent' else {'name','init','freeze'})
    assert a['distill']==(f'h256_f{fold},h256s4_f{fold}' if stage=='parent' else '') and a['distill_alpha']==.5
    j=dict(name=name,stage=stage,fold=fold,arm=arm,argv=['.venv/bin/python',rel+'/train_interval.py',*argv],resolved_args=a,expected_n_in=211,endpoint=f'artifacts/runs/{name}/model_ep{ep}.pt',reference_job=oldname,argument_changes=delta,dependency=a['init'],wall_limit_seconds=1200 if stage=='parent' else 900)
    jobs.append(j);proof.append(dict(name=name,resolved_arguments=106,exact=True,changes=delta,model_only_args={k:a[k] for k in ('interval_arm','interval_source_sha256','interval_trainer_source_sha256')}))
   out=f'{rel}/cache/pred_{head}_valid.npy'
   jobs.append(dict(name=head+'_OOF',stage='OOF',fold=fold,arm=arm,argv=['.venv/bin/python',rel+'/predict_fixed.py',head,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],require_absent=[out,out.replace('.npy','.npy.partial'),out.replace('.npy','.inputs.json')],required_outputs=[out,out.replace('.npy','.inputs.json')],dependency=head,wall_limit_seconds=600))
 for fold in (0,1):
  for stage in ('parent','H2'):
   left,right=[j['resolved_args'] for j in jobs if j['fold']==fold and j['stage']==stage]
   diff={k for k in left if left[k]!=right[k]};assert diff==({'name','interval_arm'} if stage=='parent' else {'name','init','interval_arm'})
 source_paths=[TRAINER,HERE/'interval_memory.py',HERE/'train_interval.py',HERE/'predict_fixed.py',HERE/'check_checkpoint.py',Path(__file__),HERE/'training-protocol.md',REFERENCE/'commands.json',*[REFERENCE/f'fit-inspection-f{f}.json' for f in (0,1)],*sorted((ROOT/'src/wnn').glob('*.py'))]
 result=dict(status='SOURCE_ONLY_PAIRED_INTERVAL_WARMSTART',execution_authorized=False,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_paths},counts=dict(fits=8,OOF=4),jobs=jobs,plain_parents=parents,environment=source['environment'],estimated_minutes=[25,45],generated_CUDA_preflight_required=True,allocation=None,scientific_contrast='accumulated versus instantaneous same modules, paired plain EMAep3 warmstarts, one new parent epoch0 then strict own-parent frozen-all-trunk H2ep1; serving-quarter replacement owned by root')
 write(HERE/'commands.json',result)
 write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ONLY_106_ARGUMENTS',trainer_sha256=TRAIN_SHA,model_sha256=MODEL_SHA,commands_sha256=sha(HERE/'commands.json'),jobs=proof,paired_arm_differences_only=['name','interval_arm','H2 own-arm init'],original_trainer_source_edits=['model.load_state_dict(init["model"]) replaced once by initialize_model(model, init, args)'],parent_normalization_refit=False,forecast_fits=0,model_tensor_reads=0,quality_reads=0))
 print(json.dumps(dict(status=result['status'],jobs=len(jobs),fits=8,OOF=4,commands_sha256=sha(HERE/'commands.json'))))
if __name__=='__main__':main()
