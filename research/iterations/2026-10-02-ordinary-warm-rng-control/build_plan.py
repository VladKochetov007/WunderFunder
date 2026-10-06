"""Source-only exact 103-argument original-parent H2 RNG-control plan."""
from pathlib import Path
import argparse,ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
C=ROOT/'research/iterations/2026-10-01-continuation/student'
TRAINER=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
WSHA='bd13fb2d96fa898fc1bfd2bf7ba1eae75c742ea805f6633e54ba6adf4dbc1494'
TSHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
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
def write(path,data):
 with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def main():
 assert sha(W/'commands.json')==WSHA and sha(TRAINER)==TSHA
 warm=json.loads((W/'commands.json').read_text());historical=json.loads((C/'commands.json').read_text())
 p=parser();defaults=resolve(p,['--name','none']);rel=str(HERE.relative_to(ROOT));jobs=[];proof=[];meta=[]
 for fold in (0,1):
  old=next(j for j in warm['jobs'] if j['stage']=='H2' and j['fold']==fold)
  hist=next(j for j in historical['jobs'] if j['name']==f'cont_sh256_s3h2_f{fold}')
  a=old['resolved_args'].copy();name=f'oct02_ow_rng_s3h2_f{fold}'
  a.update(name=name,init=f'cont_sh256_s3_f{fold}@ep3');argv=[]
  for k,v in a.items():
   if k!='name' and v==defaults[k]:continue
   flag='--'+k.replace('_','-')
   if isinstance(v,bool):assert v;argv.append(flag)
   else:argv.extend([flag,str(v)])
  assert len(a)==103 and resolve(p,argv)==a
  delta={k:dict(warm=old['resolved_args'][k],control=a[k]) for k in a if a[k]!=old['resolved_args'][k]}
  assert set(delta)=={'name','init'}
  assert {k for k in a if a[k]!=hist['resolved_args'][k]}=={'name'}
  wp=W/f'execution-01/{old["name"]}.H2.worker-result.json';wr=json.loads(wp.read_text());assert wr['status']=='COMPLETE'
  rp=f'artifacts/runs/{old["name"]}/initial-state.json';binding=dict(path=rp,sha256=wr['outputs'][rp]['sha256'],bytes=wr['outputs'][rp]['bytes'])
  meta.append(wp)
  jobs.append(dict(name=name,stage='H2',fold=fold,argv=['.venv/bin/python',rel+'/train_warm.py',*argv],resolved_args=a,
   expected_n_in=211,endpoint=f'artifacts/runs/{name}/model_ep1.pt',dependency=a['init'],wall_limit_seconds=300,
   warm_initial_receipt=binding,warm_reference=old['name'],historical_reference=hist['name'],argument_changes=delta))
  proof.append(dict(name=name,resolved_arguments=103,exact=True,warm_changes=delta,historical_changes=['name']))
  out=f'{rel}/cache/pred_{name}_valid.npy'
  jobs.append(dict(name=name+'_OOF',stage='OOF',fold=fold,argv=['.venv/bin/python',rel+'/predict_fixed.py',name,'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],dependency=name,wall_limit_seconds=300,
   require_absent=[out,out.replace('.npy','.npy.partial'),out.replace('.npy','.inputs.json')],required_outputs=[out,out.replace('.npy','.inputs.json')]))
 paths=[TRAINER,W/'commands.json',W/'train_warm.py',C/'commands.json',*meta,
  *[HERE/n for n in ('train_warm.py','predict_fixed.py','check_checkpoint.py','check_generated.py','run_queue.py','launch.py','design.md','protocol.md','build_plan.py')],
  *sorted((ROOT/'src/wnn').glob('*.py'))]
 plan=dict(status='SOURCE_ONLY_ORDINARY_WARM_RNG_CONTROL',execution_authorized=False,counts=dict(fits=2,OOF=2),jobs=jobs,
  plain_parents=warm['plain_parents'],environment=warm['environment'],source_sha256={str(x.relative_to(ROOT)):sha(x) for x in paths},
  proposed_resources=dict(gpu=0,cpu_ids=list(range(16,22)),allocator_fraction=.375,owned_gpu_max_MiB=7168,wall_limit_seconds=1200),
  success_status='COMPLETE_2_FITS_2_OOF_NO_METRICS',estimated_minutes=[4,10],allocation=None,
  scientific_contrast='W4 minus RNG-control4; explanatory RNG-control4 minus historical student4. Mechanistic only, no promotion.')
 write(HERE/'commands.json',plan)
 write(HERE/'argument-proof.json',dict(status='PASS_SOURCE_ONLY_103_ARGUMENTS',commands_sha256=sha(HERE/'commands.json'),warm_commands_sha256=WSHA,jobs=proof,forecast_fits=0,data_reads=0,quality_reads=0))
 print(json.dumps(dict(status=plan['status'],commands_sha256=sha(HERE/'commands.json'),fits=2,OOF=2)))
if __name__=='__main__':main()
