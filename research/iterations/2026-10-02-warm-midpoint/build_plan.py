"""Bind fixed midpoint sources from existing metadata only; no weights/data read."""
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
 rel=str(HERE.relative_to(ROOT));pairs=[];jobs=[];metadata=[]
 for seed in (3,4):
  C=ROOT/'research/iterations/2026-10-01-continuation'/('student' if seed==3 else 'student-seed4')
  W=ROOT/('research/iterations/2026-10-02-ordinary-warm'+('' if seed==3 else '-seed4'))
  cp=C/'commands.json';wp=W/'commands.json';oldplan=json.loads(cp.read_text());warmplan=json.loads(wp.read_text());metadata.extend([cp,wp])
  for fold in (0,1):
   oldname=f'cont_sh256_s{seed}h2_f{fold}';warmname=f'oct02_ow_s{seed}h2_f{fold}';name=f'oct02_wm_s{seed}h2_f{fold}'
   old=next(j for j in oldplan['jobs'] if j['name']==oldname);warm=next(j for j in warmplan['jobs'] if j['name']==warmname)
   oa=old['resolved_args'];wa=warm['resolved_args'];assert len(oa)==len(wa)==103
   assert {k for k in oa if oa[k]!=wa[k]}=={'name','init'}
   assert oa['seed']==wa['seed']==seed and oa['valid_fold']==wa['valid_fold']==fold
   assert oa['arch']=='gru' and oa['epochs']==2 and oa['freeze']=='inp,gru' and oa['hidden']==256 and oa['layers']==1
   op=f'artifacts/runs/{oldname}/model_ep1.pt';wp_model=f'artifacts/runs/{warmname}/model_ep1.pt'
   if seed==3:
    ip=C/f'fit-inspection-f{fold}.json';ir=json.loads(ip.read_text());b=ir['checkpoints']['H2'];assert b['checkpoint']==op and b['epoch']==1;oh=b['sha256']
   else:
    ip=C/'fit-inspection-recovery.json';ir=json.loads(ip.read_text());b=ir['all_new_run_files'][op];assert b['epoch']==1;oh=b['sha256']
   rp=W/f'execution-01/{warmname}.H2.worker-result.json';r=json.loads(rp.read_text());assert r['status']=='COMPLETE'
   wh=r['outputs'][wp_model]['sha256'];metadata.extend([ip,rp]);args=oa.copy();args['name']=name
   endpoint=f'artifacts/runs/{name}/model_ep1.pt'
   pairs.append(dict(name=name,seed=seed,fold=fold,alpha=.5,endpoint=endpoint,resolved_args=args,
    old=dict(path=op,sha256=oh,resolved_args=oa,metadata_path=str(ip.relative_to(ROOT)),metadata_sha256=sha(ip)),
    warm=dict(path=wp_model,sha256=wh,resolved_args=wa,metadata_path=str(rp.relative_to(ROOT)),metadata_sha256=sha(rp))))
   out=f'{rel}/cache/pred_{name}_valid.npy'
   jobs.append(dict(name=name+'_OOF',stage='OOF',seed=seed,fold=fold,argv=['.venv/bin/python',rel+'/predict_fixed.py',name,'--seed',str(seed),'--fold',str(fold),'--epoch','1','--batch','16','--output-dir',rel+'/cache'],dependency=endpoint,wall_limit_seconds=300,
    require_absent=[out,out.replace('.npy','.npy.partial'),out.replace('.npy','.inputs.json')],required_outputs=[out,out.replace('.npy','.inputs.json')]))
 names=['midpoint.py','construct.py','check_checkpoint.py','check_generated.py','predict_fixed.py','run_queue.py','launch.py','protocol.md','build_plan.py']
 sources=[*[HERE/n for n in names],*metadata,ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py',*sorted((ROOT/'src/wnn').glob('*.py'))]
 for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 plan=dict(status='SOURCE_ONLY_FIXED_WARM_MIDPOINT',execution_authorized=False,alpha=.5,counts=dict(fits=0,OOF=4),pairs=pairs,jobs=jobs,
  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},success_status='COMPLETE_0_FITS_4_OOF_NO_METRICS',
  construction=dict(argv=['.venv/bin/python',rel+'/construct.py'],post_argv=['.venv/bin/python',rel+'/check_checkpoint.py','--output',rel+'/prepare-01/checkpoints.json'],
   cpu_ids=[28],cpu_quota=100,host_memory_bytes=4*2**30,timeout_seconds=60,
   required_outputs=[q for p in pairs for q in (p['endpoint'],str(Path(p['endpoint']).parent/'construction.json'))]+[rel+'/prepare-01/checkpoints.json']),
  proposed_resources=dict(gpu=0,cpu_ids=list(range(16,22)),allocator_fraction=.375,owned_gpu_max_MiB=7168,wall_limit_seconds=1200),
  fixed_gate=dict(seed3_pooled_vs_original_min=.0001,seed4_pooled_vs_original_strict_min=0,all_four_fold_deltas_strict_min=0,no_seed_selection=True),
  allocation=None)
 write(HERE/'commands.json',plan)
 write(HERE/'source-proof.json',dict(status='PASS_SOURCE_ONLY_FIXED_MIDPOINT_BINDINGS',commands_sha256=sha(HERE/'commands.json'),
  input_checkpoints=8,outputs=4,alpha=.5,all_pairs_103args_old_warm_name_init_only=True,forecast_fits=0,data_reads=0,quality_reads=0))
 print(json.dumps(dict(status=plan['status'],commands_sha256=sha(HERE/'commands.json'),input_checkpoints=8,fits=0,OOF=4)))
if __name__=='__main__':main()
