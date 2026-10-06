"""Root-only native allocation after fixed two-seed midpoint gate; never launches."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
M=ROOT/'research/iterations/2026-10-02-warm-midpoint'
C=ROOT/'research/iterations/2026-10-01-continuation/student/native-quality-01'
TEMPLATE_SHA='f68ae688d7532ebb4c638807d71f1271d1791ccd6cb573cf5be36a632131d199'
GRADER_SHA='41b39bc0cbbe74a6ed499e09c50b552e27031701bccc044d92f3203d8865ae65'
M_SHA='79a92747abd1983ca575eee2c479bd3dd0a5ac3eeaaaf3c585885440e28af885'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def repo(relative):
 p=Path(relative);assert not p.is_absolute() and '..' not in p.parts
 return ROOT/p
def main():
 for name in ('allocation.json','supervisor.json','supervisor-result.json','controller.json','controller.log','prepared.json','fold0','fold1','cache','evaluation'):
  path=P/name;assert not path.exists() and not path.is_symlink(),('Existing native run',path)
 template=P/'allocation-template.json';assert sha(template)==TEMPLATE_SHA
 a=json.loads(template.read_text());assert a['authorized'] is False
 for rel,h in a['source_sha256'].items():assert sha(repo(rel))==h,rel
 assert sha(P/'evaluate.py')==GRADER_SHA
 assert a['members']==['oct02_wm_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500'] and a['coefficients']==[.25]*4
 assert a['lanes']=={'0':30,'1':35} and a['prediction_wall_seconds_each']==1800 and a['memory_bytes_each']==4*2**30
 tracked={}
 def pin(path,expected=None):
  path=Path(path);h=sha(path)
  if expected is not None:assert h==expected,('Provenance hash differs',str(path))
  rel=str(path.relative_to(ROOT))
  if rel in tracked:assert tracked[rel]==h
  tracked[rel]=h;return h
 ap=M/'training-allocation.json';ah=pin(ap);allocation=json.loads(ap.read_text())
 assert allocation['execution_authorized'] and allocation['success_status']=='COMPLETE_0_FITS_4_OOF_NO_METRICS'
 ep=M/'execution-01/result.json';pin(ep);execution=json.loads(ep.read_text())
 assert execution['status']==allocation['success_status'] and execution['allocation_sha256']==ah and execution['failure'] is None
 assert (execution['fits'],execution['OOF_jobs'],execution['metric_calls'],execution['retries'])==(0,4,0,0)
 assert len(execution['jobs'])==len(allocation['jobs'])==4
 assert [(j['name'],j['stage']) for j in allocation['jobs']]==[(f'oct02_wm_s{s}h2_f{f}_OOF','OOF') for s in (3,4) for f in (0,1)]
 for done,job in zip(execution['jobs'],allocation['jobs']):
  assert done['name']==job['name'] and done['stage']==job['stage'] and done['exit_code']==0 and done['failure'] is None
  assert not done['remaining_scope_pids'] and not done.get('cleanup_error')
  record=done['worker_result'];wp=repo(record['path']);pin(wp,record['sha256']);worker=json.loads(wp.read_text())
  assert worker['status']=='COMPLETE' and worker['allocation_sha256']==ah and worker['name']==job['name']
  assert worker['stage']==job['stage'] and worker['argv']==job['argv'] and worker['returncode']==0
  assert set(worker['outputs'])==set(job['required_outputs'])
 fp=M/'finish-01/result.json';pin(fp);finish=json.loads(fp.read_text())
 assert finish['status']=='COMPLETE_FIXED_FORECAST_READOUT' and finish['allocation_sha256']==ah
 assert finish['exit_code']==0 and not finish['remaining_scope_pids']
 freeze=M/'evaluation-source.json';fh=pin(freeze);frozen=json.loads(freeze.read_text())
 assert frozen['status']=='FROZEN_BEFORE_METRIC_READOUT' and frozen['training_allocation_sha256']==ah
 assert finish['evaluation_source_sha256']==fh
 pin(M/'evaluate.py',M_SHA);assert frozen['source_sha256'][str((M/'evaluate.py').relative_to(ROOT))]==M_SHA
 rp=M/'evaluation-01/report.json';rh=pin(rp,finish['report_sha256']);report=json.loads(rp.read_text())
 receipt_path=M/'evaluation-01/receipt.json';pin(receipt_path);receipt=json.loads(receipt_path.read_text())
 assert report['status']==receipt['status']=='COMPLETE_FIXED_WARM_MIDPOINT_SCREEN'
 assert report['source_sha256']==receipt['source_sha256']==M_SHA
 assert report['allocation_sha256']==receipt['allocation_sha256']==ah and report['evaluation_source_sha256']==receipt['evaluation_source_sha256']==fh
 assert receipt['outputs']['report.json']['sha256']==rh
 stats=M/'evaluation-01/sufficient.npz';pin(stats,receipt['outputs']['sufficient.npz']['sha256']);assert stats.stat().st_size==receipt['outputs']['sufficient.npz']['bytes']
 assert (report['fits'],report['OOF_jobs'],report['sequences_per_seed'],report['chains'],report['parameter_fraction'])==(0,4,860,250,.5)
 evidence={}
 for label,seed,candidate,control in (('primary',3,'midpoint_seed3_4','current_student4'),('confirmation',4,'midpoint_seed4_4','original_seed4_4')):
  contrast=report['comparisons'][label];assert contrast['candidate']==candidate and contrast['control']==control
  rows=contrast['scopes']['all'];assert set(rows['folds'])=={'0','1'}
  deltas={'pooled':float(rows['delta_WP']),**{f:float(rows['folds'][f]['delta_WP']) for f in ('0','1')}}
  assert all(math.isfinite(v) for v in deltas.values()) and all(deltas[f]>0 for f in ('0','1'))
  assert deltas['pooled']>=.0001 if seed==3 else deltas['pooled']>0
  evidence[str(seed)]=dict(candidate=candidate,control=control,existing_report_delta_WP=deltas)
 assert report['gate']==dict(seed3_pooled=True,seed4_pooled=True,seed3_folds=True,seed4_folds=True,passed=True)
 # Construction proofs bind the midpoint, never merely a renamed warm checkpoint.
 prep_allocation=M/'preparation-allocation.json';ph=pin(prep_allocation);pa=json.loads(prep_allocation.read_text())
 assert pa['execution_authorized'] and pa['CPU']==28 and pa['wall_seconds']==60 and pa['fits']==pa['OOF']==0
 proof={}
 def proofpin(path,expected=None):
  h=pin(path,expected);proof[str(path.relative_to(ROOT))]=h;return h
 prep_result=M/'prepare-01/result.json';proofpin(prep_result);pr=json.loads(prep_result.read_text())
 assert pr['status']=='COMPLETE_FIXED_MIDPOINT_CPU_PREPARATION' and pr['allocation_sha256']==ph and pr['exit_code']==0 and not pr['remaining_scope_pids']
 gp=M/'prepare-01/generated.log';proofpin(gp,pr['generated_sha256']);assert json.loads(gp.read_text())['status']=='PASS_GENERATED_FIXED_WARM_MIDPOINT'
 checkpath=M/'prepare-01/checkpoints.json';proofpin(checkpath,pr['checkpoints_sha256']);checked=json.loads(checkpath.read_text())
 assert checked['status']=='PASS_FIXED_WARM_MIDPOINT_CHECKPOINTS'
 cp=M/'commands.json';ch=pin(cp);commands=json.loads(cp.read_text());assert checked['commands_sha256']==ch
 assert ch==allocation['sources'][str(cp.relative_to(ROOT))] and commands['alpha']==.5 and commands['counts']==dict(fits=0,OOF=4)
 assert checked['checker_sha256']==pin(M/'check_checkpoint.py')
 assert checked['forecast_fits']==checked['model_forward_calls']==checked['data_reads']==checked['metrics']==0
 assert [q['name'] for q in checked['checkpoints']]==[p['name'] for p in commands['pairs']]==[f'oct02_wm_s{s}h2_f{f}' for s in (3,4) for f in (0,1)]
 hashes={}
 for pair,q in zip(commands['pairs'],checked['checkpoints']):
  run=pair['name'];seed=pair['seed'];fold=pair['fold'];assert q==report['construction_integrity'][run]
  assert q['seed']==seed and q['fold']==fold and q['alpha']==.5 and q['mean_std_byte_identical'] and q['all10_parameters_exact_midpoint'] and q['ordinary_strict_load']
  assert q['checkpoint']==dict(path=pair['endpoint'],sha256=q['checkpoint']['sha256'],epoch=1)
  for role in ('old','warm'):assert q[role+'_checkpoint']==dict(path=pair[role]['path'],sha256=pair[role]['sha256'],epoch=1)
  crp=repo(pair['endpoint']).with_name('construction.json');proofpin(crp,q['construction_receipt_sha256']);cr=json.loads(crp.read_text())
  assert cr['status']=='CONSTRUCTED_FIXED_PARAMETER_MIDPOINT' and cr['name']==run and cr['seed']==seed and cr['fold']==fold and cr['alpha']==.5
  assert cr['checkpoint']==q['checkpoint'] and cr['old_checkpoint']==q['old_checkpoint'] and cr['warm_checkpoint']==q['warm_checkpoint']
  assert cr['mean_std_byte_identical'] and cr['all10_parameters_exact_midpoint'] and cr['ordinary_strict_load']
  assert cr['commands_sha256']==ch and cr['constructor_sha256']==pin(M/'construct.py') and cr['midpoint_source_sha256']==pin(M/'midpoint.py')
  if seed==3:hashes[str(fold)]=pin(repo(pair['endpoint']),q['checkpoint']['sha256'])
 assert set(hashes)=={'0','1'}
 # CURRENT student_native is the reference; the older strong4 is not selected.
 pin(C/'prepared.json',a['reference_prepared_sha256'])
 oldrp=C/'evaluation/report.json';pin(oldrp);old=json.loads(oldrp.read_text())
 assert old['status']=='COMPLETE_ONE_FIXED_NATIVE_COMPARISON' and old['evaluation_source_sha256']=='9dd71d06373121176133e1cfa3bd2455eabcd40f35f3e38824908c52a88f4986'
 pin(C/'evaluation/sufficient.npz',old['sufficient_sha256'])
 for f in (0,1):
  rp=C/'cache'/f'pred_student_quant_f{f}_valid.json';pin(rp,a['reference_receipt_sha256'][str(f)]);r=json.loads(rp.read_text())
  assert r['prediction_sha256']==a['reference_prediction_sha256'][str(f)] and r['native_sha256']==a['native_sha256']
  assert r['coefficients']==[.25]*4 and r['fold']==f and r['epoch']==1
  assert [c['run'] for c in r['child_checkpoints']]==[f'{n}_f{f}' for n in ('cont_sh256_s3h2',*a['members'][1:])]
 required=set(a['evaluation_input_sha256']);assert len(required)==6 and required<=set(tracked)
 a['evaluation_input_sha256']={rel:tracked[rel] for rel in a['evaluation_input_sha256']}
 a['midpoint_checkpoint_sha256']=hashes;a['midpoint_proof_sha256']=proof
 for path in (P/'evaluate.py',Path(__file__),template,M/'evaluate.py',C/'evaluate.py'):a['source_sha256'][str(path.relative_to(ROOT))]=pin(path)
 a.update(authorized=True,source_preparation_only=False,created_utc=datetime.now(timezone.utc).isoformat(),
  authorization='Explicit root invocation after fixed midpoint gate; two native OOF and one fixed readout only.',
  activation_evidence=evidence,prerequisite_sha256=tracked,production_seed='Fixed seed3; no best-seed choice',allocation_builder_sha256=sha(Path(__file__)))
 a.pop('root_fill_required',None)
 for rel,h in tracked.items():assert sha(repo(rel))==h,('Prerequisite changed during allocation',rel)
 out=P/'allocation.json'
 with out.open('x') as stream:json.dump(a,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',allocation_sha256=sha(out),OOF_passes=2,metric_readouts=1,members=a['members'],lanes=a['lanes'],prerequisites=evidence)))
if __name__=='__main__':main()
