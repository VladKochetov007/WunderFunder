"""Root-callable allocation builder only; never launches native preparation/OOF.

Requires completed paired-positive seed3 AND seed4 studies. The exported seed is
fixed to seed3; seed4 is confirmation against its OWN original seed4 baseline.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math

P=Path(__file__).resolve().parent;ROOT=P.parents[3]
W=ROOT/'research/iterations/2026-10-02-ordinary-warm'
W4=ROOT/'research/iterations/2026-10-02-ordinary-warm-seed4'
C=ROOT/'research/iterations/2026-10-01-continuation/student/native-quality-01'
TEMPLATE_SHA='699b62af0a25c1d1c43dacfa7b2d6937f36d20de898b028b87f362a94fcddf98'
GRADER_SHA='7b552140ed44d82984c861f3eae02cccd6d160b429977c0ef7a8aa95ff261c54'
W_SHA='f29b97efa4e767cb85f83e9c44dcee678ed4b84e9f9e28dce18e31f22d6cf865'
W4_SHA='adcf8d422dd6bc63bf76b5c72e087d830c9984c1e8ca03d96f72789239fdf6c5'

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
 assert a['members']==['oct02_ow_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500'] and a['coefficients']==[.25]*4
 assert a['lanes']=={'0':30,'1':35} and a['prediction_wall_seconds_each']==1800 and a['memory_bytes_each']==4*2**30
 prerequisites={};tracked={}
 def pin(path,expected=None):
  path=Path(path);h=sha(path)
  if expected is not None:assert h==expected,('Provenance hash differs',str(path))
  rel=str(path.relative_to(ROOT))
  if rel in tracked:assert tracked[rel]==h
  tracked[rel]=h;return h

 def completed_study(directory,status,grader_sha,candidate,control):
  ap=directory/'training-allocation.json';allocation_hash=pin(ap);allocation=json.loads(ap.read_text())
  assert allocation['execution_authorized'] and allocation['success_status']=='COMPLETE_4_FITS_2_OOF_NO_METRICS'
  ep=directory/'execution-01/result.json';pin(ep);execution=json.loads(ep.read_text())
  assert execution['status']==allocation['success_status'] and execution['allocation_sha256']==allocation_hash
  assert execution['failure'] is None and (execution['fits'],execution['OOF_jobs'],execution['metric_calls'],execution['retries'])==(4,2,0,0)
  assert len(execution['jobs'])==len(allocation['jobs'])==6
  workers={}
  for done,job in zip(execution['jobs'],allocation['jobs']):
   assert done['name']==job['name'] and done['stage']==job['stage'] and done['exit_code']==0 and done['failure'] is None
   assert not done['remaining_scope_pids'] and not done.get('cleanup_error')
   record=done['worker_result'];wp=repo(record['path']);pin(wp,record['sha256']);worker=json.loads(wp.read_text())
   assert worker['status']=='COMPLETE' and worker['allocation_sha256']==allocation_hash and worker['name']==job['name']
   assert worker['stage']==job['stage'] and worker['argv']==job['argv'] and worker['returncode']==0
   assert set(worker['outputs'])==set(job['required_outputs'])
   if job.get('post_argv'):assert worker['post_returncode']==0
   workers[job['name']]=worker
  fp=directory/'finish-01/result.json';pin(fp);finish=json.loads(fp.read_text())
  assert finish['status']=='COMPLETE_FIXED_FORECAST_READOUT' and finish['allocation_sha256']==allocation_hash
  assert finish['exit_code']==0 and not finish['remaining_scope_pids']
  freeze=directory/'evaluation-source.json';freeze_hash=pin(freeze);frozen=json.loads(freeze.read_text())
  assert frozen['status']=='FROZEN_BEFORE_METRIC_READOUT' and frozen['training_allocation_sha256']==allocation_hash
  assert finish['evaluation_source_sha256']==freeze_hash
  pin(directory/'evaluate.py',grader_sha)
  assert frozen['source_sha256'][str((directory/'evaluate.py').relative_to(ROOT))]==grader_sha
  rp=directory/'evaluation-01/report.json';report_hash=pin(rp,finish['report_sha256']);report=json.loads(rp.read_text())
  receipt_path=directory/'evaluation-01/receipt.json';pin(receipt_path);receipt=json.loads(receipt_path.read_text())
  assert report['status']==receipt['status']==status and report['source_sha256']==receipt['source_sha256']==grader_sha
  assert report['allocation_sha256']==receipt['allocation_sha256']==allocation_hash
  assert report['evaluation_source_sha256']==receipt['evaluation_source_sha256']==freeze_hash
  assert receipt['outputs']['report.json']['sha256']==report_hash
  stats=directory/'evaluation-01/sufficient.npz';pin(stats,receipt['outputs']['sufficient.npz']['sha256'])
  assert stats.stat().st_size==receipt['outputs']['sufficient.npz']['bytes']
  contrast=report['comparisons']['primary'];assert contrast['candidate']==candidate and contrast['control']==control
  allrows=contrast['scopes']['all'];assert set(allrows['folds'])=={'0','1'}
  deltas={scope:float(record['delta_WP']) for scope,record in [('pooled',allrows),*[(f,allrows['folds'][f]) for f in ('0','1')]]}
  assert all(math.isfinite(v) and v>0 for v in deltas.values()),('Paired-positive prerequisite failed',directory.name,deltas)
  assert (report['fits'],report['OOF_jobs'],report['sequences'],report['chains'])==(4,2,860,250)
  prerequisites[directory.name]=dict(status=status,report_sha256=report_hash,allocation_sha256=allocation_hash,
   candidate=candidate,control=control,existing_report_delta_WP=deltas,rule='Pooled and BOTH fold deltas strictly positive; primary only; no interval/seed selection')
  return report,workers

 wr,workers=completed_study(W,'COMPLETE_FIXED_ORDINARY_WARM_SCREEN',W_SHA,'ordinary_warm4','current_student4')
 assert prerequisites[W.name]['existing_report_delta_WP']['pooled']>=.0001,'Seed3 pooled warm improvement below root promotion threshold'
 completed_study(W4,'COMPLETE_FIXED_ORDINARY_WARM_SEED4_SCREEN',W4_SHA,'ordinary_warm_seed4_4','original_seed4_4')
 warm_hashes={}
 for f in (0,1):
  run=f'oct02_ow_s3h2_f{f}';rel=f'artifacts/runs/{run}/model_ep1.pt';record=workers[run]['outputs'][rel]
  assert len(record['sha256'])==64 and repo(rel).stat().st_size==record['bytes']
  checked=wr['fit_integrity'][run]
  assert checked['status']=='PASS_ORDINARY_WARM_FIXED_ENDPOINT_AND_LINEAGE' and checked['all103_args_match']
  assert checked['checkpoint']==dict(path=rel,sha256=record['sha256'],epoch=1)
  warm_hashes[str(f)]=record['sha256']
 # The fixed old native artifacts remain the incumbent, never the old strong4.
 pin(C/'prepared.json',a['reference_prepared_sha256'])
 old_report_path=C/'evaluation/report.json';pin(old_report_path);old=json.loads(old_report_path.read_text())
 assert old['status']=='COMPLETE_ONE_FIXED_NATIVE_COMPARISON'
 assert old['evaluation_source_sha256']=='9dd71d06373121176133e1cfa3bd2455eabcd40f35f3e38824908c52a88f4986'
 pin(C/'evaluation/sufficient.npz',old['sufficient_sha256'])
 for f in (0,1):
  receipt_path=C/'cache'/f'pred_student_quant_f{f}_valid.json';pin(receipt_path,a['reference_receipt_sha256'][str(f)])
  r=json.loads(receipt_path.read_text());assert r['prediction_sha256']==a['reference_prediction_sha256'][str(f)]
  assert r['native_sha256']==a['native_sha256'] and r['coefficients']==[.25]*4 and r['fold']==f and r['epoch']==1
  assert [c['run'] for c in r['child_checkpoints']]==[f'{n}_f{f}' for n in ('cont_sh256_s3h2',*a['members'][1:])]
 required=set(a['evaluation_input_sha256']);assert len(required)==6 and required<=set(tracked)
 a['evaluation_input_sha256']={rel:tracked[rel] for rel in a['evaluation_input_sha256']}
 a['warm_checkpoint_sha256']=warm_hashes
 for path in (P/'evaluate.py',Path(__file__),template,W/'evaluate.py',W4/'evaluate.py',C/'evaluate.py'):
  a['source_sha256'][str(path.relative_to(ROOT))]=pin(path)
 a.update(authorized=True,source_preparation_only=False,created_utc=datetime.now(timezone.utc).isoformat(),
  authorization='Explicit root invocation after fixed seed3 and seed4 primary prerequisites; allocates2 native OOF and1 fixed readout only.',
  activation_evidence=prerequisites,prerequisite_sha256=tracked,
  production_seed='Fixed seed3 as registered before reading seed4; no best-seed choice',allocation_builder_sha256=sha(Path(__file__)))
 a.pop('root_fill_required',None)
 for rel,h in tracked.items():assert sha(repo(rel))==h,('Prerequisite changed during allocation',rel)
 out=P/'allocation.json'
 with out.open('x') as stream:json.dump(a,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',allocation_sha256=sha(out),OOF_passes=2,metric_readouts=1,
  members=a['members'],lanes=a['lanes'],prerequisites=prerequisites)))

if __name__=='__main__':main()
