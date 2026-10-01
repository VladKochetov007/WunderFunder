"""Materialize the already authorized v2 fixed readout after all six job steps pass."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
a=json.loads((HERE/'allocation-01.json').read_text());inspection=json.loads((HERE/'fit-inspection-recovery.json').read_text())
assert a['execution_authorized'] and a['score_execution_authorized']
assert inspection['status']=='PASS_SEED4_CV4_AND_OOF2_WITH_EXPLICIT_INFRASTRUCTURE_RECOVERY'
assert inspection['allocation_sha256']==sha(HERE/'allocation-01.json')
result=json.loads((HERE/'execution-01/result.json').read_text())
assert result['status']=='FAILED_STOP_NO_RETRY' and inspection['root_result_sha256']==sha(HERE/'execution-01/result.json')
recovery=json.loads((HERE/'recovery-01/result.json').read_text())
assert recovery['status']=='PASS_INFRASTRUCTURE_RECOVERY_H2_F1_OOF_F1_NO_METRICS'
assert inspection['recovery_result_sha256']==sha(HERE/'recovery-01/result.json')
assert inspection['recovery_allocation_sha256']==sha(HERE/'recovery-allocation-01.json')
assert not (HERE/'evaluation-01').exists() and not (HERE/'evaluation-v2-01').exists()
assert sha(HERE/'amendment-02.json')=='45d895efaf2c55e053c04037adc8033f1c0d21c4f3ad0fdd32b5747e8c08929b'
for name,digest in json.loads((HERE/'amendment-02.json').read_text())['original_preserved_source_sha256'].items():assert sha(HERE/name)==digest
b=json.loads((HERE/'evaluation-spec.v2.json').read_text())
b['status']='ROOT_ALLOCATED_FOR_FIXED_FOUR_COMPARISONS'
b['root_allocation']={'path':str((HERE/'allocation-01.json').relative_to(ROOT)),'sha256':sha(HERE/'allocation-01.json'),'required_field':'score_execution_authorized','required_value':True}
for name,rec in b['predictions'].items():
 meta=json.loads((ROOT/rec['receipt']).read_text())
 assert meta['run']==name and meta['epoch']==1 and meta['metrics_computed']==0
 ck=ROOT/'artifacts/runs'/name/'model_ep1.pt'
 assert Path(meta['checkpoint'])==ck and meta['checkpoint_sha256']==sha(ck)
 assert Path(meta['prediction_path'])==ROOT/rec['path'] and (ROOT/rec['path']).is_file()
 rec.update(sha256=meta['prediction_sha256'],receipt_sha256=sha(ROOT/rec['receipt']),checkpoint_sha256=meta['checkpoint_sha256'])
b['source_sha256']=sha(Path(__file__))
b['completed_fit_inspection_sha256']=sha(HERE/'fit-inspection-recovery.json')
b['completed_run_result_sha256']=sha(HERE/'execution-01/result.json')
b['completed_run_original_status']=result['status']
b['infrastructure_recovery']={'status':recovery['status'],'result_sha256':sha(HERE/'recovery-01/result.json'),'allocation_sha256':sha(HERE/'recovery-allocation-01.json'),'original_failed_H2_attempt_preserved':True,'original_successful_steps':4,'recovery_successful_steps':2}
for rel in ('execution-01/result.json','recovery-01/result.json','recovery-allocation-01.json','run_recovery.py','fit-inspection-recovery.json','inspect_recovered.py','bind_recovered_readout_v2.py'):
 p=HERE/rel;b['sources'][str(p.relative_to(ROOT))]=sha(p)
with (HERE/'bindings-v2-01.json').open('x') as f:json.dump(b,f,indent=2);f.write('\n')
print(json.dumps({'bindings_sha256':sha(HERE/'bindings-v2-01.json'),'actual_metric_computations':0,'prediction_arrays_hashed_by_evaluator_before_and_after_readout':True}))
