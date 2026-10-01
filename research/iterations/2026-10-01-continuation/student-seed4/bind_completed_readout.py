"""Materialize the already authorized fixed readout after all six job steps pass."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
a=json.loads((HERE/'allocation-01.json').read_text());inspection=json.loads((HERE/'fit-inspection.json').read_text())
assert a['execution_authorized'] and a['score_execution_authorized']
assert inspection['status']=='PASS_SEED4_CV4_FIXED_ENDPOINTS_AND_H2OOF2_LINEAGE'
assert inspection['allocation_sha256']==sha(HERE/'allocation-01.json')
result=json.loads((HERE/'execution-01/result.json').read_text())
assert result['status']=='COMPLETED_4_FITS_2_OOF_NO_METRICS' and inspection['root_result_sha256']==sha(HERE/'execution-01/result.json')
b=json.loads((HERE/'evaluation-spec.json').read_text())
b['status']='ROOT_ALLOCATED_FOR_FIXED_THREE_COMPARISONS'
b['root_allocation']={'path':str((HERE/'allocation-01.json').relative_to(ROOT)),'sha256':sha(HERE/'allocation-01.json'),'required_field':'score_execution_authorized','required_value':True}
for name,rec in b['predictions'].items():
 meta=json.loads((ROOT/rec['receipt']).read_text())
 assert meta['run']==name and meta['epoch']==1 and meta['metrics_computed']==0
 ck=ROOT/'artifacts/runs'/name/'model_ep1.pt'
 assert Path(meta['checkpoint'])==ck and meta['checkpoint_sha256']==sha(ck)
 assert Path(meta['prediction_path'])==ROOT/rec['path'] and (ROOT/rec['path']).is_file()
 rec.update(sha256=meta['prediction_sha256'],receipt_sha256=sha(ROOT/rec['receipt']),checkpoint_sha256=meta['checkpoint_sha256'])
b['source_sha256']=sha(Path(__file__))
b['completed_fit_inspection_sha256']=sha(HERE/'fit-inspection.json')
b['completed_run_result_sha256']=sha(HERE/'execution-01/result.json')
with (HERE/'bindings-01.json').open('x') as f:json.dump(b,f,indent=2);f.write('\n')
print(json.dumps({'bindings_sha256':sha(HERE/'bindings-01.json'),'actual_metric_computations':0,'prediction_arrays_hashed_by_evaluator_before_and_after_readout':True}))
