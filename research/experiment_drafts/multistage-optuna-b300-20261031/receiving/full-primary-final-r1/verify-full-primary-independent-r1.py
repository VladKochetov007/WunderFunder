from pathlib import Path
import importlib.util,json,time
import numpy as np
H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1')
source=H/'requests/verify-primary-independent-r1.py'
spec=importlib.util.spec_from_file_location('independent_primary',source)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
assert v.sha256(source)=='5df00bb5e209ba723b32d04ba8c8748380b3207d07c4108f35d0543f572a9963'
p=H/'reports/full-primary-refits-r1/full_primary_refit_report.json';digest=v.sha256(p);r=v._read(p)
assert r['status']=='complete' and r['primary_valid_used_for_weights'] is False
manifest=Path(r['primary_split']['path']);assert v.sha256(manifest)==v.PRIMARY
q=v._read(Path(r['score_request']['path']));assert v.sha256(Path(r['score_request']['path']))==r['score_request']['sha256']
targets,mask,raw=v.read_primary_labels(Path(q['data_root']),v._read(manifest))
arrays=[];results=[];bindings={str(p):digest,str(source):v.sha256(source),str(manifest):v.sha256(manifest)}
assert len(r['singleton_scores'])==4
for member in r['singleton_scores']:
 s=member['score'];receipt=s['predictions']
 assert s['status']=='complete' and s['primary_split_sha256']==v.PRIMARY
 assert all(receipt[k]==mask[k] for k in v.MASK_KEYS)
 a=v._array(Path(receipt['path']),receipt['sha256']);arrays.append(a)
 m=v.independent_metrics(targets,a);err=v._compare(m,s['metrics'])
 results.append({'name':s['fit_id'],'metrics':m,'maximum_metric_absolute_error':err})
 bindings[receipt['path']]=receipt['sha256']
for name,s in r['fixed_ensemble_scores']['ensembles'].items():
 receipt=s['predictions'];assert all(receipt[k]==mask[k] for k in v.MASK_KEYS)
 a=v._array(Path(receipt['path']),receipt['sha256']);w=np.asarray(s['weights'],dtype=np.float64)
 assert list(s['weights'])==r['weights'][name]
 for start in range(0,len(a),65536):
  end=min(start+65536,len(a));expected=v.reconstruct_block([x[start:end] for x in arrays],w)
  assert np.array_equal(expected,a[start:end]),'mixture reconstruction mismatch'
 m=v.independent_metrics(targets,a);err=v._compare(m,s['metrics'])
 results.append({'name':name,'metrics':m,'maximum_metric_absolute_error':err,'fp32_reconstruction_bit_exact':True})
 bindings[receipt['path']]=receipt['sha256']
assert v.sha256(p)==digest and len(results)==6
out=H/'reports/independent-full-primary-r1.json'
result={'schema':'wnn.independent-full-primary-refit-arithmetic.v1','status':'verified','created_at_epoch':time.time(),'claim_scope':'six fixed prediction arrays: raw-label/mask and arithmetic only; not independent training replication or public transfer','report_sha256':digest,'tolerance':1e-10,'rows':len(targets),'sequences':1873,'raw_validation_source':raw,'mask_and_order':mask,'file_sha256':bindings,'outputs':results}
with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
print('VERIFIED',len(results),'MAX_ERROR',max(x['maximum_metric_absolute_error'] for x in results),flush=True)
