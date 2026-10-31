from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
Q=Path('/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1');H=Path('/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1');sys.path.insert(0,str(H/'source6-eval/src'));sys.path.append(str(H/'cpu-deps-r1'))
from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import _centered_moments
p=H/'requests/verify-primary-independent-r1.py';s=importlib.util.spec_from_file_location('raw_primary_verifier',p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
manifest=json.loads((H/'data/chronological_primary.json').read_text());target,mask,source=v.read_primary_labels(H/'data',manifest);report=json.loads((Q/'reports/fixed-tl3-mixtures-r1/REPORT.json').read_text());counts=[row['selected_rows'] for row in mask['per_sequence_selected_rows']];mapping={row['row_group']:row['component'] for row in manifest['sequences']['valid']};groups=np.asarray([mapping[g] for g in manifest['evaluation']['valid']]);offsets=np.cumsum([0]+counts);reference=next(row for row in report['records'] if row['name']=='replayed_TL3');baseline=v._array(Path(reference['prediction']['path']),reference['prediction']['sha256'])
def moments(prediction:np.ndarray)->np.ndarray:
 result=np.zeros((1873,2,6),dtype=np.float64)
 for i in range(1873):
  a,b=offsets[i:i+2];y=np.clip(target[a:b],-2,2).astype(np.float64);p=np.clip(prediction[a:b],-2,2).astype(np.float64)
  for k in range(2):result[i,k]=_centered_moments(y[:,k],p[:,k])
 return result
bm=moments(baseline);results=[]
for row in report['records']:
 if row['name']=='replayed_TL3':continue
 pred=v._array(Path(row['prediction']['path']),row['prediction']['sha256']);result=paired_component_bootstrap(moments(pred),bm,groups,n_boot=600,seed=20261031,centered=True);assert abs(result['candidate_wp']-row['metrics']['wp'])<1e-10 and abs(result['reference_wp']-reference['metrics']['wp'])<1e-10;result.update(name=row['name'],diagnostic_only=True);results.append(result);print(row['name'],result['mean_diff'],result['ci95'],flush=True)
out=Q/'reports/fixed-mixture-sensitivity-r1';out.mkdir(exist_ok=False);payload={'schema':'wnn.fixed-mixture-conditional-sensitivity.v1','status':'complete','created_epoch':time.time(),'source_report_sha256':v.sha256(Q/'reports/fixed-tl3-mixtures-r1/REPORT.json'),'primary_split_sha256':v.PRIMARY,'grouping':'provided feature-successor connected components from frozen official manifest; no inferred temporal order','results':results,'new_fits':0,'training_variance_estimated':False,'selection_adjusted':False,'public_transfer_established':False,'interpretation':'Post-hoc descriptive uncertainty of fixed predictions on development-exposed valid after six tested mixtures; not an independent confirmation or calibrated significance test. p_le_0 is a conditional resampling fraction, not a calibrated p-value.'};(out/'REPORT.json').write_text(json.dumps(payload,sort_keys=True,indent=2))
print('SENSITIVITY_COMPLETE',len(results),flush=True)
