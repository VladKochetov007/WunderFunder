"""Synthetic predictions only; exact frozen error aggregation, no timing."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
out=P/'synthetic-cross-isa.json';assert not out.exists()
reference=np.load(P/'synthetic-v3.pred.npy')[99:]
reports={}
for level in ['v2','avx2']:
 candidate=np.load(P/f'synthetic-{level}.pred.npy')[99:]
 assert np.isfinite(candidate).all();delta=candidate.astype(np.float64)-reference.astype(np.float64);target=[]
 for k in range(2):
  error=np.abs(delta[:,k]);rmse=float(np.sqrt(np.mean(delta[:,k]**2)));std=float(np.std(reference[:,k].astype(np.float64)));r={'target':k,'rmse':rmse,'reference_population_std':std,'relative_rmse':rmse/max(std,1e-8),'p99_abs':float(np.quantile(error,.99)),'max_abs':float(error.max())}
  assert r['relative_rmse']<=.03 and r['p99_abs']<=.05 and r['max_abs']<=.5;target.append(r)
 reports[level]={'per_target':target,'bitexact':candidate.tobytes()==reference.tobytes(),'candidate_sha256':hashlib.sha256(candidate.tobytes()).hexdigest()}
result={'status':'PASS','synthetic_streams':1,'postwarmup_rows':len(reference),'pooled_equals_single_stream':True,'reference':'v3/VNNI','reports':reports,'quality_runtime_qualification':'none from synthetic test; different-ISA WP unqualified','competition_rows_read':False,'targets_read':False,'WP_queries':0}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
