"""Separate processes for old and new native modules: actual callback CPU audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,subprocess,tempfile,hashlib
from pathlib import Path
CHILD = r'''
import os,sys,json,time,zipfile,tempfile
from pathlib import Path
from types import SimpleNamespace as Point
import numpy as np
archive,probe,tag,output=sys.argv[1:5]
root=Path(tempfile.mkdtemp())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 z.extractall(root)
sys.path.insert(0,str(root));import solution,wnn_cpp
z=np.load(probe);x=z['x'];assert x.shape==(20000,112)
parity={}
if tag=='candidate':
 assert wnn_cpp.CLASS_HEAD_SUPPORTED
 members=json.loads((root/'members.json').read_text())
 weights=[{k:np.ascontiguousarray(v,np.float32) for k,v in np.load(root/m['weights']).items()} for m in members]
 for quant in (False,True):
  got=wnn_cpp.Ensemble(weights,[.5,.5],dynamic_int8=quant).run(x[:2000]);ref=z['reference']
  err=float(abs(got-ref).max());corr=[float(np.corrcoef(got[:,i],ref[:,i])[0,1]) for i in (0,1)]
  assert min(corr)>(.999 if quant else .9999999)
  if not quant:assert err<.001
  parity[str(quant)]={'max_abs':err,'correlation':corr}
model=solution.PredictionModel();first=[];start=time.perf_counter()
for seq in range(5):
 for i,row in enumerate(x):
  got=model.predict(Point(seq_ix=seq,step_in_seq=i,need_prediction=i>=99,state=row))
  if i<99:assert got is None
  else:
   got=np.asarray(got);assert got.shape==(2,) and np.isfinite(got).all()
   if seq==0:first.append(got.copy())
   else:assert np.array_equal(first[i-99],got)
seconds=time.perf_counter()-start
Path(output).write_text(json.dumps({'seconds':seconds,'cpu_level':solution.LEVEL,'backend':wnn_cpp.quant_backend(),'reset_bit_exact':True,'finite_shape_pass':True,'parity':parity}))
'''
def main():
 old,new,probe,output=map(Path,sys.argv[1:5]);root=Path(tempfile.mkdtemp());runs={'reference':[],'candidate':[]}
 for order in (('reference','candidate'),('candidate','reference')):
  for tag in order:
   path=root/f'{tag}-{len(runs[tag])}.json'
   subprocess.run([sys.executable,'-c',CHILD,str(old if tag=='reference' else new),str(probe),tag,str(path)],check=True)
   runs[tag].append(json.loads(path.read_text()))
 ratio=sum(x['seconds'] for x in runs['candidate'])/sum(x['seconds'] for x in runs['reference'])
 report={'runs':runs,'rows_per_run':100000,'candidate_to_reference_ratio':ratio,'public_minutes_relative_projection':ratio*24.95,'candidate_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'scope':'OOF class-head pair, not final candidate. Python3.11 independent DataSphere CPU. Separate processes load distinct old/new native engines. One full real sequence repeated5times, exact resets, FP32 and int8 class parity. Relative to user-reported24m57s; no scorer runtime certification or quality claim.'}
 output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__ == '__main__':
 main()
