"""Four saved-parent inference jobs; no training and no mutation of old campaigns."""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S=str(HERE.relative_to(ROOT));B=ROOT/'research/iterations/2026-10-01-ensemble-backbone'
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
sha=Q.sha;write=Q.write
old=json.loads((B/'training-allocation-02.json').read_text());report=json.loads((B/'evaluation-01/report.json').read_text())
assert sha(B/'evaluation-01/report.json')=='7074a125872163deeaa84184483c0f0f67a329d0eceba4b0a7ff0ba31ac10453'
assert json.loads((B/'finish-02/result.json').read_text())['status']=='COMPLETE_ONE_PAIRED_STUDY'
expected={}
for absname,h in report['provenance']['input_sha256'].items():
 p=Path(absname)
 if ('pred_on_ctl' in p.name and p.suffix in ['.npy','.json']) or p.name in ['valid_targets.npz','chain_folds.npz','scored-rows.npz'] or str(p).endswith('2026-10-01-error-analysis/extraction-01/receipt.json'):expected[p]=h
for name in ['report.json','sufficient.npz']:
 p=B/'evaluation-01'/name;expected[p]=json.loads((B/'evaluation-01/receipt.json').read_text())['outputs'][name]['sha256']
expected[B/'evaluation-01/receipt.json']=sha(B/'evaluation-01/receipt.json')
expected[B/'evaluation-bindings-01.json']=report['provenance']['bindings_sha256']
for fold in [0,1]:
 for arm in ['control','candidate']:
  name=f'eb_{arm}_s3_parent_f{fold}';directory='execution-01' if fold==0 else 'execution-02'
  wr=B/directory/f'{name}.parent.worker-result.json';w=json.loads(wr.read_text());assert w['status']=='COMPLETE'
  p=ROOT/'artifacts/runs'/name/'model_ep3.pt';expected[p]=w['outputs'][str(p.relative_to(ROOT))]['sha256'];expected[wr]=sha(wr)
records=[]
for p,h in expected.items():
 assert sha(p)==h,str(p);st=p.stat();records.append(dict(path=str(p),resolved=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,sha256=h))
write(HERE/'input-bindings.json',dict(records=records,scope='Fixed parent endpoint, old H2 sufficient statistics, targets/folds and frozen partners; all hashed before parent inference.'))
g=Q.H.gpu(0);assert g['uuid']==old['resources']['gpu_uuid'] and g['free_MiB']>=8192,g
sources=old['sources'].copy()
for name in ['run_queue.py','predict_fixed.py','allocate.py','evaluate_stages.py','contract.md']:sources[S+'/'+name]=sha(HERE/name)
# Pin the already tested numerical module and old summary statistics provenance.
sources[str((B/'evaluate.py').relative_to(ROOT))]=sha(B/'evaluate.py')
jobs=[]
for fold in [0,1]:
 for arm in ['control','candidate']:
  name=f'eb_{arm}_s3_parent_f{fold}';pred=S+'/cache/pred_'+name+'_valid.npy';rec=pred.replace('.npy','.inputs.json')
  jobs.append(dict(name=name,stage='OOF',argv=['.venv/bin/python',S+'/predict_fixed.py',name,'--fold',str(fold),'--epoch','3','--batch','16','--output-dir',S+'/cache'],timeout_seconds=180,require_absent=[pred,pred+'.partial',rec],required_outputs=[pred,rec]))
a=dict(execution_authorized=True,created_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),execution_directory=S+'/execution-01',success_status='FOUR_PARENT_OOF',resources=old['resources'],environment=old['environment'],sources=sources,input_bindings=[dict(path=S+'/input-bindings.json',sha256=sha(HERE/'input-bindings.json'))],jobs=jobs,gpu_preflight=g)
write(HERE/'allocation.json',a);print(json.dumps(dict(status='ALLOCATED_FOUR_PARENT_OOF',sha256=sha(HERE/'allocation.json'),inputs=len(records),gpu=g)))
