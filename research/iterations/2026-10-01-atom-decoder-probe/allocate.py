"""Two frozen atom-head counterfactual prediction passes, no fitting."""
import json,importlib.util,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];A=str(HERE.relative_to(ROOT));S=ROOT/'research/iterations/2026-10-01-stage-diagnosis';B=ROOT/'research/iterations/2026-10-01-ensemble-backbone'
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
sha=Q.sha;write=Q.write
old=json.loads((S/'allocation.json').read_text());previous=json.loads((S/'input-bindings.json').read_text())
expected={}
for r in previous['records']:
 p=Path(r['path'])
 if 'pred_on_ctl' in p.name or p.name in ['valid_targets.npz','chain_folds.npz','scored-rows.npz'] or p.parent==B/'evaluation-01' or str(p).endswith('2026-10-01-error-analysis/extraction-01/receipt.json'):expected[p]=r['sha256']
checkpoints={0:('/home/grinchenkoeg/WunderFunder/artifacts/runs/ftah128_f0h/model_ep0.pt','6781e2a7f86f9195ee7abf6c3dbce735b1cfafcfbfc21d26219f0c5e065b3e7d'),1:('/home/grinchenkoeg/WunderFunder/artifacts/runs/ftah128_f1h/model_ep0.pt','4333c514858c68cb3bf10a2f88f6f4739d7c80f8c907adca3085a49f43516710')}
for p,h in checkpoints.values():expected[Path(p)]=h
for f,(p,h) in checkpoints.items():
 dest=HERE/'source-checkpoints'/f'fold{f}'/'model_ep0.pt';dest.parent.mkdir(parents=True);assert not dest.exists();shutil.copy2(p,dest);assert sha(dest)==h
records=[]
for p,h in expected.items():
 assert sha(p)==h,str(p);st=p.stat();records.append(dict(path=str(p),resolved=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,sha256=h))
write(HERE/'input-bindings.json',dict(records=records))
sources=old['sources'].copy()
for name in ['run_queue.py','predict_variants.py','evaluate.py','allocate.py','contract.md']:sources[A+'/'+name]=sha(HERE/name)
sources['src/wnn/atom_head.py']=sha(ROOT/'src/wnn/atom_head.py')
jobs=[]
for f,(cp,h) in checkpoints.items():
 pred=f'{A}/cache/fold{f}.npz';receipt=f'{A}/cache/fold{f}.json'
 jobs.append(dict(name=f'atom_f{f}',stage='OOF',argv=['.venv/bin/python',A+'/predict_variants.py','--fold',str(f),'--checkpoint',cp,'--checkpoint-sha256',h,'--output-dir',A+'/cache'],timeout_seconds=300,require_absent=[pred,receipt],required_outputs=[pred,receipt]))
g=Q.H.gpu(0);assert g['uuid']==old['resources']['gpu_uuid'] and g['free_MiB']>=8192
now=datetime.now(timezone.utc);a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=30)).isoformat(),execution_directory=A+'/execution-01',success_status='TWO_ATOM_VARIANT_PASSES',resources=old['resources'],environment=old['environment'],sources=sources,input_bindings=[dict(path=A+'/input-bindings.json',sha256=sha(HERE/'input-bindings.json'))],jobs=jobs,gpu_preflight=g)
write(HERE/'allocation.json',a);print(json.dumps(dict(status='ALLOCATED_TWO_ATOM_VARIANT_PASSES',sha256=sha(HERE/'allocation.json'),inputs=len(records),gpu=g)))
