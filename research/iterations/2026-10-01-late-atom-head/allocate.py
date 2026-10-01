"""Bind the one predeclared four-fit late-head screen after vocabulary completion."""
import json,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REL=str(HERE.relative_to(ROOT))
A=ROOT/'research/iterations/2026-10-01-atom-decoder-probe';B=ROOT/'research/iterations/2026-10-01-ensemble-backbone'
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
sha=Q.sha;write=Q.write
assert json.loads((HERE/'vocab-run.json').read_text())['status']=='COMPLETE'
plan=json.loads((HERE/'commands.json').read_text());old=json.loads((A/'allocation.json').read_text())
records=[]
def bind(p,h=None,prior=None):
 p=Path(p);s=p.stat();r=dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
 if prior:
  for k in ('resolved','size','mtime_ns','inode','device'):assert r[k]==prior[k],(str(p),k)
 if h:assert sha(p)==h,str(p);r['sha256']=h
 records.append(r)
for r in json.loads((A/'input-bindings.json').read_text())['records']:
 if '/WunderFunder/artifacts/runs/ftah128_' not in r['path']:bind(r['path'],r['sha256'],r)
for r in json.loads((B/'input-bindings.json').read_text())['records']:
 if 'datasets/' in r['path']:bind(r['path'],prior=r)
 if r['path'].endswith('market-classes-v1.npz'):bind(r['path'],r['sha256'],r)
for f,h in enumerate(('1e0cfb8931e7fb351aff9903aac173887fb7fbdd085c28c90514aa6a0bb13616','d90770ba95c8801bacacd7040ae0c239f8fab8ef322c3c6ea8d31ecb10982c8e')):
 bind(ROOT/f'artifacts/runs/cont_sh256_s3h2_f{f}/model_ep1.pt',h)
for rel in plan['vocab']['required_outputs']:bind(ROOT/rel,sha(ROOT/rel))
bind(HERE/'vocab-run.json',sha(HERE/'vocab-run.json'))
assert len({r['path'] for r in records})==len(records)
write(HERE/'input-bindings.json',dict(records=records))
paths=[*HERE.glob('*.py'),HERE/'contract.md',HERE/'commands.json',ROOT/'scripts/train_rnn.py',*sorted((ROOT/'src/wnn').glob('*.py')),B/'evaluate.py',Q.HELPER]
sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
g=Q.H.gpu(0);assert g['uuid']==old['resources']['gpu_uuid'] and g['free_MiB']>=8192
assert len(plan['jobs'])==6 and sum(j['stage']=='H2' for j in plan['jobs'])==4 and sum(j['stage']=='OOF' for j in plan['jobs'])==2
now=datetime.now(timezone.utc)
a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=80)).isoformat(),execution_directory=REL+'/execution-01',success_status='FOUR_LATE_HEAD_FITS_TWO_OOF',resources=old['resources'],environment=old['environment'],sources=sources,input_bindings=[dict(path=REL+'/input-bindings.json',sha256=sha(HERE/'input-bindings.json'))],jobs=plan['jobs'],gpu_preflight=g)
write(HERE/'allocation.json',a)
print(json.dumps(dict(status='ALLOCATED_FOUR_FITS_TWO_OOF',sha256=sha(HERE/'allocation.json'),inputs=len(records),gpu=g)))
