"""Two inference-only checks of the saved seed4 parent endpoint."""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S=str(HERE.relative_to(ROOT));C=ROOT/'research/iterations/2026-10-01-continuation/student-seed4'
spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
sha=Q.sha;write=Q.write
a=json.loads((HERE/'allocation.json').read_text());assert json.loads((HERE/'finish-result.json').read_text())['status']=='COMPLETE_FIXED_STAGE_DIAGNOSIS'
expected={}
localpins=json.loads((HERE/'seed4-existing-inputs.json').read_text())
for rel,h in localpins.items():expected[ROOT/rel]=h
inspection=json.loads((C/'fit-inspection-recovery.json').read_text());assert inspection['status']=='PASS_SEED4_CV4_AND_OOF2_WITH_EXPLICIT_INFRASTRUCTURE_RECOVERY'
for f in [0,1]:
 p=ROOT/f'artifacts/runs/cont_sh256_s4_f{f}/model_ep3.pt';expected[p]=inspection['all_new_run_files'][str(p.relative_to(ROOT))]['sha256']
receipt=json.loads((HERE/'evaluation-01/receipt.json').read_text())
for name in ['sufficient.npz','report.json']:expected[HERE/'evaluation-01'/name]=receipt['outputs'][name]['sha256']
records=[]
for p,h in expected.items():
 assert sha(p)==h,str(p);st=p.stat();records.append(dict(path=str(p),resolved=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev,sha256=h))
write(HERE/'input-bindings-02.json',dict(records=records))
a['input_bindings'].append(dict(path=S+'/input-bindings-02.json',sha256=sha(HERE/'input-bindings-02.json')))
for name in ['allocate_seed4.py','evaluate_seed4_parent.py','seed4-parent-contract.md','seed4-existing-inputs.json']:a['sources'][S+'/'+name]=sha(HERE/name)
a['jobs']=[]
for f in [0,1]:
 name=f'cont_sh256_s4_f{f}';p=S+'/cache/pred_'+name+'_valid.npy';r=p.replace('.npy','.inputs.json')
 a['jobs'].append(dict(name=name,stage='OOF',argv=['.venv/bin/python',S+'/predict_fixed.py',name,'--fold',str(f),'--epoch','3','--batch','16','--output-dir',S+'/cache'],timeout_seconds=180,require_absent=[p,p+'.partial',r],required_outputs=[p,r]))
g=Q.H.gpu(0);assert g['uuid']==a['resources']['gpu_uuid'] and g['free_MiB']>=8192
now=datetime.now(timezone.utc);a.update(created_utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=20)).isoformat(),execution_directory=S+'/execution-02',success_status='TWO_SEED4_PARENT_OOF',gpu_preflight=g)
write(HERE/'allocation-02.json',a);print(json.dumps(dict(status='ALLOCATED_TWO_SEED4_PARENT_OOF',sha256=sha(HERE/'allocation-02.json'),gpu=g)))
