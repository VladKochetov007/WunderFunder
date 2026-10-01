"""One detached launch of the already explicitly root-authorized final2 runner."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((HERE/'allocation-01.json').read_text())
assert a['execution_authorized'] is True and a['maximum_fits']==2 and a['OOF_authorized'] is False
assert sha(HERE/'run_plan.py')==a['runner_sha256'] and sha(HERE/'commands.json')==a['commands_sha256']
assert sha(ROOT/a['root_decision'])==a['root_decision_sha256']
for name,digest in json.loads((ROOT/a['root_decision']).read_text())['bindings'].items():assert sha(ROOT/name)==digest,(name,'binding changed')
assert not (HERE/'supervisor.json').exists() and not (HERE/'execution-01').exists()
cmd=[str(ROOT/'.venv/bin/python'),str(HERE/'run_plan.py'),'--execute','--allocation',str(HERE/'allocation-01.json')]
subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'run_plan.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
info=subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
total,used,free=map(int,info.strip().split(','))
assert total==a['GPU_total_MiB'] and used<=a['initial_max_used_MiB'] and free>=a['minimum_free_MiB']
with (HERE/'supervisor.log').open('x') as f:
    proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
stat=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
r={'utc':datetime.now(timezone.utc).isoformat(),'PID':proc.pid,'start_ticks':stat[19],'argv':cmd,'allocation_sha256':sha(HERE/'allocation-01.json'),'launch_source_sha256':sha(Path(__file__)),'GPU_before':{'total_MiB':total,'used_MiB':used,'free_MiB':free},'status':'SUPERVISOR_STARTED_NOT_FIT_COMPLETION'}
with (HERE/'supervisor.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
