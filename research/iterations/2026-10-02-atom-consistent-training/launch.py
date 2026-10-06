"""Start the already allocated queue detached; record process identity, no retries."""
from pathlib import Path
import hashlib,json,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=HERE/'training-allocation.json';allocation=json.loads(a.read_text());assert allocation['execution_authorized']
receipt=HERE/'launch.json';log=HERE/'queue.log'
assert not receipt.exists() and not log.exists() and not (ROOT/allocation['execution_directory']).exists()
argv=['.venv/bin/python',str((HERE/'run_queue.py').relative_to(ROOT)),'--allocation',str(a.relative_to(ROOT))]
with log.open('x') as stream:
 p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]
r=dict(pid=p.pid,start_ticks=ticks,argv=argv,allocation_sha256=sha(a),log=str(log.relative_to(ROOT)),status='STARTED_NOT_COMPLETED')
with receipt.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
