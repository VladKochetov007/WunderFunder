"""Local-only detached terminal preservation, no training, scoring or retries."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
assert str(ROOT)=='/Users/nb/Documents/wunder/WunderFunder-creative'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=HERE/'training-allocation.json';allocation=json.loads(a.read_text());assert allocation['execution_authorized']
receipt=HERE/'collector-launch.json';log=HERE/'collector.log'
assert not receipt.exists() and not receipt.is_symlink() and not log.exists() and not log.is_symlink()
argv=[sys.executable,str(HERE/'collect_when_finished.py'),'--campaign',str(HERE.relative_to(ROOT))]
with log.open('x') as stream:
 p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,argv=argv,allocation_sha256=sha(a),collector_sha256=sha(HERE/'collect_when_finished.py'),log=str(log.relative_to(ROOT)),status='STARTED_NOT_COMPLETED')
with receipt.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
