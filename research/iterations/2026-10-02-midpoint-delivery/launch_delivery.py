"""Root-owned detached finisher launch; no runtime authorization."""
from pathlib import Path
import hashlib,json,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=HERE/'finish-config.json';h=hashlib.sha256(p.read_bytes()).hexdigest();a=json.loads(p.read_text());assert a['execution_authorized']
for n in ('finish-delivery-01','finish-delivery.log','delivery-launch.json'):assert not (HERE/n).exists()
argv=['.venv/bin/python',str((HERE/'finish_delivery.py').relative_to(ROOT)),'--config',str(p.relative_to(ROOT)),'--config-sha256',h]
with (HERE/'finish-delivery.log').open('x') as f:
 process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='STARTED_NOT_COMPLETED',pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],argv=argv,config_sha256=h)
with (HERE/'delivery-launch.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
