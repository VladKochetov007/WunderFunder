"""Root-only start of an already frozen allocation; no automatic activation."""
from pathlib import Path
import hashlib,json,os,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
allocation=P/'allocation.json';a=json.loads(allocation.read_text());assert a['authorized'] is True
for path,digest in a['source_sha256'].items():assert sha(ROOT/path)==digest,path
for name in ('supervisor.json','supervisor-result.json','controller.json','controller.log','prepared.json','fold0','fold1','cache','evaluation'):
 path=P/name;assert not path.exists() and not path.is_symlink(),path
argv=[str(ROOT/'.venv/bin/python'),str(P/'launch.py')]
with (P/'controller.log').open('x') as stream:
 p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='STARTED_NOT_COMPLETED',pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],argv=argv,allocation_sha256=sha(allocation))
with (P/'controller.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
