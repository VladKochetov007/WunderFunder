from pathlib import Path
import json,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
with (HERE/'controller.log').open('x') as f:
 p=subprocess.Popen(['.venv/bin/python',str((HERE/'run_once.py').relative_to(ROOT))],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
r=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21])
with (HERE/'launch.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
