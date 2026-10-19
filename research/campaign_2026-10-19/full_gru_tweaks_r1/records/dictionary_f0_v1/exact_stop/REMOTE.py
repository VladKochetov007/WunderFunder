import json,subprocess,datetime
from pathlib import Path
unit="computer-vision-homework-fg19-dictionary-f0-v1.service"
inv="4e6d90ff7d4344df92d61d0d1c3f01fb"
def show(u):
 p=subprocess.run(["systemctl","--user","show","--all",u,*sum((["-p",x] for x in ["InvocationID","ActiveState","SubState","ExecMainCode","ExecMainStatus","ControlGroup","LoadState","MemoryCurrent"]),[])],capture_output=True,text=True,timeout=15);assert p.returncode==0;return dict(a.split("=",1) for a in p.stdout.splitlines())
u=show(unit);assert u["InvocationID"]==inv and u["SubState"]=="exited" and int(u["ExecMainCode"])==1 and int(u["ExecMainStatus"])==0
q=show("wnn-campaign.slice");assert q["ControlGroup"] and q["ControlGroup"]!="/"
g=Path("/sys/fs/cgroup")/q["ControlGroup"].lstrip("/")
paths=[g/"cgroup.procs",*g.rglob("cgroup.procs")];assert len(paths)<128
parent={n:(g/n).read_text() for n in ["memory.current","memory.events","memory.events.local"]};pop={str(a):a.read_text() for a in paths if a.read_text().strip()}
r=subprocess.run(["systemctl","--user","stop",unit],capture_output=True,text=True,timeout=30)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.UTC).isoformat(),before=u,parent=parent,populated=pop,stop_returncode=r.returncode,stop_stdout=r.stdout,stop_stderr=r.stderr,after=show(unit))))
