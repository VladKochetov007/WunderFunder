import datetime as dt,json,subprocess,os
from pathlib import Path
b=Path('/home/grinchenkoeg/computer-vision-homework-20261018');unit='computer-vision-homework-fg19-propensity-f0-e4-v3.service';inv='85b87a83f96945f4b78908f124435a2b'
def show(name):
 keys=['LoadState','ActiveState','SubState','InvocationID','MainPID','ControlGroup','CPUUsageNSec','MemoryCurrent','MemoryPeak','ExecMainCode','ExecMainStatus','CPUQuotaPerSecUSec','MemoryHigh','MemoryMax','MemorySwapMax']
 p=subprocess.run(['systemctl','--user','show','--all',name,*sum((['-p',k] for k in keys),[])],capture_output=True,text=True,timeout=10)
 if p.returncode:raise RuntimeError(p.stderr)
 return dict(x.split('=',1) for x in p.stdout.splitlines())
def cgfiles(group):
 if not group:return None
 p=Path('/sys/fs/cgroup')/group.lstrip('/');return {n:(p/n).read_text() if (p/n).exists() else None for n in ['memory.current','memory.events','memory.events.local','cgroup.procs','cpu.stat']}
u=show(unit);assert u['InvocationID']==inv,(u,inv)
parent=show('wnn-campaign.slice');pg=Path('/sys/fs/cgroup')/parent['ControlGroup'].lstrip('/');files=[pg/'cgroup.procs',*pg.rglob('cgroup.procs')];assert len(files)<128
populated={str(p):p.read_text() for p in files if p.read_text().strip()}
log=b/'research_tmp/fg19_propensity_clean_f0_admission_v3/unit.log'
with log.open('rb') as f:f.seek(max(0,log.stat().st_size-16000));tail=f.read().decode(errors='replace')
run=b/'data/artifacts/runs/fg19_propensity_clean_f0_seed0_e4';names=['provenance.json','training_state.pt','model_ep0.pt','model_ep1.pt','model_ep2.pt','model_ep3.pt','training_progress.json','training_journal.jsonl','progress.json']
stats={n:({'bytes':(run/n).stat().st_size,'mtime_ns':(run/n).stat().st_mtime_ns} if (run/n).exists() else None) for n in names}
print(json.dumps(dict(utc=dt.datetime.now(dt.UTC).isoformat(),unit=u,own_files=cgfiles(u['ControlGroup']),parent=parent,parent_files=cgfiles(parent['ControlGroup']),populated=populated,log_tail=tail,run_exists=run.exists(),outputs_stat_only=stats)))
