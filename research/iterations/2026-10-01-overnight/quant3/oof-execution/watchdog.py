"""Hard process-group deadline for already registered OOF lanes, including grandchildren."""
from pathlib import Path
import datetime,json,os,resource,signal,time
D=Path(__file__).resolve().parent
os.sched_setaffinity(0,{36})
resource.setrlimit(resource.RLIMIT_AS,(128<<20,128<<20))
jobs=[json.loads(x) for x in (D/'execution-ledger.jsonl').read_text().splitlines() if json.loads(x)['event']=='lane_start']
assert len(jobs)==2
for j in jobs:
 j['deadline']=datetime.datetime.fromisoformat(j['utc']).timestamp()+3600
 assert os.getpgid(j['pid'])==j['pid']
with (D/'watchdog.jsonl').open('x') as log:
 log.write(json.dumps({'event':'start','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lanes':jobs,'scope':'registered lane process groups only','changed_producer_source':False})+'\n');log.flush()
 while jobs:
  for j in jobs[:]:
   try:os.killpg(j['pid'],0)
   except ProcessLookupError:
    jobs.remove(j);log.write(json.dumps({'event':'group_exited','fold':j['fold'],'pid':j['pid'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n');log.flush();continue
   if time.time()>=j['deadline']:
    os.killpg(j['pid'],signal.SIGKILL);jobs.remove(j)
    log.write(json.dumps({'event':'hard_deadline_kill','fold':j['fold'],'pid':j['pid'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n');log.flush()
  if jobs:time.sleep(5)
