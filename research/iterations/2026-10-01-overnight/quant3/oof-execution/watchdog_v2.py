"""Versioned orchestration guard; no producer, candidate, input or output changes."""
from pathlib import Path
import datetime,hashlib,json,os,resource,signal,time
D=Path(__file__).resolve().parent
os.sched_setaffinity(0,{36})
resource.setrlimit(resource.RLIMIT_AS,(128<<20,128<<20))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def proc(pid):
 try:
  t=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'pgid':int(t[2]),'start_ticks':int(t[19]),'rss_bytes':max(0,int(t[21]))*os.sysconf('SC_PAGE_SIZE')}
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None
jobs=[json.loads(x) for x in (D/'execution-ledger.jsonl').read_text().splitlines() if json.loads(x)['event']=='lane_start']
assert len(jobs)==2
for j in jobs:
 leader=proc(j['pid']);assert leader and leader['pgid']==j['pid']
 j['leader_start_ticks']=leader['start_ticks'];j['deadline_utc_epoch']=datetime.datetime.fromisoformat(j['utc']).timestamp()+3600
 j['deadline_monotonic']=time.monotonic()+max(0,j['deadline_utc_epoch']-time.time())
 j['seen']={};j['max_observed_aggregate_rss_bytes']=0
old=proc(4084613)
assert old and old['start_ticks']==499064064
cmd=(Path('/proc/4084613/cmdline')).read_bytes().split(b'\0')
assert any(x.endswith(b'/oof-execution/watchdog.py') for x in cmd)
with (D/'watchdog-v2.jsonl').open('x') as log:
 def record(v):log.write(json.dumps(v)+'\n');log.flush()
 record({'event':'start','utc':now(),'lanes':jobs,'old_guard':old,'old_guard_replaced_reason':'identity-bound PGIDs; aggregate RSS observations; sleep bounded by exact remaining deadline','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 os.kill(4084613,signal.SIGTERM)
 while jobs:
  members={j['pid']:[] for j in jobs}
  for path in Path('/proc').iterdir():
   if not path.name.isdigit():continue
   p=proc(int(path.name))
   if p and p['pgid'] in members:members[p['pgid']].append(p)
  for j in jobs[:]:
   group=members[j['pid']];leader=proc(j['pid'])
   if not group:
    record({'event':'group_exited','utc':now(),'fold':j['fold'],'max_observed_aggregate_rss_bytes':j['max_observed_aggregate_rss_bytes']});jobs.remove(j);continue
   if leader:
    identity_ok=leader['start_ticks']==j['leader_start_ticks'] and leader['pgid']==j['pid']
   else:
    identity_ok=all(j['seen'].get(str(p['pid']))==p['start_ticks'] for p in group)
   if not identity_ok:
    record({'event':'IDENTITY_MISMATCH_NO_SIGNAL','utc':now(),'fold':j['fold'],'group':group});jobs.remove(j);continue
   j['seen'].update({str(p['pid']):p['start_ticks'] for p in group})
   rss=sum(p['rss_bytes'] for p in group);j['max_observed_aggregate_rss_bytes']=max(rss,j['max_observed_aggregate_rss_bytes'])
   record({'event':'sample','utc':now(),'fold':j['fold'],'aggregate_rss_bytes':rss,'members':group})
   reason='deadline' if time.monotonic()>=j['deadline_monotonic'] else 'aggregate_RSS_above_8GiB' if rss>8*(1<<30) else None
   if reason:
    os.killpg(j['pid'],signal.SIGKILL);record({'event':'guard_kill','reason':reason,'utc':now(),'fold':j['fold'],'leader_start_ticks':j['leader_start_ticks']});jobs.remove(j)
  if jobs:time.sleep(min(5,max(0,min(j['deadline_monotonic'] for j in jobs)-time.monotonic())))
