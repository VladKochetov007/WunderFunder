from pathlib import Path
import base64,datetime,hashlib,json,os,stat,subprocess
D=Path('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_allthree_verify_v1')
s=D.lstat()
if not stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_uid!=1061:raise RuntimeError('exact owned support directory differs')
def read(name):
 p=D/name;b=p.lstat()
 if not stat.S_ISREG(b.st_mode) or b.st_uid!=1061 or b.st_size<=0 or b.st_size>1000000:raise RuntimeError('unsafe receipt')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  o=os.fstat(fd)
  if (o.st_dev,o.st_ino,o.st_size,o.st_mtime_ns)!=(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns):raise RuntimeError('receipt changed before read')
  chunks=[];total=0
  while True:
   q=os.read(fd,min(65536,1000001-total))
   if not q:break
   chunks.append(q);total+=len(q)
   if total>1000000:raise RuntimeError('receipt cap exceeded')
  a=os.fstat(fd)
 finally:os.close(fd)
 if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns):raise RuntimeError('receipt changed after read')
 raw=b''.join(chunks)
 return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
files={n:read(n) for n in ('VERIFIER_ADMISSION.json','VERIFIER_LAUNCH.json')}
if sum(i['bytes'] for i in files.values())>2000000:raise RuntimeError('aggregate cap exceeded')
unit='computer-vision-homework-fg19-child-allthree-verify-v1.service';K=('LoadState','ActiveState','SubState','MainPID','InvocationID','Result','ExecMainCode','ExecMainStatus','ControlGroup','MemoryCurrent','MemoryPeak','CPUUsageNSec','ExecMainStartTimestamp','ExecMainExitTimestamp')
r=subprocess.run(['systemctl','--user','show','--all',unit,*sum((['-p',k] for k in K),[])],capture_output=True,timeout=15)
if len(r.stdout)+len(r.stderr)>131072 or r.returncode!=0:raise RuntimeError('bounded exact unit query failed')
x={'schema':'fg19.child-allthree-root-empty-reply-receipt-recovery.v1','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'support_unit':unit,'unit_properties':dict(line.split('=',1) for line in r.stdout.decode().splitlines() if '=' in line),'files':files,'scope':'One readonly exact support-unit and two durable-receipt recovery after SSH0 empty reply; no admission/retry/restart/stop/source/model/state/data/cache/dictionary payload read/hash.'}
print(json.dumps(x,sort_keys=True))
