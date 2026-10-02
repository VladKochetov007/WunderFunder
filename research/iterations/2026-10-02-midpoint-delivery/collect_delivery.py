"""Local once-only lossless collection after closed delivery phases. Never runs a model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,hashlib,json,shlex,shutil,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REL=str(HERE.relative_to(ROOT));REMOTE='/home/grinchenkoeg/WunderFunder-creative';FLOOR=3*2**30
sys.path.insert(0,'/tmp/wunder-ssh-client')
import paramiko

def utc():return datetime.now(timezone.utc)
def digest(path,compressed=False):
 h=hashlib.sha256();n=0
 with (gzip.open(path,'rb') if compressed else Path(path).open('rb')) as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b)
 return n,h.hexdigest()
def connect():
 c=paramiko.SSHClient();c.load_host_keys('/Users/nb/.ssh/known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
 c.connect('185.185.59.19',username='grinchenkoeg',key_filename='/Users/nb/.ssh/beleriand_grinchenkoeg_ed25519',look_for_keys=False,allow_agent=False,timeout=10);c.get_transport().set_keepalive(30);return c
def command(c,code,cpu,seconds):
 cmd='cd '+shlex.quote(REMOTE)+' && taskset -c '+str(cpu)+' nice -n 10 timeout --signal=TERM --kill-after=5s '+str(seconds)+'s python3 -c '+shlex.quote(code)
 i,o,e=c.exec_command(cmd,timeout=seconds+15);i.close();body=o.read();err=e.read();assert o.channel.recv_exit_status()==0,err.decode();return json.loads(body)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['ready','runtime'],required=True);ap.add_argument('--deadline-utc',required=True);ap.add_argument('--remote-cpu',type=int,required=True);a=ap.parse_args()
 assert str(ROOT)=='/Users/nb/Documents/wunder/WunderFunder-creative';deadline=datetime.fromisoformat(a.deadline_utc);assert deadline.tzinfo is not None and utc()<deadline
 target=HERE/('preservation-'+a.phase+'.json');assert not target.exists();records=[];c=None;rec=dict(status='FAILED_PRESERVATION_NO_RETRY',phase=a.phase,started_utc=utc().isoformat(),records=records)
 readycode='''from pathlib import Path
import json,subprocess
p=Path(%r);phase=%r;missing=[];live=[];results={}
paths=['execution-01/result.json','native-quality/supervisor-result.json','finish-delivery-01/result.json']
if phase=='runtime':paths.append('runtime/execution/launch-result.json')
if phase=='ready' and (p/'runtime/supervisor.json').exists():raise AssertionError('runtime already started; use explicit runtime phase after completion')
for name in paths:
 q=p/name
 if not q.exists():missing.append(name);continue
 r=json.loads(q.read_text());results[name]=r
 for j in r.get('jobs',[]):assert not j.get('remaining_scope_pids') and not j.get('cleanup_error')
 for j in r.get('stages',[]):
  if j.get('unit'):
   z=subprocess.run(['systemctl','--user','show',j['unit']+'.scope','--property=ActiveState','--value'],capture_output=True,text=True,timeout=10)
   if z.stdout.strip() in ('active','activating','deactivating'):live.append(j['unit'])
for name in ['native-quality/supervisor.json','finish-delivery-01/started.json','runtime/supervisor.json']:
 q=p/name
 if not q.exists():continue
 r=json.loads(q.read_text());pid=r.get('PID',r.get('pid'));tick=r.get('start_ticks');s=Path('/proc')/str(pid)/'stat'
 if pid and tick and s.exists():
  v=s.read_text().rsplit(')',1)[1].split()
  if int(v[19])==tick and v[0]!='Z':live.append(pid)
print(json.dumps(dict(missing=missing,live=live,results=results)))
'''%(REL,a.phase)
 try:
  while True:
   assert utc()<deadline,'collection waiting deadline';c=connect()
   try:ready=command(c,readycode,a.remote_cpu,60)
   finally:c.close();c=None
   if not ready['missing'] and not ready['live']:break
   print(json.dumps(dict(status='WAITING_CLOSED_PHASE',missing=ready['missing'],live=ready['live'])),flush=True);time.sleep(min(20,max(.01,(deadline-utc()).total_seconds())))
  roots=[REL,'artifacts/runs/oct02_ow_final_s3','artifacts/runs/oct02_ow_final_s3h2','artifacts/runs/oct02_wm_final_s3h2']
  inventory='''from pathlib import Path
import hashlib,json
rows=[]
for root in %r:
 p0=Path(root);assert p0.is_dir(),root
 for p in sorted(p0.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('preservation-'):continue
  assert not p.is_symlink(),str(p)
  before=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  s=p.stat();assert (s.st_size,s.st_mtime_ns,s.st_ino)==(before.st_size,before.st_mtime_ns,before.st_ino)
  rows.append(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest()))
print(json.dumps(rows))
'''%roots
  c=connect();rows=command(c,inventory,a.remote_cpu,180);assert len({r['path'] for r in rows})==len(rows);s=c.open_sftp()
  for row in rows:
   assert utc()<deadline and shutil.disk_usage(ROOT).free>FLOOR
   relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
   dest=ROOT/relative
   # Existing differing local output is preserved; final bytes receive a separate mapped path.
   if not dest.resolve().is_relative_to(ROOT) or (dest.exists() and digest(dest)!=(row['bytes'],row['sha256'])):
    dest=HERE/'remote-final-snapshot'/relative
   compressed=dest.suffix=='.npy' and row['bytes']>200*2**20 and not dest.exists()
   if compressed:dest=dest.with_name(dest.name+'.gz')
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert digest(dest,compressed)==(row['bytes'],row['sha256'])
   else:
    tmp=dest.with_name(dest.name+'.download');assert not tmp.exists()
    if not compressed:assert shutil.disk_usage(ROOT).free-row['bytes']>FLOOR
    with s.open(REMOTE+'/'+row['path'],'rb') as src,tmp.open('xb') as dst:
     src.prefetch(row['bytes'],max_concurrent_requests=32)
     stream=gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) if compressed else dst
     try:
      while True:
       b=src.read(1<<20)
       if not b:break
       assert utc()<deadline and shutil.disk_usage(ROOT).free>FLOOR+2**20
       stream.write(b)
     finally:
      if compressed:stream.close()
    assert digest(tmp,compressed)==(row['bytes'],row['sha256']);tmp.rename(dest)
   records.append(dict(**row,local_path=str(dest.relative_to(ROOT)),representation='gzip_lossless' if compressed else 'raw',local_bytes=dest.stat().st_size,local_sha256=digest(dest)[1]))
  assert command(c,inventory,a.remote_cpu,180)==rows,'remote tree changed during collection'
  after=command(c,readycode,a.remote_cpu,60);assert not after['missing'] and not after['live'] and after['results']==ready['results']
  rec.update(status='PASS_LOCAL_SSH_LOSSLESS_PRESERVATION',terminal_results=ready['results'],files=len(rows),original_bytes=sum(r['bytes'] for r in rows))
 except BaseException as exc:rec['failure']=repr(exc);raise
 finally:
  rec.update(finished_utc=utc().isoformat(),collector_sha256=digest(Path(__file__))[1],free_disk_bytes=shutil.disk_usage(ROOT).free)
  with target.open('x') as f:json.dump(rec,f,indent=2);f.write('\n')
  if c is not None:c.close()
 # Only after complete byte verification publish the new small receipt remotely.
 c=connect()
 try:
  s=c.open_sftp();rp=REMOTE+'/'+str(target.relative_to(ROOT))
  with s.open(rp,'wx') as f:f.write(target.read_bytes())
  with s.open(rp,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==digest(target)[1]
 finally:c.close()
 print(json.dumps(dict(status=rec['status'],files=rec['files'],original_bytes=rec['original_bytes'],receipt=str(target))),flush=True)
if __name__=='__main__':main()
