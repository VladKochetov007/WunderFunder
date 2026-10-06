"""Local terminal-only preservation of the allocated campaign and four run directories.

Does not train, score, restart, delete or publish. Waits for both known remote
controllers to exit and their terminal receipts; copies byte-verified artifacts.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,shlex,shutil,sys,time
sys.path.insert(0,'/tmp/wunder-ssh-client')
import paramiko
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REL=str(HERE.relative_to(ROOT));REMOTE='/home/grinchenkoeg/WunderFunder-creative'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def connect():
 c=paramiko.SSHClient();c.load_host_keys('/Users/nb/.ssh/known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
 c.connect('185.185.59.19',username='grinchenkoeg',key_filename='/Users/nb/.ssh/beleriand_grinchenkoeg_ed25519',look_for_keys=False,allow_agent=False,timeout=10);c.get_transport().set_keepalive(30);return c
def command(c,code,limit=60):
 cmd='cd '+shlex.quote(REMOTE)+' && taskset -c 22 nice -n 10 timeout '+str(limit)+'s .venv/bin/python -c '+shlex.quote(code)
 i,o,e=c.exec_command(cmd,timeout=limit+15);i.close();raw=o.read();error=e.read();assert o.channel.recv_exit_status()==0,error.decode();return json.loads(raw)
def main():
 assert str(ROOT)=='/Users/nb/Documents/wunder/WunderFunder-creative','Local collector only'
 a=json.loads((HERE/'training-allocation.json').read_text());ah=sha(HERE/'training-allocation.json');deadline=datetime.fromisoformat(a['deadline_utc'])+timedelta(minutes=30)
 dest=HERE/'preservation-complete.json';assert not dest.exists()
 readycode='''from pathlib import Path
import json
p=Path(%r);live=[]
for name in ('launch.json','finish-launch.json'):
 r=json.loads((p/name).read_text());s=Path('/proc')/str(r['pid'])/'stat'
 if s.exists():
  z=s.read_text().split()
  if z[21]==r['start_ticks'] and z[2]!='Z':live.append(r['pid'])
results={n:json.loads((p/n).read_text()) if (p/n).exists() else None for n in ('execution-01/result.json','finish-01/result.json')}
print(json.dumps(dict(live=live,results=results)))
'''%REL
 while True:
  assert datetime.now(timezone.utc)<deadline,'Preservation waiting deadline exceeded'
  c=connect()
  try:r=command(c,readycode)
  finally:c.close()
  if not r['live'] and all(v is not None for v in r['results'].values()):break
  print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status='WAITING_TERMINAL_CONTROLLERS',live=r['live'])),flush=True);time.sleep(60)
 assert r['results']['execution-01/result.json']['allocation_sha256']==ah
 for result in r['results'].values():
  assert not result.get('remaining_scope_pids') and all(not j.get('remaining_scope_pids') for j in result.get('jobs',[]))
 runs=[j['name'] for j in a['jobs'] if j['stage'] in ('parent','H2')];assert len(runs)==4
 roots=[REL,*['artifacts/runs/'+n for n in runs]]
 inventory='''from pathlib import Path
import hashlib,json
rows=[]
for root in %r:
 for p in sorted(Path(root).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
  s=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  t=p.stat();assert (s.st_size,s.st_mtime_ns)==(t.st_size,t.st_mtime_ns)
  rows.append(dict(path=str(p),bytes=t.st_size,sha256=h.hexdigest()))
print(json.dumps(rows))
'''%roots
 c=connect()
 try:
  rows=command(c,inventory,180);assert len({x['path'] for x in rows})==len(rows)
  need=sum(x['bytes'] for x in rows if not (ROOT/x['path']).exists());assert shutil.disk_usage(ROOT).free-need>3*2**30,('local headroom',need)
  s=c.open_sftp()
  for row in rows:
   p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():
    tmp=p.with_name(p.name+'.download');assert not tmp.exists();s.get(REMOTE+'/'+row['path'],str(tmp),max_concurrent_prefetch_requests=64)
    assert tmp.stat().st_size==row['bytes'] and sha(tmp)==row['sha256'],row['path'];tmp.rename(p)
   assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
  receipt=dict(status='PASS_LOCAL_SSH_BYTE_IDENTITY',allocation_sha256=ah,terminal_results=r['results'],files=len(rows),bytes=sum(x['bytes'] for x in rows),records=rows,created_utc=datetime.now(timezone.utc).isoformat(),collector_sha256=sha(Path(__file__)))
  with dest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
  remote_dest=REMOTE+'/'+REL+'/'+dest.name
  try:s.stat(remote_dest)
  except FileNotFoundError:pass
  else:raise FileExistsError(remote_dest)
  with s.open(remote_dest,'wx') as f:f.write(dest.read_bytes())
  with s.open(remote_dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(dest)
  s.close();print(json.dumps(dict(status=receipt['status'],files=receipt['files'],bytes=receipt['bytes'])),flush=True)
 finally:c.close()
if __name__=='__main__':main()
