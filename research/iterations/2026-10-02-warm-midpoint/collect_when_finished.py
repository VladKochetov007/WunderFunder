"""Local terminal-only preservation of the allocated campaign and registered run directories, lossless gzip for large local prediction caches.

Does not train, score, restart, delete or publish. Waits for both known remote
controllers to exit and their terminal receipts; copies byte-verified artifacts.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,gzip,hashlib,json,os,shlex,shutil,sys,time
sys.path.insert(0,'/tmp/wunder-ssh-client')
import paramiko
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);args=ap.parse_args()
HERE=ROOT/args.campaign
assert HERE.resolve().is_relative_to(ROOT/'research/iterations') and HERE.name == '2026-10-02-warm-midpoint'
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
 if not (p/name).exists():live.append('missing '+name);continue
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
 runs=[j['name'] for j in json.loads((HERE/'commands.json').read_text())['pairs']];assert len(runs) == 4
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
  assert shutil.disk_usage(ROOT).free>3*2**30, 'local headroom'; records=[]
  s=c.open_sftp()
  for row in rows:
   p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
   compressed=p.suffix=='.npy' and row['bytes']>200*2**20 and not p.exists()
   destpath=p.with_name(p.name+'.gz') if compressed else p
   if not destpath.exists():
    tmp=destpath.with_name(destpath.name+'.download');assert not tmp.exists()
    if compressed:
     with s.open(REMOTE+'/'+row['path'],'rb') as src, tmp.open('xb') as dst:
      src.prefetch(row['bytes'],max_concurrent_requests=32)
      with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as gz:
       while True:
        chunk=src.read(1<<20)
        if not chunk:break
        assert shutil.disk_usage(ROOT).free>3*2**30, 'local headroom while compressing'
        gz.write(chunk)
    else:
     assert shutil.disk_usage(ROOT).free-row['bytes']>3*2**30, 'local headroom'
     s.get(REMOTE+'/'+row['path'],str(tmp),max_concurrent_prefetch_requests=32)
    destpath_for_check=tmp
   else:destpath_for_check=destpath
   h=hashlib.sha256();length=0
   with (gzip.open(destpath_for_check,'rb') if compressed else destpath_for_check.open('rb')) as check:
    for chunk in iter(lambda:check.read(1<<20),b''):h.update(chunk);length+=len(chunk)
   assert length==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
   if destpath_for_check!=destpath:destpath_for_check.rename(destpath)
   records.append(dict(**row,local_path=str(destpath.relative_to(ROOT)),representation='gzip_lossless' if compressed else 'raw',local_bytes=destpath.stat().st_size,local_sha256=sha(destpath)))
  receipt=dict(status='PASS_LOCAL_SSH_LOSSLESS_PRESERVATION',allocation_sha256=ah,terminal_results=r['results'],files=len(rows),bytes=sum(x['bytes'] for x in rows),records=records,created_utc=datetime.now(timezone.utc).isoformat(),collector_sha256=sha(Path(__file__)))
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
