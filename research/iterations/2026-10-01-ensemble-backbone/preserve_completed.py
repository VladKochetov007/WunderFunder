"""Archive completed study only. Read-only SSH source; never mutates remote runs."""
import sys, json, hashlib, pathlib, shlex, gzip, shutil, time
sys.path.insert(0, '/tmp/wunder-ssh-client')
import paramiko
root=pathlib.Path('/Users/nb/Documents/wunder/WunderFunder-creative')
b=pathlib.Path('research/iterations/2026-10-01-ensemble-backbone')
remote='/home/grinchenkoeg/WunderFunder-creative'
started=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20), b''):h.update(x)
 return h.hexdigest()
archives=json.loads((root/b/'cache-local-archive.json').read_text())
assert archives['status']=='PASS_LOSSLESS_LOCAL_ARCHIVE_REMOTE_RAW_RETAINED'
fit=json.loads((root/b/'fit-local-preservation.json').read_text())
assert fit['status']=='PASS_ALL_12_JOB_OUTPUT_BYTE_IDENTITIES'
archive_by_path={r['original_path']:r for r in archives['records']}
c=paramiko.SSHClient();c.load_host_keys('/Users/nb/.ssh/known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('185.185.59.19',username='grinchenkoeg',key_filename='/Users/nb/.ssh/beleriand_grinchenkoeg_ed25519',look_for_keys=False,allow_agent=False,timeout=10);c.get_transport().set_keepalive(30)
script="""
import json,hashlib
from pathlib import Path
b=Path('research/iterations/2026-10-01-ensemble-backbone')
assert json.loads((b/'finish-02/result.json').read_text())['status']=='COMPLETE_ONE_PAIRED_STUDY'
assert json.loads((b/'execution-02/result.json').read_text())['status']=='COMPLETED_REMAINING_4_FITS_3_OOF_NO_METRICS'
paths={p for p in b.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for dirname in ['execution-01','execution-02']:
 for p in (b/dirname).glob('*.worker-result.json'):
  r=json.loads(p.read_text());assert r['status']=='COMPLETE';paths.update(map(Path,r['outputs']))
rows=[]
for p in sorted(paths):
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 rows.append(dict(path=str(p),bytes=after.st_size,sha256=h.hexdigest()))
print(json.dumps(dict(status='COMPLETE_REMOTE_BYTE_INVENTORY',records=rows)))
"""
cmd='cd '+shlex.quote(remote)+' && taskset -c 22,23 nice -n 10 timeout 240s .venv/bin/python -c '+shlex.quote(script)
try:
 i,o,e=c.exec_command(cmd,timeout=260);i.close();raw=o.read();err=e.read();assert o.channel.recv_exit_status()==0,err.decode()
 inv=json.loads(raw);ip=root/b/'remote-completed-inventory.json';assert not ip.exists();ip.write_bytes(raw)
 s=c.open_sftp();records=[]
 for r in inv['records']:
  rel=r['path'];p=root/rel
  if rel in archive_by_path:
   a=archive_by_path[rel];assert r['bytes']==a['original_bytes'] and r['sha256']==a['original_sha256']
   z=root/a['local_archive_path'];assert z.stat().st_size==a['archive_bytes'] and sha(z)==a['archive_sha256']
   h=hashlib.sha256();n=0
   with gzip.open(z,'rb') as f:
    for block in iter(lambda:f.read(1<<20),b''):h.update(block);n+=len(block)
   assert n==r['bytes'] and h.hexdigest()==r['sha256']
   records.append(dict(**r,local_path=str(z.relative_to(root)),representation='gzip_lossless',archive_sha256=a['archive_sha256']));continue
  earlier=None
  if p.exists() and (p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']):
   earlier=dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
   p=root/b/'remote-final-snapshot'/pathlib.Path(rel).relative_to(b)
  if not p.exists():
   assert shutil.disk_usage(root).free-r['bytes']>3*2**30
   p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.download');assert not tmp.exists()
   s.get(remote+'/'+rel,str(tmp),max_concurrent_prefetch_requests=64)
   assert tmp.stat().st_size==r['bytes'] and sha(tmp)==r['sha256'],rel
   tmp.rename(p)
  assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],rel
  records.append(dict(**r,local_path=str(p.relative_to(root)),representation='raw',preserved_earlier_local=earlier))
 s.close()
 out=dict(status='PASS_COMPLETE_REMOTE_LOCAL_BYTE_PRESERVATION',records=records,remote_file_count=len(records),remote_total_bytes=sum(r['bytes'] for r in records),lossless_archives=len(archive_by_path),seconds=time.monotonic()-started,scope='Completed scientific artifacts and all remote study files except Python bytecode; preservation inventory and this receipt are outside their own recursive inventory. Raw remote cache originals retained; local gzip payloads independently decompressed and verified. Earlier local snapshots retained when final remote logs differ.')
 p=root/b/'completed-local-preservation.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
finally:c.close()
