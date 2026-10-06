"""Opaque-byte preservation of remote-only and divergent overnight files."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,tarfile,io,datetime
D=Path('/home/grinchenkoeg/delivery-2026-10-01')
ROOT=Path('/home/grinchenkoeg/WunderFunder-creative')
P=ROOT/'research/iterations/2026-10-01-overnight'
LIMIT=64<<20
names=['remote-supplement.inventory.json','remote-supplement.manifest.json','remote-supplement.receipt.json','remote-supplement.tar.gz','remote-supplement.tar.gz.incomplete']
for name in names:assert not(D/name).exists(),name

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def safe(name):
 assert name and not name.startswith('/') and '..' not in PurePosixPath(name).parts
 assert str(PurePosixPath(name))==name and '\\' not in name

def write(name,obj):
 data=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();(D/name).write_bytes(data);return data
main=json.loads((D/'research-archive.manifest.json').read_text())
main_receipt=json.loads((D/'research-archive.receipt.json').read_text())
assert main_receipt['status']=='PASS_ARCHIVE_BYTES_VERIFIED'
assert digest(D/'research-archive.manifest.json')==main_receipt['manifest_sha256']
assert digest(D/'research-overnight.tar.gz')==main_receipt['archive_sha256']
prefix='WunderFunder-creative/'+P.relative_to(ROOT).as_posix()+'/'
known={r['path'][len(prefix):]:r for r in main['files'] if r['path'].startswith(prefix)}
files=[];links=[]
for directory,dirs,fnames in os.walk(P,followlinks=False):
 dirs[:]=sorted(n for n in dirs if n!='__pycache__')
 for name in list(dirs):
  path=Path(directory)/name
  if path.is_symlink():links.append({'path':path.relative_to(P).as_posix(),'target':os.readlink(path)});dirs.remove(name)
 for name in sorted(fnames):
  if name=='.DS_Store' or name.endswith('.pyc'):continue
  path=Path(directory)/name;relative=path.relative_to(P).as_posix();safe(relative)
  if path.is_symlink():links.append({'path':relative,'target':os.readlink(path)});continue
  assert path.is_file();before=path.stat();value=digest(path);after=path.stat()
  assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),relative
  old=known.get(relative)
  classification='remote_only' if old is None else 'shared_equal' if old['bytes']==before.st_size and old['sha256']==value else 'shared_divergent'
  files.append({'path':relative,'bytes':before.st_size,'sha256':value,'mtime_ns':before.st_mtime_ns,'classification':classification})
observed={r['path'] for r in files};missing=sorted(set(known)-observed)
selected=[r for r in files if r['classification']!='shared_equal' and r['bytes']<=LIMIT]
excluded=[r for r in files if r['classification']!='shared_equal' and r['bytes']>LIMIT]
# All large omissions in this specific snapshot must be the known full OOF caches.
assert all(r['path'].startswith('cache/') and r['path'].endswith('.npy') and r['bytes']==299680128 for r in excluded)
assert sum(r['bytes'] for r in selected)<=1<<30
inventory={'schema':'wunder-remote-supplement-inventory-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(P),'main_archive_sha256':main_receipt['archive_sha256'],'main_manifest_sha256':main_receipt['manifest_sha256'],'files':files,'symlinks_not_followed':links,'main_only_paths':missing,'limit_bytes_per_added_file':LIMIT,'packing_rule':'All remote_only or shared_divergent regular files at most64MiB; larger complete OOF caches remain remote with full hashes. Symlinks recorded, not archived.'}
write('remote-supplement.inventory.json',inventory)
records=[{'path':'beleriand-snapshot/'+r['path'],'remote_relative_path':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'classification':r['classification']} for r in selected]
manifest={'schema':'wunder-remote-supplement-manifest-v1','main_archive_sha256':main_receipt['archive_sha256'],'inventory_sha256':digest(D/'remote-supplement.inventory.json'),'remote_root':str(P),'files':records,'excluded_large_files':excluded,'main_only_paths':missing,'symlinks_not_followed':links,'not_a_competition_submission':True,'no_new_forecast_or_metric':True}
manbytes=write('remote-supplement.manifest.json',manifest)
temp=D/'remote-supplement.tar.gz.incomplete';out=D/'remote-supplement.tar.gz'
with tarfile.open(temp,'w:gz',compresslevel=6,dereference=True) as archive:
 for record in records:
  path=P/record['remote_relative_path'];assert not path.is_symlink();assert digest(path)==record['sha256'];safe(record['path'])
  archive.add(path,arcname=record['path'],recursive=False)
 for name,data in [('supplement-manifest.json',manbytes),('remote-supplement.inventory.json',(D/'remote-supplement.inventory.json').read_bytes()),('build-remote-supplement.py',Path(__file__).read_bytes())]:
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;archive.addfile(info,io.BytesIO(data))
expected={r['path']:r for r in records};extras={'supplement-manifest.json':manbytes,'remote-supplement.inventory.json':(D/'remote-supplement.inventory.json').read_bytes(),'build-remote-supplement.py':Path(__file__).read_bytes()};seen=set()
with tarfile.open(temp,'r|gz') as archive:
 for member in archive:
  safe(member.name);assert member.isfile() and member.name not in seen;seen.add(member.name);s=archive.extractfile(member)
  if member.name in extras:assert s.read()==extras[member.name]
  else:
   r=expected[member.name];h=hashlib.sha256()
   for b in iter(lambda:s.read(1<<20),b''):h.update(b)
   assert member.size==r['bytes'] and h.hexdigest()==r['sha256']
assert seen==set(expected)|set(extras)
os.replace(temp,out)
receipt={'status':'PASS_REMOTE_SUPPLEMENT_BYTES_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest(__file__),'main_archive_sha256':main_receipt['archive_sha256'],'inventory_sha256':digest(D/'remote-supplement.inventory.json'),'manifest_sha256':digest(D/'remote-supplement.manifest.json'),'archive_sha256':digest(out),'archive_bytes':out.stat().st_size,'remote_regular_files':len(files),'classification_counts':{k:sum(r['classification']==k for r in files) for k in ['shared_equal','shared_divergent','remote_only']},'main_only_paths':missing,'added_files':len(records),'added_file_bytes':sum(r['bytes'] for r in records),'embedded_metadata_files':3,'excluded_large_files':len(excluded),'excluded_large_bytes':sum(r['bytes'] for r in excluded),'symlinks_not_followed':len(links),'new_inference':0,'new_WP_queries':0,'main_archive_unchanged':digest(D/'research-overnight.tar.gz')==main_receipt['archive_sha256']}
write('remote-supplement.receipt.json',receipt);print(json.dumps(receipt))
