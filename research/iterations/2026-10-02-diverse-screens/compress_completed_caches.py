"""Lossless local archival of four completed OOF caches; remote originals unchanged."""
from pathlib import Path
import gzip,hashlib,json,shutil,time
ROOT=Path(__file__).resolve().parents[3]
assert str(ROOT)=='/Users/nb/Documents/wunder/WunderFunder-creative'
paths=[ROOT/'research/iterations'/study/'cache'/filename for study,filename in [
('2026-10-02-mingru-forecast','pred_oct02_mingru_s3h2_f0_valid.npy'),
('2026-10-02-mingru-forecast','pred_oct02_mingru_s3h2_f1_valid.npy'),
('2026-10-01-online-normalization','pred_cont_onsh256_s3_h2_f0_valid.npy'),
('2026-10-01-online-normalization','pred_cont_onsh256_s3_h2_f1_valid.npy')]]
def digest(f):
 h=hashlib.sha256();n=0
 for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b)
 return h.hexdigest(),n
records=[]
for p in paths:
 r=json.loads(p.with_suffix('.inputs.json').read_text());expected=r['prediction_sha256']
 with p.open('rb') as f:before,n=digest(f)
 assert before==expected
 z=p.with_name(p.name+'.gz');assert not z.exists()
 with p.open('rb') as src,z.open('xb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as out:shutil.copyfileobj(src,out,1<<20)
 with gzip.open(z,'rb') as f:after,m=digest(f)
 assert (after,m)==(before,n)
 with z.open('rb') as f:archive,nz=digest(f)
 p.unlink()
 records.append(dict(original_path=str(p.relative_to(ROOT)),original_bytes=n,sha256=before,local_path=str(z.relative_to(ROOT)),archive_bytes=nz,archive_sha256=archive,decompressed_byte_identity=True,remote_original_retained=True))
 print(str(p),n-nz,flush=True)
receipt=dict(status='PASS_LOSSLESS_LOCAL_CACHE_ARCHIVE',records=records,saved_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in records),restore='gzip -dk path/to/prediction.npy.gz',note='Supersedes local raw representation in earlier preservation receipts; original byte content and SSH originals unchanged.')
with Path(__file__).with_name('local-cache-archive.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(status=receipt['status'],saved_bytes=receipt['saved_bytes'])))
