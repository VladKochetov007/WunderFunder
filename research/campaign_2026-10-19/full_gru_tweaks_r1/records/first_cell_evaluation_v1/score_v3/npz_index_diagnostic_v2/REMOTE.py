import ast,io,json,struct,zipfile
from pathlib import Path
from datetime import datetime,timezone
B=Path('/home/grinchenkoeg/computer-vision-homework-20261018')
class Limited:
    def __init__(self,stream):self.stream=stream;self.bytes_read=0
    def read(self,n=-1):
        if n<0:
            here=self.stream.tell();self.stream.seek(0,2);n=self.stream.tell()-here;self.stream.seek(here)
        if n>262144 or self.bytes_read+n>1048576:raise ValueError('bounded metadata read exceeded')
        raw=self.stream.read(n);self.bytes_read+=len(raw);return raw
    def seek(self,*a):return self.stream.seek(*a)
    def tell(self):return self.stream.tell()
    def seekable(self):return True
def stat(p):
    s=p.stat();l=p.lstat()
    return dict(path=str(p),resolved=str(p.resolve()),is_symlink=p.is_symlink(),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,leaf_inode=l.st_ino)
out=[]
for rel in ('cache/train_targets.npz','cache/valid_targets.npz','cache/chains.npz','cache/chain_folds.npz'):
    p=B/'data'/rel;before=stat(p)
    with p.open('rb') as f:
        limited=Limited(f)
        with zipfile.ZipFile(limited) as z:
            infos=z.infolist()
            if len(infos)>32:raise ValueError('unexpected archive member count')
            rows=[]
            for item in infos:
                row=dict(name=item.filename,expanded_bytes=item.file_size,compressed_bytes=item.compress_size,method=item.compress_type,flags=item.flag_bits,crc=item.CRC)
                if item.filename in ('row_group.npy','seq_ix.npy'):
                    with z.open(item) as member:
                        prefix=member.read(12)
                        if prefix[:6]!=b'\x93NUMPY':raise ValueError('bad selected NPY magic')
                        version=tuple(prefix[6:8]);width=2 if version==(1,0) else 4
                        if version not in ((1,0),(2,0),(3,0)):raise ValueError('bad selected NPY version')
                        size=int.from_bytes(prefix[8:8+width],'little')
                        if size>8192:raise ValueError('selected header too large')
                        offset=8+width;initial=prefix[offset:]
                        header=(initial+member.read(size-len(initial))).decode('latin1' if version!=(3,0) else 'utf8')
                        parsed=ast.literal_eval(header)
                        row['selected_npy_header']=dict(version=version,header_bytes=size,fields={k:list(v) if isinstance(v,tuple) else v for k,v in parsed.items()})
                        row['selected_npy_payload_decoded']=False
                rows.append(row)
        count=limited.bytes_read
    after=stat(p)
    if before!=after:raise ValueError('registered metadata source changed')
    out.append(dict(relative=rel,identity=before,members=rows,bounded_compressed_metadata_bytes_read=count,before_after_stat_equal=True))
print(json.dumps(dict(schema='fg19.first-cell-score-npz-index-metadata-diagnostic.v1',captured_utc=datetime.now(timezone.utc).isoformat(),archives=out,unit_queries=0,nonselected_members_opened=0,array_payloads_decoded=False)))
