import fcntl,hashlib,json,os,shutil,subprocess
from datetime import UTC,datetime
from pathlib import Path
root=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
path=root/'datasets/train.parquet'; runtime=root/'research/campaign_2026-10-06/bandit_continuation/runtime'
receipt=root/'research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_PREEXPORT_TRAIN_CACHE_HANDOFF_V1_20261008.json'
cg=Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice')
if receipt.exists():raise FileExistsError(receipt)
fold=json.loads((root/'research/campaign_2026-09-29/artifacts/splits/fold0.json').read_text())
expected=fold['sources']['datasets/train.parquet']; s=path.stat()
if s.st_size!=expected['bytes'] or s.st_mtime_ns!=expected['mtime_ns']:raise RuntimeError('train.parquet stat differs from frozen fold manifest')
limits={n:(cg/n).read_text().strip() for n in ('memory.high','memory.max','memory.swap.max')}
if int(limits['memory.high'])>18_000_000_000 or int(limits['memory.max'])>20_000_000_000 or int(limits['memory.swap.max'])!=0:raise RuntimeError(f'campaign limit mismatch: {limits}')
fuser=shutil.which('fuser')
if not fuser:raise RuntimeError('fuser unavailable')
p=subprocess.run([fuser,'-s',str(path)],capture_output=True,text=True,timeout=5)
if p.returncode==0:raise RuntimeError('train.parquet has active reader; no cache hint applied')
if p.returncode!=1:raise RuntimeError(f'cannot verify train parquet readers (rc={p.returncode})')
def snap():
 st=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
 return {'current':int((cg/'memory.current').read_text()),'file':int(st['file']),'anon':int(st['anon'])}
def ident():
 q=path.stat();return {'device':q.st_dev,'inode':q.st_ino,'bytes':q.st_size,'mtime_ns':q.st_mtime_ns}
lockpath=runtime/'gpu.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
with lockpath.open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 before=snap(); ident_before=ident()
 if ident_before['bytes']!=expected['bytes'] or ident_before['mtime_ns']!=expected['mtime_ns']:raise RuntimeError('train.parquet identity changed before hint')
 fd=os.open(path,os.O_RDONLY)
 try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
 finally:os.close(fd)
 after=snap();ident_after=ident()
 if ident_after!=ident_before:raise RuntimeError('train.parquet stat changed during hint')
 doc={'schema':'h218-preexport-known-parquet-cache-handoff-v1','status':'complete','observed_utc':datetime.now(UTC).isoformat(),'action':'POSIX_FADV_DONTNEED on exact frozen datasets/train.parquet; no file contents or metadata changed','path':'datasets/train.parquet','frozen_fold_identity':expected,'stat_before':ident_before,'stat_after':ident_after,'reader_preflight':'fuser confirmed no open reader','gpu_lock':'research/campaign_2026-10-06/bandit_continuation/runtime/gpu.lock','gpu_lock_held':True,'parent_cgroup':str(cg),'parent_limits':limits,'parent_memory_before':before,'parent_memory_after':after,'limits_changed':False,'swap_changed':False,'fit_or_prediction_changed':False,'next_reservation_bytes':9_000_000_000}
 tmp=receipt.with_suffix(receipt.suffix+'.partial');tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');os.replace(tmp,receipt)
print(json.dumps({'receipt':str(receipt.relative_to(root)),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'before':before,'after':after,'stat_unchanged':ident_before==ident_after},sort_keys=True))
