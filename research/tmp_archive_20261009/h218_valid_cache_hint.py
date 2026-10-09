import fcntl, hashlib, json, os, shutil, subprocess
from datetime import UTC, datetime
from pathlib import Path
root=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
valid=root/'datasets/valid.parquet'
runtime=root/'research/campaign_2026-10-06/bandit_continuation/runtime'
receipt=root/'research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_PREEXPORT_VALID_CACHE_HANDOFF_V1_20261008.json'
cgroup=Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice')
if receipt.exists(): raise FileExistsError(receipt)
if not valid.is_file(): raise FileNotFoundError(valid)
props={n:(cgroup/n).read_text().strip() for n in ('memory.high','memory.max','memory.swap.max')}
if int(props['memory.high'])>18_000_000_000 or int(props['memory.max'])>20_000_000_000 or int(props['memory.swap.max'])!=0:
    raise RuntimeError(f'campaign bounds mismatch: {props}')
fdstat=valid.stat()
expected_bytes=5_615_509_128
expected_mtime=1_577_826_000_000_000_000
if fdstat.st_size!=expected_bytes or fdstat.st_mtime_ns!=expected_mtime:
    raise RuntimeError('valid.parquet stat differs from frozen fold identity')
fuser=shutil.which('fuser')
if not fuser: raise RuntimeError('fuser unavailable; cannot confirm no active exact-file reader')
probe=subprocess.run([fuser,'-s',str(valid)],capture_output=True,text=True,timeout=5)
if probe.returncode==0: raise RuntimeError('valid.parquet currently has an active reader; no cache hint applied')
if probe.returncode!=1: raise RuntimeError(f'cannot verify valid.parquet readers (fuser rc={probe.returncode})')
lock_path=runtime/'gpu.lock'
lock_path.parent.mkdir(parents=True,exist_ok=True)
def snapshot():
    stat=dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
    return {'current':int((cgroup/'memory.current').read_text()),'file':int(stat['file']),'anon':int(stat['anon'])}
def file_identity():
    s=valid.stat()
    return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
with lock_path.open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=snapshot(); identity_before=file_identity()
    if identity_before!={'device':fdstat.st_dev,'inode':fdstat.st_ino,'bytes':expected_bytes,'mtime_ns':expected_mtime}:
        raise RuntimeError('valid.parquet identity changed before hint')
    fd=os.open(valid,os.O_RDONLY)
    try:
        os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    after=snapshot(); identity_after=file_identity()
    if identity_after!=identity_before: raise RuntimeError('valid.parquet stat identity changed during cache hint')
    doc={
      'schema':'h218-preexport-known-parquet-cache-handoff-v1',
      'status':'complete',
      'observed_utc':datetime.now(UTC).isoformat(),
      'action':'POSIX_FADV_DONTNEED on the exact frozen datasets/valid.parquet path; no file contents or metadata changed',
      'path':'datasets/valid.parquet',
      'frozen_fold_identity':{'bytes':expected_bytes,'mtime_ns':expected_mtime,'footer_sha256':'4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b'},
      'stat_before':identity_before,'stat_after':identity_after,
      'reader_preflight':'fuser confirmed no open reader',
      'gpu_lock':'research/campaign_2026-10-06/bandit_continuation/runtime/gpu.lock',
      'gpu_lock_held':True,
      'parent_cgroup':str(cgroup),
      'parent_limits':props,
      'parent_memory_before':before,'parent_memory_after':after,
      'limits_changed':False,
      'swap_changed':False,
      'fit_or_prediction_changed':False,
      'next_reservation_bytes':9_000_000_000,
    }
    tmp=receipt.with_suffix(receipt.suffix+'.partial')
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    os.replace(tmp,receipt)
print(json.dumps({'receipt':str(receipt.relative_to(root)),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'before':before,'after':after,'stat_unchanged':identity_before==identity_after},sort_keys=True))
