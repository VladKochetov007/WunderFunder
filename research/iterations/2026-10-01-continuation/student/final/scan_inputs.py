"""One bounded read-only finite/hash pass over the four historical final teacher caches."""
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
ROOT = Path('/home/grinchenkoeg/WunderFunder-creative')
START = time.monotonic()
def identity(p):
    s = p.stat()
    return dict(path=str(p), resolved=str(p.resolve()), size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)
def emit(kind, **kw):
    print(json.dumps(dict(kind=kind, seconds=time.monotonic()-START, **kw)), flush=True)
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
rowmaps = {}
emit('start', PID=os.getpid(), target_values_decoded=False, GPU_execution=False)
for split in ('train','valid'):
    p = ROOT/'cache'/f'{split}_targets.npz'
    before = identity(p)
    with np.load(p, allow_pickle=False) as z: g = z['row_group'].copy()
    expected = np.delete(np.arange(10607, dtype=np.int64),528) if split=='train' else np.arange(1873,dtype=np.int64)
    assert g.dtype==np.int64 and np.array_equal(g,expected)
    assert identity(p)==before
    rowmaps[split] = g
    emit('row_map', **before, shape=list(g.shape), dtype=str(g.dtype), bytes_sha256=hashlib.sha256(g.tobytes()).hexdigest(), target_values_decoded=False)
for teacher in ('fin256_s3','fin256_s4'):
    for split in ('train','valid'):
        p=ROOT/'cache'/f'pred_{teacher}_{split}.npy'; before=identity(p)
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        assert a.shape==(len(rowmaps[split]),20000,2) and a.dtype==np.float32 and a.flags.c_contiguous
        assert before['size']==a.offset+a.nbytes
        h=hashlib.sha256()
        with p.open('rb') as f: h.update(f.read(a.offset))
        nonfinite=0
        for start in range(0,len(a),4):
            b=np.asarray(a[start:start+4]); h.update(memoryview(b).cast('B'))
            good=np.isfinite(b); nonfinite+=int(good.size-good.sum()); del b,good
        shape=list(a.shape); a._mmap.close(); del a; gc.collect()
        assert identity(p)==before
        emit('teacher_cache', **before, teacher=teacher, split=split, shape=shape, dtype='float32', sha256=h.hexdigest(), nonfinite_values=nonfinite, all_values_finite=nonfinite==0, row_map_sha256=hashlib.sha256(rowmaps[split].tobytes()).hexdigest())
        assert nonfinite==0
for split in ('train','valid'):
    emit('dataset_stat', **identity(ROOT/'datasets'/f'{split}.parquet'))
p=ROOT/'artifacts/runs/prop_f0/model.pt'; model_sha=sha(p)
emit('propensity_model', **identity(p), sha256=model_sha)
for split in ('train','valid'):
    p=ROOT/'cache'/f'propcache_prop_f0_{split}.npy'; m=p.with_suffix('.json')
    meta=json.loads(m.read_text()); assert meta['model_sha256']==model_sha
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    emit('propensity', **identity(p), shape=list(a.shape), dtype=str(a.dtype), sidecar=meta, sidecar_sha256=sha(m))
    a._mmap.close(); del a
emit('complete', status='PASS_FINITE_HASH_ROW_ORDER', peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, target_values_decoded=False, forecast_fit_or_metric=False, files_written=False)
