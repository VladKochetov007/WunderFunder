"""Read-only CPU inventory, sent through SSH stdin; no model forward or target decode."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path('/home/grinchenkoeg/WunderFunder-creative')
ORIGINAL = Path('/home/grinchenkoeg/WunderFunder')
BEGIN = time.monotonic()


def emit(kind, **values):
    print(json.dumps({'kind': kind, 'seconds': time.monotonic()-BEGIN, **values}), flush=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve()), 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


emit('start', PID=os.getpid(), GPU_execution=False, target_values_decoded=False,
     chunk_sequences=4, input_repo=str(ROOT))
groups = {}
for split in ('train', 'valid'):
    p = ROOT / 'cache' / f'{split}_targets.npz'
    with np.load(p, allow_pickle=False) as z:
        groups[split] = np.asarray(z['row_group']).copy()
    g = groups[split]
    emit('row_map', **identity(p), row_group_shape=list(g.shape), row_group_dtype=str(g.dtype),
         unique=int(len(np.unique(g))), minimum=int(g.min()), maximum=int(g.max()),
         row_group_bytes_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
         target_arrays_not_loaded=True)
p = ROOT / 'cache/chain_folds.npz'
with np.load(p, allow_pickle=False) as z:
    split_values = z['split']; row_values = z['row_group']; fold_values = z['fold']
valid_folds = {int(g): int(f) for s, g, f in zip(split_values, row_values, fold_values) if str(s) == 'valid'}
emit('chain_folds', **identity(p), sha256=sha(p), valid_rowgroups=len(valid_folds),
     missing_valid_metadata=[int(g) for g in groups['valid'] if int(g) not in valid_folds][:20])

for fold in (0, 1):
    for teacher in (f'h256_f{fold}', f'h256s4_f{fold}'):
        for split in ('train', 'valid'):
            path = ROOT / 'cache' / f'pred_{teacher}_{split}.npy'
            if not path.is_file():
                emit('teacher_cache_missing', path=str(path)); continue
            before = identity(path); a = np.load(path, mmap_mode='r', allow_pickle=False)
            assert a.ndim == 3 and a.shape == (len(groups[split]), 20000, 2), (path, a.shape)
            assert a.dtype == np.float32 and a.flags.c_contiguous
            assert before['size'] == a.offset + a.nbytes
            h = hashlib.sha256()
            with path.open('rb') as f:
                h.update(f.read(a.offset))
            bad = []; nonfinite = 0
            for start in range(0, len(a), 4):
                block = np.asarray(a[start:start+4]); h.update(memoryview(block).cast('B'))
                ok = np.isfinite(block); nonfinite += int(ok.size - ok.sum())
                bad.extend((start + np.flatnonzero(~ok.all(axis=(1, 2)))).tolist())
                del block, ok
            shape = list(a.shape); a._mmap.close(); del a; gc.collect()
            assert before == identity(path), ('input changed', path)
            bad_training = bad if split == 'train' else [i for i in bad if valid_folds[int(groups[split][i])] != fold]
            emit('teacher_cache', **before, teacher=teacher, split=split, fold=fold,
                 shape=shape, dtype='float32', sha256=h.hexdigest(), all_values_finite=not bad,
                 nonfinite_values=nonfinite, nonfinite_sequences=len(bad),
                 nonfinite_rowgroups_first20=[int(groups[split][i]) for i in bad[:20]],
                 nonfinite_training_side_sequences=len(bad_training),
                 row_map_bytes_sha256=hashlib.sha256(groups[split].tobytes()).hexdigest(),
                 existing_sidecars=[str(x) for x in (path.with_suffix('.json'), path.with_suffix('.inputs.json')) if x.exists()])

import torch
torch.set_num_threads(1)
keys = ['name', 'hidden', 'layers', 'cell', 'enc', 'arch', 'seed', 'valid_fold', 'n_valid_folds',
        'train_source', 'include_heldout_valid', 'lookahead', 'init', 'distill', 'distill_alpha',
        'epochs', 'eval_ema', 'ema_teacher', 'propensity', 'since_weight']
for fold in (0, 1):
    for teacher in (f'h256_f{fold}', f'h256s4_f{fold}'):
        for endpoint in ('model.pt', 'model_ep3.pt'):
            p = ORIGINAL / 'artifacts/runs' / teacher / endpoint
            if not p.exists():
                emit('teacher_checkpoint_missing', path=str(p)); continue
            ck = torch.load(p, map_location='cpu', weights_only=False)
            emit('teacher_checkpoint', **identity(p), sha256=sha(p), epoch=ck.get('epoch'),
                 n_in=ck.get('n_in'), args={k: ck['args'].get(k, '<absent>') for k in keys})
            del ck
    p = ROOT / 'artifacts/runs' / f'prop_f{fold}' / 'model.pt'
    model_sha = sha(p)
    emit('propensity_checkpoint', **identity(p), sha256=model_sha)
    for split in ('train', 'valid'):
        p = ROOT / 'cache' / f'propcache_prop_f{fold}_{split}.npy'
        m = p.with_suffix('.json'); meta = json.loads(m.read_text())
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        emit('propensity_cache', **identity(p), shape=list(a.shape), dtype=str(a.dtype),
             sidecar_sha256=sha(m), sidecar=meta, model_hash_matches=meta.get('model_sha256')==model_sha)
        a._mmap.close(); del a
    for run in ('on_ctld256', 'on_ctl128s3', 'on_ctl128s4', 'on_ctlc500'):
        name = f'{run}_f{fold}'; p = ROOT / 'artifacts/runs' / name / 'model_ep1.pt'
        q = ROOT / 'research/iterations/2026-10-01-overnight/cache' / f'pred_{name}_valid.npy'
        record = {'name': name, 'checkpoint_exists': p.is_file(), 'prediction_exists': q.is_file()}
        if p.exists():
            record['checkpoint'] = identity(p); record['checkpoint_sha256'] = sha(p)
        if q.exists():
            a = np.load(q, mmap_mode='r', allow_pickle=False)
            record['prediction'] = identity(q); record['shape'] = list(a.shape); record['dtype'] = str(a.dtype)
            a._mmap.close(); del a
        receipt = q.with_suffix('.inputs.json')
        record['receipt_exists'] = receipt.is_file()
        if receipt.exists():
            r = json.loads(receipt.read_text())
            record['receipt_sha256'] = sha(receipt)
            record['receipt_checkpoint_sha256'] = r.get('checkpoint_sha256')
            record['receipt_checkpoint_matches'] = r.get('checkpoint_sha256') == record.get('checkpoint_sha256')
            record['receipt_prediction_sha256'] = r.get('prediction_sha256')
        emit('comparison_H2', **record)

import pyarrow.parquet as pq
for split in ('train', 'valid'):
    p = ROOT / 'datasets' / f'{split}.parquet'; before = identity(p)
    metadata = pq.ParquetFile(p).metadata
    with p.open('rb') as f:
        f.seek(max(0, before['size'] - 65536)); footer_sha = hashlib.sha256(f.read()).hexdigest()
    emit('dataset_metadata', **before, row_groups=metadata.num_row_groups, rows=metadata.num_rows,
         last65536bytes_sha256=footer_sha, whole_file_sha256=None)

# Search only project-owned small text source/log locations, bounded at 16MiB total.
scanned = 0; matches = []; files = 0; budget = 16 * 2**20
for directory in (ORIGINAL/'artifacts/logs', ORIGINAL/'scripts', ORIGINAL/'research', ROOT/'artifacts/logs'):
    if not directory.exists(): continue
    for p in sorted(directory.rglob('*')):
        if scanned >= budget: break
        if not p.is_file() or p.suffix not in ('.sh', '.log', '.py') or p.stat().st_size > 2**20: continue
        b = p.read_bytes(); scanned += len(b); files += 1
        for n, line in enumerate(b.decode('utf-8', errors='replace').splitlines(), 1):
            if 'h256' in line and ('predict_run.py' in line or 'pred_h256' in line):
                matches.append({'path': str(p), 'line': n, 'text': line[:1000]})
emit('producer_log_search', bytes_scanned=scanned, files_scanned=files, bounded_bytes=budget,
     exact_prediction_command_matches=matches[:40], match_count=len(matches),
     limitation='Search scope is bounded; absence of a producer receipt does not establish checkpoint epoch.')
emit('complete', peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     GPU_execution=False, target_values_decoded=False, metric_computed=False, remote_files_written=False)
