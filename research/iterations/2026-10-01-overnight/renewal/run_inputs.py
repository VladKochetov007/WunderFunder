#!/usr/bin/env python3
"""The only empirical entry point: exactly ten named causal input columns."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ALLOWED_COLUMNS = ['a5', 'a6'] + [f'i0_dv{i}' for i in range(4)] + [f'i0_dp{i}' for i in range(4)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def fingerprint(path):
    st = path.stat()
    with path.open('rb') as f:
        f.seek(-8, 2); tail = f.read(8)
        if tail[4:] != b'PAR1':
            raise ValueError('Parquet trailer mismatch')
        size = int.from_bytes(tail[:4], 'little')
        f.seek(-8 - size, 2)
        footer_hash = hashlib.sha256(f.read(size) + tail).hexdigest()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'inode': st.st_ino, 'device': st.st_dev, 'footer_bytes': size + 8,
            'footer_sha256': footer_hash, 'whole_file_hashed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--execute-inputs', action='store_true', help='Required after independent review; no other stage exists')
    args = parser.parse_args()
    if not args.execute_inputs:
        parser.error('No action: --execute-inputs is required after allocation and independent review')
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if plan['id'] != 'QRENEWAL-01-inputs-only' or plan['columns'] != ALLOWED_COLUMNS:
        raise ValueError('Unexpected plan ID or non-whitelisted columns')
    if plan['caps'] != {'cpu_threads': 2, 'address_space_bytes': 4294967296, 'wall_seconds': 480, 'cpu_seconds': 480}:
        raise ValueError('Resource caps differ from fixed plan')
    for name, expected in plan['files_sha256'].items():
        if Path(name).name != name or sha(HERE / name) != expected:
            raise ValueError(f'Source or manifest drift: {name}')
    output = HERE / 'result'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[key] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
    resource.setrlimit(resource.RLIMIT_CPU, (480, 480))
    affinity = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, affinity)
    def timeout(signum, frame):
        raise TimeoutError('Fixed 480-second wall cap reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(480)
    receipt = {'id': plan['id'], 'plan_sha256': sha(plan_path), 'files_sha256': plan['files_sha256'],
               'columns': ALLOWED_COLUMNS, 'caps': plan['caps'], 'cpu_affinity': affinity,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'stage': 'inputs-only', 'target_stage_implemented': False}
    write_new(output / 'started.json', receipt)
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from core import audit_clock, audit_marks, fit_clock, to_json
        manifest = json.loads((HERE / 'selected-chains.json').read_text())
        selected = manifest['selected']
        repo = HERE.parents[3]
        for path_key, hash_key, selection_key in [
                ('source_manifest', 'source_manifest_sha256', 'selected'),
                ('original_manifest', 'original_manifest_sha256', 'selected_before_feature_reads')]:
            historical = repo / manifest[path_key]
            if sha(historical) != manifest[hash_key]:
                raise ValueError('Historical selection manifest hash mismatch')
            if json.loads(historical.read_text())[selection_key] != selected:
                raise ValueError('Exact historical ordered selection mismatch')
        assert len(selected) == 256 and len({r['chain'] for r in selected}) == 256
        assert len({r['row_group'] for r in selected}) == 256
        for half in range(2):
            for cls in range(4):
                assert sum(r['half'] == half and r['class'] == cls for r in selected) == 32
        source = Path(manifest['source']['path'])
        before = fingerprint(source)
        for key in ['bytes', 'mtime_ns', 'footer_bytes', 'footer_sha256']:
            if before[key] != manifest['source'][key]:
                raise ValueError(f'Historical source fingerprint mismatch: {key}')
        pf = pq.ParquetFile(source)
        records = []
        for row in selected:
            if pf.metadata.row_group(row['row_group']).num_rows != 20000:
                raise ValueError('Unexpected sequence length')
            table = pf.read_row_group(row['row_group'], columns=ALLOWED_COLUMNS, use_threads=False)
            if table.column_names != ALLOWED_COLUMNS:
                raise ValueError('Unexpected columns returned')
            values = np.column_stack([table.column(name).to_numpy(zero_copy_only=False) for name in ALLOWED_COLUMNS]).astype(np.float32)
            # Preserve existing payload detector's float32 comparisons; solve clock
            # constraints in float64 without subtractive precision loss in the reset threshold.
            records.append(dict(row, age=values[:, :2].astype(np.float64), dv=values[:, 2:6], dp=values[:, 6:]))
        read_seconds = time.monotonic() - started
        directions = []
        for fit_half in [0, 1]:
            training = [r for r in records if r['half'] == fit_half]
            audit = [r for r in records if r['half'] != fit_half]
            clock_fit = fit_clock(training)
            clock = audit_clock(audit, clock_fit)
            marks = audit_marks(training, audit)
            direction = to_json({'fit_half': fit_half, 'audit_half': 1 - fit_half,
                                 'clock_fit': clock_fit, 'clock_audit': clock, 'mark_audit': marks})
            directions.append(direction)
            write_new(output / f'direction-fit{fit_half}.json', direction)
        after = fingerprint(source)
        if before != after:
            raise ValueError('Source changed during execution')
        passed = all(all(d['clock_audit']['gates'].values()) and all(d['mark_audit']['gates'].values()) for d in directions)
        result = dict(receipt, status='COMPLETE', input_gate_pass=passed, directions=directions,
                      source_before=before, source_after=after, read_seconds=read_seconds,
                      wall_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      numpy_version=np.__version__, pyarrow_version=__import__('pyarrow').__version__,
                      scipy_version=__import__('scipy').__version__,
                      interpretation='Input structural audit only; no price-target association, forecast fit, or WP query')
        write_new(output / 'result.json', result)
        print(json.dumps({'status': 'COMPLETE', 'input_gate_pass': passed, 'wall_seconds': result['wall_seconds']}))
    except BaseException as e:
        write_new(output / 'failure.json', dict(receipt, status='FAILED_NO_RETRY', error=repr(e),
                  wall_seconds=time.monotonic() - started,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
