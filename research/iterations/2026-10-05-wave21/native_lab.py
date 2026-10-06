"""Independent Python 3.11 audit of the complete candidate ZIP."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import hashlib
import importlib.util
import tempfile
import zipfile
import time
from pathlib import Path
from types import SimpleNamespace as Point


def main():
    import numpy as np
    old_zip, new_zip, probe, output = map(Path, sys.argv[1:5])
    root = Path(tempfile.mkdtemp())
    models = {}
    for tag, archive in (('reference', old_zip), ('candidate', new_zip)):
        dst = root / tag
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(dst)
        spec = importlib.util.spec_from_file_location('solution_' + tag, dst/'solution.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        models[tag] = mod
    # Both complete artifacts use the same frozen native engine.
    for level in ('v2', 'avx2', 'v3'):
        a = next((root/'reference'/level).glob('*.so'))
        b = next((root/'candidate'/level).glob('*.so'))
        assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    z = np.load(probe)
    x = z['x']
    assert x.shape == (20000, 112)
    timings = {tag: [] for tag in models}
    results = {}
    for order in (('reference', 'candidate'), ('candidate', 'reference')):
        for tag in order:
            model = models[tag].PredictionModel()
            first = []
            start = time.perf_counter()
            for seq in range(5):
                for i, row in enumerate(x):
                    got = model.predict(Point(seq_ix=seq, step_in_seq=i, need_prediction=i >= 99, state=row))
                    if i < 99:
                        assert got is None
                    else:
                        got = np.asarray(got)
                        assert got.shape == (2,) and np.isfinite(got).all()
                        if seq == 0:
                            first.append(got.copy())
                        else:
                            assert np.array_equal(first[i-99], got)
            timings[tag].append(time.perf_counter()-start)
            results[tag] = {'reset_bit_exact': True, 'finite_shape_pass': True,
                            'cpu_level': models[tag].LEVEL, 'scored_rows': 5*(len(x)-99)}
    ratio = float(np.mean(timings['candidate'])/np.mean(timings['reference']))
    report = {'audit': results, 'timings_seconds': timings, 'rows_per_run': 100000,
              'candidate_to_reference_ratio': ratio,
              'candidate_public_minutes_relative_projection': ratio*24.95,
              'reference_public_minutes_user_reported': 24.95,
              'candidate_sha256': hashlib.sha256(new_zip.read_bytes()).hexdigest(),
              'scope': 'Complete actual ZIPs, independent DataSphere CPU, Python3.11. One full real sequence repeated five times; model reset checked exactly. Timing includes Python row-level validation overhead; projected scorer time is not certified.'}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
