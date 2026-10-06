import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import subprocess
import time
import numpy as np

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave26'
engines = {'legacy': R/'research/iterations/2026-10-05-wave16/native-sparse/build',
           'extended': D/'native-class/build'}
weights = {'h256': R/'research/iterations/2026-10-05-wave21/student_final_s5.npz',
           'h384': R/'research/iterations/2026-10-05-wave22/wide-init.npz',
           'sparse': R/'research/iterations/2026-10-05-wave16/sparse_twostream_fastbase/weights_twostream.npz',
           'linear': R/'research/iterations/2026-10-05-wave16/sparse_twostream_fastbase/weights_5.npz'}

if len(sys.argv) > 1:
    engine, level, run = sys.argv[1:4]
    sys.path.insert(0, str(engines[engine]/level))
    import wnn_cpp
    x = np.load(D/'fixtures-probe.npz')['x'][0]
    output, timing = {}, {}
    for name, file in weights.items():
        w = {k: np.ascontiguousarray(v, np.float32) for k, v in np.load(file).items()}
        model = wnn_cpp.Ensemble([w], [1.], dynamic_int8=True)
        output[name] = model.run(x)
        if level == 'v3':
            start = time.perf_counter()
            for _ in range(30):
                model.run(x)
            timing[name] = time.perf_counter()-start
    if engine == 'extended' and level == 'v3':
        for name in ('zero', 'soft', 'uniform'):
            w = {k: np.ascontiguousarray(v, np.float32) for k, v in np.load(D/f'fixture-{name}.npz').items()}
            model = wnn_cpp.Ensemble([w], [1.], dynamic_int8=True)
            model.run(x)
            start = time.perf_counter()
            for _ in range(30):
                model.run(x)
            timing['class_'+name] = time.perf_counter()-start
    np.savez_compressed(D/f'compat-{engine}-{level}-{run}.npz', **output)
    (D/f'timing-{engine}-{level}-{run}.json').write_text(json.dumps(timing))
else:
    report = {'legacy_bit_exact': [], 'timing_runs': {},
              'scope': 'Four legacy weight formats, 2000 real rows each, all three builds. Timing is a local native run microbenchmark with sequence reset every2000 rows; no full-package runtime certification.'}
    for level in ('v2', 'avx2', 'v3'):
        orders = (('legacy', 'extended'), ('extended', 'legacy')) if level == 'v3' else (('legacy', 'extended'),)
        for run, order in enumerate(orders):
            for engine in order:
                subprocess.run([sys.executable, __file__, engine, level, str(run)], check=True)
                report['timing_runs'][f'{engine}-{level}-{run}'] = json.loads((D/f'timing-{engine}-{level}-{run}.json').read_text())
            a = np.load(D/f'compat-legacy-{level}-{run}.npz')
            b = np.load(D/f'compat-extended-{level}-{run}.npz')
            for name in weights:
                assert a[name].tobytes() == b[name].tobytes(), (level, run, name, float(abs(a[name]-b[name]).max()))
            report['legacy_bit_exact'].append({'level': level, 'run': run, 'formats': list(weights), 'pass': True})
    report['extended_to_legacy_time_ratio'] = {}
    for name in weights:
        old = np.mean([report['timing_runs'][f'legacy-v3-{run}'][name] for run in (0, 1)])
        new = np.mean([report['timing_runs'][f'extended-v3-{run}'][name] for run in (0, 1)])
        report['extended_to_legacy_time_ratio'][name] = float(new/old)
    (D/'compatibility.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
