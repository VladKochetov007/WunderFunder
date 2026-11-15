"""Time already-built diagnostic libraries; do not qualify or replace a submission."""

from pathlib import Path
import json
import subprocess


root = Path('/data/epanov_s/tabdata-final-submission-20261112-r1')
old = Path('/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1')
baseline_script = (root / 'profiles/precision_diag_r2.py').read_text()
results = {}
for name, directory in (
    ('gelu-simd-quick-r1', root / 'builds/gelu-r2/v3-loader-r3'),
    ('cache-order-quick-r1', root / 'builds/cache-order-r1/v3'),
):
    source = baseline_script.replace('precision-r2', name)
    assert source.count('import solution,numpy as np,wnn_native') == 1
    replacement = 'import numpy as np\nsys.path.insert(0,' + repr(str(directory)) + ')\nimport wnn_native'
    source = source.replace('import solution,numpy as np,wnn_native', replacement)
    compile(source, name, 'exec')
    script = root / 'profiles' / (name + '.py')
    script.write_text(source)
    response = subprocess.run(
        ['/usr/bin/timeout', '--kill-after=5s', '120s',
         str(old / 'tools/build-env/bin/python'), str(script)],
        capture_output=True, text=True, timeout=130,
    )
    record = {'returncode': response.returncode, 'stderr': response.stderr[-3000:]}
    if response.returncode == 0:
        data = json.loads(response.stdout)
        record['policies'] = data['policies']
    else:
        record['stdout'] = response.stdout[-3000:]
    results[name] = record
(root / 'profiles/optimized-quick-timings-r1.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results))
