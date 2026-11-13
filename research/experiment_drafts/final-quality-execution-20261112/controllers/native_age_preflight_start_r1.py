"""Validate actual canonical configs before four owned engineering profiles."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/data/lc-epanov_s/tabdata-final-quality-20261112-r1')
packet = root / 'age-intake-r1/since-scale-followup-r1'
python = '/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python'
source = '/data2/lc-epanov_s/tabdata-all-followups-20261112-r1/source-r2'
preflight = '''
import json, hashlib
from pathlib import Path
from wnn.training.sequence_config import load_config
packet=Path('/data/lc-epanov_s/tabdata-final-quality-20261112-r1/age-intake-r1/since-scale-followup-r1')
panel=json.loads((packet/'PANEL_MANIFEST_R1.json').read_bytes())
results=[]
for entry in panel['cells']:
    path=packet/entry['config']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['config_sha256']
    baseline=packet.parents[1]/'intake/pr49-distillation-r1/configs'/('label_since32_k16_seed'+str(entry['seed'])+'.json')
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==entry['matched_since32_config_sha256']
    config=load_config(path)
    control=load_config(baseline)
    difference=sorted(key for key in set(config)|set(control) if config.get(key)!=control.get(key))
    assert difference==['since_weight'],difference
    assert config['kd_weight']==0 and config['since_weight']==entry['since_weight']
    results.append({'cell':entry['cell'],'status':'canonical_load_config_passed','changed_fields':difference})
print(json.dumps({'status':'passed','results':results}))
'''
compile(preflight, '<canonical-config-preflight>', 'exec')
env = dict(os.environ, PYTHONPATH=source + '/src', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
result = subprocess.run([python, '-c', preflight], cwd=source, env=env,
                        capture_output=True, text=True, timeout=90)
receipt = {'at_epoch':time.time(), 'returncode':result.returncode,
           'stdout':result.stdout, 'stderr':result.stderr}
target = root / 'control/AGE_CANONICAL_PREFLIGHT_R1.json'
with target.open('x') as handle:
    json.dump(receipt, handle, indent=2)
if result.returncode:
    print(json.dumps(receipt))
else:
    command = [python, str(packet / 'l4_age_wave_r1.py'), 'profile-age']
    log_path = root / 'control/age-profiles-r1.log'
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    owner = {'pid':process.pid, 'pgid':os.getpgid(process.pid),
             'birth':int(Path('/proc/'+str(process.pid)+'/stat').read_text().split(') ')[1].split()[19]),
             'command':command, 'at_epoch':time.time(), 'log':str(log_path)}
    with (root/'control/AGE_PROFILE_LAUNCH_R1.json').open('x') as handle:
        json.dump(owner, handle, indent=2)
    print(json.dumps({'preflight':receipt,'profile_owner':owner}))
