"""Attach read-only canonical resource observers to four active torchrun families."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/data/epanov_s/tabdata-final-quality-20261112-r1')
source = Path('/data/epanov_s/tabdata-all-20261112-r1/runtimes/pr75')
python = '/data/epanov_s/tabdata/.venv/bin/python'
observer = source/'scripts/observe_sequence.py'
assert observer.is_file() and '--pid' in observer.read_text()
assert 'never sends signals' in observer.read_text()
clock = json.loads((root/'control/CLOCK_R1.json').read_bytes())
remaining = int(clock['fit_deadline_epoch']-time.time())
assert remaining > 0
control = root/'control/resource-observers-r1'
control.mkdir(exist_ok=False)
rows = []
for output in sorted((root/'runs/pr91-input').glob('*/fit')):
    assert not (output/'monitor_observer.jsonl').exists()
    progress = json.loads((output/'monitor_progress.json').read_bytes())
    assert progress['phase'] in ('training','evaluation','checkpointing','validation')
    worker_pid = int(progress['pid'])
    parent = int(Path('/proc/'+str(worker_pid)+'/stat').read_text().split(') ')[1].split()[1])
    parent_cmd = Path('/proc/'+str(parent)+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'torch' in parent_cmd, parent_cmd
    env = dict(os.environ, PYTHONPATH=str(source/'src'), OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command = [python,str(observer),'--output',str(output),'--pid',str(parent),
               '--interval','10','--max-seconds',str(remaining)]
    with (control/(output.parent.name+'.log')).open('xb') as log:
        process = subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rows.append({'cell':output.parent.name,'observer_pid':process.pid,'bound_worker_pid':worker_pid,
                 'bound_parent_pid':parent,'command':command,'at_epoch':time.time()})
with (control/'LAUNCH_OWNERS_R1.json').open('x') as handle:
    json.dump(rows,handle,indent=2)
print(json.dumps({'observers':rows,'read_only':True,'may_signal_workers':False}))
