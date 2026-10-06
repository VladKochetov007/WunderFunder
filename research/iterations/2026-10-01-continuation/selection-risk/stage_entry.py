"""Record/verify owned cgroup and affinity, then exec the allocated stage."""
from pathlib import Path
import json, os, sys
stage, allocation, digest = sys.argv[1:]
here = Path(__file__).resolve().parent
assert stage in ('extract', 'analysis')
group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
limits = {n: (cgroup / n).read_text().strip() for n in ('cpu.max', 'memory.max', 'memory.swap.max')}
quota, period = map(int, limits['cpu.max'].split())
assert quota / period == 2 and int(limits['memory.max']) == 4 * 2**30 and limits['memory.swap.max'] == '0'
a = json.loads(Path(allocation).read_text())
assert sorted(os.sched_getaffinity(0)) == a[stage + '_cpus']
with (here / f'{stage}-resource.json').open('x') as out:
    json.dump({'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
               'cgroup': group, 'limits': limits, 'affinity': sorted(os.sched_getaffinity(0)),
               'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}, out, indent=2)
script = here / ('extract_selected.py' if stage == 'extract' else 'fit_selectors.py')
os.execv(sys.executable, [sys.executable, str(script), '--allocation', allocation, '--allocation-sha256', digest])
