"""One root-authorized SELRISK01 run, bounded owned scopes, no retries."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, subprocess, time, uuid
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allocation', type=Path, required=True)
    ap.add_argument('--allocation-sha256', required=True)
    args = ap.parse_args()
    allocation = args.allocation.resolve()
    assert sha(allocation) == args.allocation_sha256
    a = json.loads(allocation.read_text())
    assert a['status'] == 'ROOT_ALLOCATED_SELECTION_RISK' and not a['retry']
    bp = ROOT / a['bindings']
    assert sha(bp) == a['bindings_sha256']
    b = json.loads(bp.read_text())
    for path, digest in b['sources'].items(): assert sha(ROOT / path) == digest
    assert not (HERE / 'launch-started.json').exists() and not (ROOT / a['output']).exists()
    gpu = str(a['gpu_index'])
    used = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
    assert float(used) <= 256 and not apps, 'Allocated GPU not idle; no stage launched'
    launch = {'utc': utc(), 'pid': os.getpid(), 'allocation_sha256': args.allocation_sha256,
              'GPU_precheck': {'gpu': gpu, 'used_MiB': float(used), 'compute_apps': apps}, 'stages': []}
    (HERE / 'launch-started.json').write_text(json.dumps(launch, indent=2)+'\n')
    for stage, seconds in (('extract', 600), ('analysis', 300)):
        scope = 'selrisk-' + stage + '-' + uuid.uuid4().hex
        cpu = ','.join(map(str, a[stage + '_cpus']))
        argv = ['systemd-run', '--user', '--scope', '--unit='+scope, '-p', 'CPUQuota=200%', '-p', 'MemoryMax=4G', '-p', 'MemorySwapMax=0', '--',
                'taskset', '-c', cpu, 'timeout', '--signal=TERM', '--kill-after=5s', str(seconds)+'s',
                'env', 'CUDA_VISIBLE_DEVICES='+(gpu if stage=='extract' else ''), 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                'MKL_NUM_THREADS=2', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH='+str(ROOT/'src'),
                str(ROOT/'.venv/bin/python'), str(HERE/'stage_entry.py'), stage, str(allocation), args.allocation_sha256]
        item = {'stage': stage, 'scope': scope+'.scope', 'argv': argv, 'utc_start': utc()}
        launch['stages'].append(item)
        (HERE / 'launch-progress.json').write_text(json.dumps(launch, indent=2)+'\n')
        start = time.monotonic()
        with (HERE/(stage+'-stdout.log')).open('x') as out:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            item['launcher_pid'] = proc.pid
            (HERE / 'launch-progress.json').write_text(json.dumps(launch, indent=2)+'\n')
            try:
                code = proc.wait(timeout=seconds+15)
            except subprocess.TimeoutExpired:
                subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', scope+'.scope'], capture_output=True)
                proc.kill(); code = proc.wait(); item['outer_timeout'] = True
        item.update(exit_code=code, elapsed_seconds=time.monotonic()-start, utc_end=utc())
        resource_path = HERE/(stage+'-resource.json')
        if resource_path.exists():
            resource = json.loads(resource_path.read_text())
            item['worker_pid'] = resource['pid']
            item['worker_proc_exists_after'] = Path('/proc') .joinpath(str(resource['pid'])).exists()
        show = subprocess.run(['systemctl', '--user', 'show', scope+'.scope', '-p', 'ActiveState', '-p', 'SubState'], capture_output=True, text=True)
        item['scope_show_stdout'] = show.stdout
        (HERE / 'launch-progress.json').write_text(json.dumps(launch, indent=2)+'\n')
        if code != 0: break
        assert not item['worker_proc_exists_after']
    launch['status'] = 'COMPLETE' if len(launch['stages'])==2 and all(x['exit_code']==0 for x in launch['stages']) else 'FAILED_NO_RETRY'
    launch['utc_end'] = utc()
    (HERE / 'launch-result.json').write_text(json.dumps(launch, indent=2)+'\n')
    print(json.dumps({'status':launch['status'], 'receipt':str(HERE/'launch-result.json')}))
    return 0 if launch['status']=='COMPLETE' else 1
if __name__=='__main__': raise SystemExit(main())
