"""Allocation-gated single fresh-process pair with cgroup and identity-bound monitoring."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time
from compare import compare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'pgrp': int(fields[2]), 'state': fields[0]}
    except FileNotFoundError:
        return None


def same_process(a, b):
    return b is not None and all(a[k] == b[k] for k in ['pid', 'start_ticks', 'pgrp'])


def process_memory(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
        return {s.split(':')[0]: int(s.split()[1]) for s in lines if s.startswith(('VmRSS:', 'VmHWM:', 'Threads:'))}
    except FileNotFoundError:
        return {}


def gpu_state():
    out = subprocess.check_output(['nvidia-smi', '--id=6', '--query-gpu=uuid,memory.free,memory.used',
                                   '--format=csv,noheader,nounits'], text=True, timeout=2).strip().split(',')
    gpu = {'uuid': out[0].strip(), 'free_MiB': int(out[1]), 'total_used_MiB': int(out[2]), 'processes': []}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=2)
    for line in apps.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) == 3 and fields[0] == gpu['uuid']:
            gpu['processes'].append({'pid': int(fields[1]), 'MiB': int(fields[2])})
    return gpu


def terminate_owned(proc, owner):
    current = identity(proc.pid)
    if same_process(owner, current):
        os.killpg(owner['pgrp'], signal.SIGKILL)
    elif proc.poll() is None:
        raise RuntimeError('refusing signal to changed process identity')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allocation', type=Path, required=True)
    ap.add_argument('--allocation-sha256', required=True)
    a = ap.parse_args()
    assert sha(a.allocation) == a.allocation_sha256
    allocation = json.loads(a.allocation.read_text())
    assert allocation['prefetch_generated_pair_authorized'] is True
    assert not allocation.get('competition_inputs_authorized', False)
    assert not allocation.get('forecast_fits_authorized', False)
    assert allocation['physical_GPU'] == 6 and allocation['attempts_per_arm'] == 1
    plan_path = HERE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert allocation['plan_sha256'] == sha(plan_path)
    assert allocation['freeze_sha256'] == sha(HERE / 'FREEZE.json')
    def verify_sources():
        for rel, expected in plan['source_sha256'].items():
            assert sha(ROOT / rel) == expected, rel
    verify_sources()
    os.sched_setaffinity(0, {38})
    group_path = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    cg = Path('/sys/fs/cgroup') / group_path.lstrip('/')
    limits = {key: (cg / key).read_text().strip() for key in ['memory.max', 'memory.swap.max', 'cpu.max']}
    assert limits == {'memory.max': str(4 << 30), 'memory.swap.max': '0', 'cpu.max': '400000 100000'}
    for key in ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF']:
        assert os.environ.get(key, '') == ''
    out = HERE / 'execution'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    started = now()
    jobs = []
    failure = None
    status = 'RUNNING'
    compared = None
    current_proc = current_owner = None
    def deadline_handler(signum, frame):
        if current_proc is not None:
            terminate_owned(current_proc, current_owner)
        raise TimeoutError('allocated generated-only arm or stage deadline')
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, 300.)
    try:
        for arm in ['per_epoch', 'reuse']:
            verify_sources()
            prelaunch = gpu_state()
            assert prelaunch['free_MiB'] >= 6144 and prelaunch['total_used_MiB'] <= 5632 and not prelaunch['processes'], prelaunch
            result = out / (arm + '.json')
            log = out / (arm + '.log')
            sample_path = out / (arm + '-samples.jsonl')
            argv = ['taskset', '-c', '24-29', str(ROOT / '.venv/bin/python'), str(HERE / 'probe.py'),
                    '--arm', arm, '--allocation', str(a.allocation.resolve()),
                    '--allocation-sha256', a.allocation_sha256, '--output', str(result)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            arm_start = time.monotonic()
            with log.open('x') as output:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                owner = identity(proc.pid)
                assert owner and owner['pgrp'] == proc.pid
                current_proc, current_owner = proc, owner
                job = {'arm': arm, 'started_UTC': now(), 'argv': argv, 'identity': owner,
                       'prelaunch': prelaunch, 'environment': {k: env[k] for k in ['CUDA_VISIBLE_DEVICES',
                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG']},
                       'peaks': {'GPU_MiB': 0, 'GPU_total_MiB': 0, 'process_RSS_KiB': 0, 'process_HWM_KiB': 0, 'cgroup_memory_bytes': 0},
                       'termination': None}
                jobs.append(job)
                signal.setitimer(signal.ITIMER_REAL, min(120., max(.001, 300. - (arm_start - start))))
                try:
                    with sample_path.open('x') as samples:
                        while proc.poll() is None:
                            gpu = gpu_state()
                            mem = process_memory(proc.pid)
                            cg_bytes = int((cg / 'memory.current').read_text())
                            own = [p for p in gpu['processes'] if p['pid'] == proc.pid]
                            foreign = [p for p in gpu['processes'] if p['pid'] != proc.pid]
                            gpu_mib = max([p['MiB'] for p in own], default=0)
                            sample = {'UTC': now(), 'elapsed': time.monotonic() - arm_start, 'GPU': gpu,
                                      'process_identity': identity(proc.pid), 'process_memory': mem,
                                      'cgroup_memory_bytes': cg_bytes}
                            samples.write(json.dumps(sample) + '\n'); samples.flush()
                            peaks = job['peaks']
                            for key, value in [('GPU_MiB', gpu_mib), ('GPU_total_MiB', gpu['total_used_MiB']), ('process_RSS_KiB', mem.get('VmRSS', 0)),
                                               ('process_HWM_KiB', mem.get('VmHWM', 0)), ('cgroup_memory_bytes', cg_bytes)]:
                                peaks[key] = max(peaks[key], value)
                            if foreign or gpu_mib > 5632 or gpu['total_used_MiB'] > 5632 or cg_bytes > (4 << 30):
                                job['termination'] = 'foreign GPU occupancy or resource limit'
                                terminate_owned(proc, owner)
                                raise RuntimeError(job['termination'])
                            time.sleep(.5)
                    assert proc.wait() == 0 and result.exists(), (arm, proc.returncode)
                    data = json.loads(result.read_text())
                    assert data['status'] == 'COMPLETE'
                    assert data['peak_reserved_bytes'] <= (5 << 30)
                    job['result_sha256'] = sha(result)
                finally:
                    if proc.poll() is None:
                        terminate_owned(proc, owner)
                    proc.wait()
                    job.update(exit_code=proc.returncode, finished_UTC=now(), seconds=time.monotonic() - arm_start,
                               log_sha256=sha(log), sample_sha256=sha(sample_path) if sample_path.exists() else None,
                               process_released=identity(proc.pid) is None)
                    current_proc = current_owner = None
                    signal.setitimer(signal.ITIMER_REAL, max(.001, 300. - (time.monotonic() - start)))
        verify_sources()
        compared = compare(out / 'per_epoch.json', out / 'reuse.json')
        (out / 'comparison.json').write_text(json.dumps(compared, indent=2) + '\n')
        status = compared['status']
    except BaseException as exc:
        status = 'CLOSED_DIAGNOSTIC_FAILURE'
        failure = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt = {'status': status, 'failure': failure, 'started_UTC': started, 'finished_UTC': now(),
                   'seconds': time.monotonic() - start, 'allocation_sha256': a.allocation_sha256,
                   'plan_sha256': sha(plan_path), 'driver_sha256': sha(__file__), 'jobs': jobs,
                   'cgroup': group_path, 'cgroup_limits': limits, 'cgroup_memory_events': (cg / 'memory.events').read_text(),
                   'comparison': compared, 'competition_inputs_read': False, 'forecast_fits': 0, 'WP_queries': 0,
                   'attempts_per_arm': 1, 'no_retry': True, 'no_production_change_authorized': True,
                   'monitor_limits': 'cgroup enforces aggregate host memory and no swap; NVML sampled at 0.5s plus bounded query latency may miss between-sample GPU peaks; natural allocator peak checked only at equal epoch boundaries without changing allocator behavior. SIGALRM deadline depends on OS scheduling. Signals require owned PID/start/pgrp.'}
        (out / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2), flush=True)
    if status == 'CLOSED_DIAGNOSTIC_FAILURE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
