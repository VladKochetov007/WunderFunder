"""Root allocation for reviewed fixed branch commands; does not launch training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import importlib.util
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    p.add_argument('--gpu', type=int, required=True, choices=[0, 3, 7])
    p.add_argument('--cpus', required=True)
    p.add_argument('--hours', type=float, default=2)
    a = p.parse_args()
    here = ROOT / a.campaign
    assert here.resolve().is_relative_to(ROOT / 'research/iterations')
    assert here.name.startswith('2026-10-02-') and here.name != '2026-10-02-mingru-forecast'
    rel = str(here.relative_to(ROOT))
    spec = importlib.util.spec_from_file_location('branch_queue', here / 'run_queue.py')
    Q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Q)
    assert Q.sha(here / 'run_queue.py') == 'a6d51d912635c071bcf001d98e1e7e251718992a40c84db2a4586345019bd5c2'
    plan = json.loads((here / 'commands.json').read_text())
    for path, digest in plan['source_sha256'].items():
        assert Q.sha(ROOT / path) == digest, path
    g = Q.H.gpu(a.gpu)
    assert g['free_MiB'] >= 8192
    assert shutil.disk_usage(ROOT).free > 10 * 2**30
    assert 0 < a.hours <= 4
    cpus = list(map(int, a.cpus.split(',')))
    assert len(cpus) == len(set(cpus)) == 6
    if a.gpu != 0:
        assert not set(cpus).intersection([22,24,32,34,36,39])
    old = json.loads((ROOT / 'research/iterations/2026-10-01-ensemble-backbone/training-allocation.json').read_text())
    sources = {k:v for k,v in old['sources'].items()
               if k.startswith(('src/', 'scripts/')) or k.endswith('/train_aux.py') or k.endswith('context-replay/run.py')}
    sources.update(plan['source_sha256'])
    for f in (here / 'commands.json', here / 'run_queue.py', here / 'launch.py', Path(__file__),
              Path(__file__).with_name('contract.json')):
        sources[str(f.relative_to(ROOT))] = Q.sha(f)
    jobs = []
    for j in plan['jobs']:
        stage = j['stage']
        assert stage in ('parent', 'H2', 'OOF')
        if stage == 'OOF':
            jobs.append(dict(name=j['name'], stage=stage, argv=j['argv'], timeout_seconds=900,
                             require_absent=j['require_absent'], required_outputs=j['required_outputs']))
            continue
        check = rel + '/execution-01/' + j['name'] + '.check.json'
        run = 'artifacts/runs/' + j['name']
        ep = int(j['resolved_args']['epochs']) - 1
        assert ep == 3 if stage == 'parent' else ep in (0, 1)
        jobs.append(dict(name=j['name'], stage=stage, argv=j['argv'],
                         timeout_seconds=3600 if stage == 'parent' else 1200,
                         require_absent=[run, check],
                         required_outputs=[run+f'/model_ep{i}.pt' for i in range(ep+1)] +
                                          [run+'/model.pt',run+'/history.json',check],
                         post_argv=['.venv/bin/python', rel+'/check_checkpoint.py',j['name'],
                                    '--commands',rel+'/commands.json','--output',check]))
    fits = sum(j['stage'] != 'OOF' for j in jobs)
    oof = sum(j['stage'] == 'OOF' for j in jobs)
    assert (fits,oof) in ((4,2),(8,4))
    now = datetime.now(timezone.utc)
    env = dict(old['environment'])
    env.update(WNN_THREADS='4',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='4')
    allocation = dict(execution_authorized=True,created_utc=now.isoformat(),
        deadline_utc=(now+timedelta(hours=a.hours)).isoformat(), execution_directory=rel+'/execution-01',
        success_status=f'COMPLETE_{fits}_FITS_{oof}_OOF_NO_METRICS',
        resources=dict(gpu_id=a.gpu,gpu_uuid=g['uuid'],allocator_fraction=.375,
                       owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,
                       cpu_ids=cpus,cpu_quota=600,host_memory_bytes=20*2**30),
        environment=env,sources=sources,input_bindings=[old['input_bindings'][0]],
        jobs=jobs,gpu_preflight=g,quality_queries=0,retries=0)
    for j in jobs:
        for path in j['require_absent']:
            assert not (ROOT/path).exists() and not (ROOT/path).is_symlink(), path
    target = here / 'training-allocation.json'
    Q.write(target, allocation)
    Q.contract(target, Q.sha(target))
    print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',fits=fits,OOF=oof,
                         allocation_sha256=Q.sha(target),gpu=g,deadline=allocation['deadline_utc'])))


if __name__ == '__main__':
    main()
