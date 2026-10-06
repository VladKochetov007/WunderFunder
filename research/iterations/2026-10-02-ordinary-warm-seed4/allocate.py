"""Root-invoked bounded allocation only. Never launches a fit or a grader."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, importlib.util, json, shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUEUE_SHA = 'a6d51d912635c071bcf001d98e1e7e251718992a40c84db2a4586345019bd5c2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0, choices=[0])
    p.add_argument('--cpus', default='16,17,18,19,20,21')
    p.add_argument('--hours', type=float, default=1.0)
    a = p.parse_args()
    assert 0 < a.hours <= 1.0
    cpus = list(map(int, a.cpus.split(',')))
    assert cpus == [16,17,18,19,20,21]
    rel = str(HERE.relative_to(ROOT))
    spec = importlib.util.spec_from_file_location('interval_queue', HERE/'run_queue.py')
    Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
    assert Q.sha(HERE/'run_queue.py') == QUEUE_SHA
    target = HERE/'training-allocation.json'; binding_path = HERE/'plain-parent-bindings.json'
    assert not target.exists() and not target.is_symlink()
    assert not binding_path.exists() and not binding_path.is_symlink()
    plan = json.loads((HERE/'commands.json').read_text())
    assert plan['counts'] == dict(fits=4, OOF=2)
    for path, digest in plan['source_sha256'].items():
        assert Q.sha(ROOT/path) == digest, path

    # GPU0/CPU16:21 are a dependency on the complete atom queue AND grader.
    receipts = {}
    for relative, status in (
        ('research/iterations/2026-10-02-ordinary-warm/execution-01/result.json', 'COMPLETE_4_FITS_2_OOF_NO_METRICS'),
        ('research/iterations/2026-10-02-ordinary-warm/finish-01/result.json', 'COMPLETE_FIXED_FORECAST_READOUT'),
        (rel+'/generated-01.json', 'PASS_GENERATED_ORDINARY_WARM')):
        value = json.loads((ROOT/relative).read_text())
        assert value['status'] == status, (relative, value['status'])
        assert not value.get('remaining_scope_pids')
        assert all(not j.get('remaining_scope_pids') for j in value.get('jobs', []))
        receipts[relative] = Q.sha(ROOT/relative)
    for name in ('launch.json', 'finish-launch.json'):
        prior = json.loads((ROOT/'research/iterations/2026-10-02-ordinary-warm'/name).read_text())
        stat = Path('/proc')/str(prior['pid'])/'stat'
        if stat.exists():
            fields = stat.read_text().split()
            assert fields[21] != prior['start_ticks'] or fields[2] == 'Z', 'Atom controller still running'
    previous = json.loads((ROOT/'research/iterations/2026-10-02-ordinary-warm/evaluation-01/report.json').read_text())
    primary=previous['comparisons']['primary']['scopes']['all']
    assert primary['delta_WP'] >= .0001 and all(v['delta_WP']>0 for v in primary['folds'].values()), 'Ordinary warm spending gate did not pass'
    g = Q.H.gpu(a.gpu)
    assert g['free_MiB'] >= 8192
    assert shutil.disk_usage(ROOT).free > 10*2**30

    old = json.loads((ROOT/'research/iterations/2026-10-01-ensemble-backbone/training-allocation.json').read_text())
    sources = {k:v for k,v in old['sources'].items()
               if k.startswith(('src/', 'scripts/')) or k.endswith('/train_aux.py') or k.endswith('context-replay/run.py')}
    sources.update(plan['source_sha256']); sources.update(receipts)
    for name in ('commands.json','run_queue.py','allocate.py','launch.py','finish.py','launch_finish.py',
                 'collect_when_finished.py','launch_collector.py','freeze_evaluation.py','evaluate.py','decision-contract.json'):
        path = HERE/name; sources[str(path.relative_to(ROOT))] = Q.sha(path)
    contract = ROOT/'research/iterations/2026-10-02-diverse-screens/contract.json'
    sources[str(contract.relative_to(ROOT))] = Q.sha(contract)
    for relative, digest in sources.items(): assert Q.sha(ROOT/relative) == digest, relative

    jobs = []
    for j in plan['jobs']:
        stage = j['stage']; assert stage in ('parent','H2','OOF')
        if stage == 'OOF':
            jobs.append(dict(name=j['name'],stage=stage,argv=j['argv'],timeout_seconds=600,
                             require_absent=j['require_absent'],required_outputs=j['required_outputs']))
            continue
        assert len(j['resolved_args']) == 103
        ep = int(j['resolved_args']['epochs'])-1
        assert ep == (0 if stage == 'parent' else 1)
        run = 'artifacts/runs/'+j['name']; check = rel+'/execution-01/'+j['name']+'.check.json'
        jobs.append(dict(name=j['name'],stage=stage,argv=j['argv'],timeout_seconds=1200 if stage=='parent' else 900,
            require_absent=[run,check],required_outputs=[run+f'/model_ep{i}.pt' for i in range(ep+1)]+
                [run+'/model.pt',run+'/history.json',run+'/initial-state.json',check],
            post_argv=['.venv/bin/python',rel+'/check_checkpoint.py',j['name'],
                       '--commands',rel+'/commands.json','--output',check]))
    assert sum(j['stage']!='OOF' for j in jobs)==4 and sum(j['stage']=='OOF' for j in jobs)==2
    for j in jobs:
        for path in j['require_absent']:
            assert not (ROOT/path).exists() and not (ROOT/path).is_symlink(), path
    records = []
    for fold in ('0','1'):
        b = plan['plain_parents'][fold]; path = ROOT/b['checkpoint']; before = path.stat()
        assert Q.sha(path) == b['sha256']; after = path.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        records.append(dict(path=str(path),resolved=str(path.resolve()),size=after.st_size,mtime_ns=after.st_mtime_ns,
                            inode=after.st_ino,device=after.st_dev,sha256=b['sha256'],fold=int(fold),endpoint_epoch=3))
    now = datetime.now(timezone.utc)
    Q.write(binding_path,dict(status='BOUND_ORIGINAL_SHALLOW256_EMA_EP3',created_utc=now.isoformat(),records=records))
    env = dict(old['environment']); env.update(WNN_THREADS='4',WNN_LOADER_WORKERS='2',WNN_LOADER_RAM_GB='4')
    allocation = dict(execution_authorized=True,created_utc=now.isoformat(),
        deadline_utc=(now+timedelta(hours=a.hours)).isoformat(),execution_directory=rel+'/execution-01',
        success_status='COMPLETE_4_FITS_2_OOF_NO_METRICS',resources=dict(gpu_id=0,gpu_uuid=g['uuid'],
            allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,running_free_MiB=1024,
            cpu_ids=cpus,cpu_quota=600,host_memory_bytes=20*2**30),environment=env,sources=sources,
        input_bindings=[old['input_bindings'][0],dict(path=str(binding_path.relative_to(ROOT)),sha256=Q.sha(binding_path))],
        jobs=jobs,gpu_preflight=g,quality_queries=0,retries=0)
    Q.write(target,allocation); Q.contract(target,Q.sha(target))
    print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',fits=4,OOF=2,allocation_sha256=Q.sha(target),
                          gpu=g,deadline=allocation['deadline_utc'])))


if __name__ == '__main__': main()
