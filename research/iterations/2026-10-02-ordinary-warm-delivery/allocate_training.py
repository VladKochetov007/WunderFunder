"""Root-invoked allocation; no model loading, inference, training, or launch."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util, json, shutil

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]

def main():
    spec=importlib.util.spec_from_file_location('queue',HERE/'run_queue.py')
    Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
    rel=str(HERE.relative_to(ROOT));out=HERE/'training-allocation.json';bp=HERE/'final-input-bindings.json'
    assert not out.exists() and not bp.exists()
    plan=json.loads((HERE/'commands.json').read_text())
    assert plan['counts']==dict(fits=2,OOF=0,metrics=0)
    sources=dict(plan['source_sha256']);receipts={}
    for branch,status in [('ordinary-warm','COMPLETE_FIXED_ORDINARY_WARM_SCREEN'),('ordinary-warm-seed4','COMPLETE_FIXED_ORDINARY_WARM_SEED4_SCREEN')]:
        directory=ROOT/('research/iterations/2026-10-02-'+branch)
        p=directory/'evaluation-01/report.json';r=json.loads(p.read_text());assert r['status']==status
        v=r['comparisons']['primary']['scopes']['all']
        assert v['delta_WP']>=(.0001 if branch=='ordinary-warm' else 0.) and all(f['delta_WP']>0 for f in v['folds'].values())
        receipts[str(p.relative_to(ROOT))]=Q.sha(p)
        for suffix,status in [('execution-01/result.json','COMPLETE_4_FITS_2_OOF_NO_METRICS'),('finish-01/result.json','COMPLETE_FIXED_FORECAST_READOUT')]:
            p=directory/suffix;value=json.loads(p.read_text());assert value['status']==status
            assert not value.get('remaining_scope_pids') and all(not j.get('remaining_scope_pids') for j in value.get('jobs',[]))
            receipts[str(p.relative_to(ROOT))]=Q.sha(p)
        for name in ('launch.json','finish-launch.json'):
            previous=json.loads((directory/name).read_text());stat=Path('/proc')/str(previous['pid'])/'stat'
            if stat.exists():
                fields=stat.read_text().split();assert fields[21]!=previous['start_ticks'] or fields[2]=='Z'
    records=[]
    def bind(path,old=None,digest=None):
        p=Path(path);s=p.stat();r=dict(path=str(p),resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
        if old:
            for key in ('resolved','size','mtime_ns','inode'):assert r[key]==old[key],(path,key)
            digest=digest or old.get('sha256')
        if digest:
            if r['size']<=20*2**20:assert Q.sha(p)==digest
            r['sha256']=digest
        else:r['sha256']=None
        records.append(r)
    original=ROOT/'research/iterations/2026-10-01-continuation/student/final/inputs.json'
    assert Q.sha(original)==sources[str(original.relative_to(ROOT))]
    legacy=json.loads(original.read_text());assert legacy['status']=='PASS_FINAL_TEACHER_CACHE_INPUTS_WITH_LEGACY_EPOCH_LIMITATION'
    for r in legacy['records']:
        if 'path' not in r:continue
        bind(r['path'],r)
        if r.get('sidecar_sha256'):bind(Path(r['path']).with_suffix('.json'),digest=r['sidecar_sha256'])
    parent=plan['plain_parents']['-1'];p=ROOT/parent['checkpoint'];assert Q.sha(p)==parent['sha256'];bind(p,digest=parent['sha256'])
    # Chain metadata also participates in data loading even for full-data fits.
    lb=ROOT/'research/iterations/2026-10-01-late-atom-head/input-bindings.json'
    assert Q.sha(lb)=='573cd5cbaaddbda844abd4fd6736fe01de01726b2ceb118ed1b5b4d1682b8ff8'
    for r in json.loads(lb.read_text())['records']:
        if r['path'].endswith('/cache/chain_folds.npz'):bind(r['path'],r)
    g=Q.H.gpu(0);assert g['uuid']=='GPU-aa969728-9b37-ac58-7855-3f03494fb9e8' and g['free_MiB']>=8192
    assert shutil.disk_usage(ROOT).free>10*2**30
    jobs=[]
    for job in plan['jobs']:
        assert job['stage'] in ('parent','H2') and len(job['resolved_args'])==103 and job['resolved_args']['seed']==3
        j={k:job[k] for k in ('name','stage','argv','timeout_seconds','post_argv','require_absent','required_outputs')}
        if j['stage']=='H2':j['required_outputs']=[*j['required_outputs'],'artifacts/runs/'+j['name']+'/model_ep0.pt']
        for p in j['require_absent']:assert not (ROOT/p).exists() and not (ROOT/p).is_symlink()
        jobs.append(j)
    sources.update(receipts)
    for name in ('commands.json','allocate_training.py','launch.py','source-check.json','argument-proof.json','promotion-contract.json'):
        p=HERE/name;sources[str(p.relative_to(ROOT))]=Q.sha(p)
    for p,h in sources.items():assert Q.sha(ROOT/p)==h,p
    now=datetime.now(timezone.utc)
    Q.write(bp,dict(status='CURRENT_IDENTITIES_MATCH_ACCEPTED_FINAL_INPUTS',created_utc=now.isoformat(),records=records,
        teacher_lineage_limitation=legacy['accepted_lineage_limitation'],large_arrays='Prior full hashes plus unchanged inode/mtime/size; no new data scan.'))
    a=dict(execution_authorized=True,created_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=1)).isoformat(),
        execution_directory=plan['execution_directory'],success_status=plan['success_status'],sources=sources,
        input_bindings=[dict(path=str(bp.relative_to(ROOT)),sha256=Q.sha(bp))],jobs=jobs,environment=plan['environment'],
        resources=dict(gpu_id=0,gpu_uuid=g['uuid'],allocator_fraction=.375,owned_gpu_max_MiB=7168,initial_free_MiB=8192,
            running_free_MiB=1024,cpu_ids=list(range(16,22)),cpu_quota=600,host_memory_bytes=20*2**30),
        gpu_preflight=g,quality_queries=0,retries=0,final_seed=3,seed_selection=False)
    Q.write(out,a);Q.contract(out,Q.sha(out))
    print(json.dumps(dict(status='ALLOCATED_NOT_LAUNCHED',allocation_sha256=Q.sha(out),fits=2,OOF=0,deadline=a['deadline_utc'])))

if __name__=='__main__':main()
