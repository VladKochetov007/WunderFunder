"""Bind one final student runtime using an explicit, clean plan schema."""
import datetime,hashlib,json
from pathlib import Path

P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (P/'allocation.json').exists() and not (P/'runtime-plan.json').exists()
    package=json.loads((P/'package/student_fixed4.manifest.json').read_text())
    check=json.loads((P/'package-check.json').read_text())
    assert check['status']=='PASS' and check['zip_sha256']==package['zip_sha256']
    assert sha(P/'package/student_fixed4.zip')==package['zip_sha256']
    members=['cont_sh256_final_s3h2','fin128_s3h2','fin128_s4h2','finc500b_s3h2']
    assert [Path(c['checkpoint']).parent.name for c in package['checkpoints']]==members
    assert [m['coef']for m in package['members']]==[.25]*4
    prior=json.loads((P.parent/'full4-int8/runtime-plan.json').read_text())
    sources={str(f.relative_to(ROOT)):sha(f)for f in P.glob('*.py')}
    sources['src/wnn/data.py']=sha(ROOT/'src/wnn/data.py')
    plan={k:prior[k]for k in ['input','scorer_environment','environment_sha256','library_versions']}
    plan.update(id='STUDENT-FINAL-INT8-RUNTIME-01',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_authorized=False,source_sha256=sources,
        artifact={'path':str((P/'package/student_fixed4.zip').relative_to(ROOT)),
            'sha256':package['zip_sha256'],'bytes':package['zip_bytes'],
            'files':{k:{**v,'is_dir':False}for k,v in package['files'].items()},
            'members':members,'member_spec':package['members'],'coefficients':[.25]*4,'governor':False},
        callbacks={'count':39400000,'emitted_count':39204970,'sequence_resets':1970,
            'need_prediction':'step_in_seq >= 99; no target or mask reads',
            'schedule':'validation rowgroups 0..1872 then 0..96 with fresh callback IDs',
            'warmup_none_count':195030,'prediction_arrays_saved':False},
        resources={'CPU':39,'SMT_sibling':79,'CPU_quota':'100%','RAM_bytes':16*(1<<30),'swap_bytes':0,
            'hard_wall_seconds':4200,'official_full_process_seconds':3600,'GPU_jobs':0,'new_disk_limit_bytes':200*(1<<20)},
        limits=['One actual attempt; no tuning or model changes during the trace.',
            'Current-host natural v3/VNNI path, shared host and warm filesystem cache; not a guarantee for other hardware.',
            'SMT activity is reported, not an extra veto or a timing correction.',
            'Exposed validation inputs, with 97 repeated raw sequences; runtime and correctness coverage only.'],
        package_check_sha256=sha(P/'package-check.json'))
    (P/'runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    allocation={'student_runtime_audit_authorized':True,'attempts':1,
        'authority':'Root continuation decision: complete the measured student improvement as a final artifact within the user-authorized research task.',
        'plan_sha256':sha(P/'runtime-plan.json'),'zip_sha256':package['zip_sha256'],
        'CPU':39,'CPU_quota':'100%','RAM_bytes':16*(1<<30),'swap_bytes':0,'hard_wall_seconds':4200,
        'criterion':'Complete unchanged 39.4M callback process in <=3600 seconds on the measured natural v3/VNNI host.',
        'no_public_submission':True,'no_GitHub_push':True}
    (P/'allocation.json').write_text(json.dumps(allocation,indent=2)+'\n')
    print(json.dumps({'allocation_sha256':sha(P/'allocation.json'),'plan_sha256':sha(P/'runtime-plan.json'),'zip_sha256':package['zip_sha256']}))
if __name__=='__main__':main()
