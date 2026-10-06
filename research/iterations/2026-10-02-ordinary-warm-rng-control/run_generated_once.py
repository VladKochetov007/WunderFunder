"""Root-owned single bounded CPU diagnostic; no automatic retries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, os, subprocess, time, uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('resources',ROOT/'research/iterations/2026-10-01-context-replay/run.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def main():
    plan=json.loads((HERE/'generated-allocation.json').read_text());assert plan['execution_authorized']
    receipt=HERE/'generated-execution.json';log=HERE/'generated-01.json'
    assert not receipt.exists() and not log.exists() and not (HERE/'generated-01.json').exists()
    unit='ordinary-warm-rng-generated-'+uuid.uuid4().hex;assert not H.scope(unit)[1]
    argv=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%',
          '-p',f'MemoryMax={1*2**30}','-p','MemorySwapMax=0','--','taskset','-c','37',
          'nice','-n','10','timeout','--signal=TERM','--kill-after=5s','15s',*plan['argv']]
    rec=dict(status='RUNNING',argv=argv,unit=unit,allocation_sha256=H.sha(HERE/'generated-allocation.json'),
             started_utc=datetime.now(timezone.utc).isoformat());started=time.monotonic()
    try:
        with log.open('x') as f:
            rc=subprocess.run(argv,cwd=ROOT,env=dict(os.environ,**plan['environment']),stdout=f,stderr=subprocess.PIPE).returncode
        rec['exit_code']=rc;assert rc==0,('CPU diagnostic failed',rc)
        result=HERE/'generated-01.json';q=json.loads(result.read_text())
        assert q['status'] in ('PASS_GENERATED_ORDINARY_WARM_RNG',)
        rec.update(status='COMPLETE',diagnostic_status=q['status'],result_sha256=H.sha(result))
    except BaseException as exc:
        rec.update(status='FAILED_NO_RETRY',failure=repr(exc));raise
    finally:
        fields,pids=H.scope(unit)
        if pids:
            H.command(['systemctl','--user','stop',unit+'.scope']);fields,pids=H.scope(unit)
        rec.update(scope_state=fields,remaining_scope_pids=sorted(pids),seconds=time.monotonic()-started)
        with receipt.open('x') as f:json.dump(rec,f,indent=2);f.write('\n')

if __name__=='__main__':main()
