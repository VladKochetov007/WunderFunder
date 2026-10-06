"""Wait for the owned distillation lane to finish; launch this fixed queue once."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util, json, subprocess, time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('temporal_queue',HERE/'run_queue.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)

def main():
    allocation=HERE/'training-allocation.json';digest=Q.sha(allocation)
    a=Q.contract(allocation,digest)
    prior=ROOT/'research/iterations/2026-10-02-distill07'
    expected='b5e527147ef3d38704c2c46387d91e4108a9c4984478f5d6ad2067e7ba0acf65'
    assert Q.sha(prior/'training-allocation.json')==expected
    receipt=HERE/'wait-launch-result.json';assert not receipt.exists()
    result=dict(status='WAITING',allocation_sha256=digest,prior_allocation_sha256=expected)
    try:
        while True:
            remaining=(datetime.fromisoformat(a['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
            assert remaining>2.25*3600,'Insufficient remaining fixed training budget'
            p=prior/'execution-01/result.json'
            if p.exists():
                old=json.loads(p.read_text())
                assert old['allocation_sha256']==expected
                assert all(not j.get('remaining_scope_pids') for j in old['jobs'])
                launch=json.loads((prior/'launch.json').read_text());stat=Path(f'/proc/{launch["pid"]}/stat')
                live=False
                if stat.exists():
                    fields=stat.read_text().split();live=fields[21]==launch['start_ticks'] and fields[2]!='Z'
                g=Q.H.gpu(a['resources']['gpu_id'])
                if not live and g['free_MiB']>=a['resources']['initial_free_MiB']:
                    Q.contract(allocation,digest)
                    subprocess.run(['.venv/bin/python',str((HERE/'launch.py').relative_to(ROOT))],cwd=ROOT,check=True)
                    result.update(status='LAUNCHED_FIXED_QUEUE',prior_status=old['status'],gpu_before=g)
                    break
            time.sleep(20)
    except BaseException as exc:
        result.update(status='NOT_LAUNCHED',failure=repr(exc));raise
    finally:
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        Q.write(receipt,result)

if __name__=='__main__':main()
