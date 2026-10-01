"""Start the one allocated final-runtime process in its own bounded user scope."""
import argparse,datetime,hashlib,json,subprocess,uuid
from pathlib import Path

P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    allocation=P/'allocation.json';value=json.loads(allocation.read_text())
    assert value['student_runtime_audit_authorized'] is True and value['attempts']==1
    assert value['plan_sha256']==sha(P/'runtime-plan.json')
    assert not (P/'execution').exists() and not (P/'supervisor.json').exists()
    assert not (P/'outer.log').exists()
    prior=P.parent/'full4-int8/execution/launch-result.json'
    old=json.loads(prior.read_text());assert old['status']=='COMPLETE' and old['functional_complete']
    state=subprocess.run(['systemctl','--user','show','q4-full-runtime-20261001.scope','--property=ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
    assert state in ('inactive','failed'),state
    unit='student-final-runtime-'+uuid.uuid4().hex
    cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,
         '-p','CPUQuota=100%','-p','MemoryMax=16G','-p','MemorySwapMax=0','--',
         'taskset','-c','39','/usr/bin/timeout','--kill-after=5s','4205s',
         str(ROOT/'.venv/bin/python'),str(P/'launch_runtime.py'),
         '--allocation',str(allocation),'--allocation-sha256',sha(allocation)]
    if not a.execute:print(json.dumps({'execute':False,'argv':cmd}));return
    with (P/'outer.log').open('x') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    tail=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':process.pid,
        'start_ticks':int(tail[19]),'unit':unit,'argv':cmd,'allocation_sha256':sha(allocation),
        'prior_scope_state':state,'prior_full4_result_sha256':sha(prior),
        'state':'STARTED; completion must be read from execution/launch-result.json'}
    with (P/'supervisor.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
