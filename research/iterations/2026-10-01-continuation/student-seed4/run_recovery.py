"""One root-authorized infrastructure continuation; original failed attempt stays immutable."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,time,uuid
from run_plan import ROOT,HERE,sources,required_files,gpu_info,command,parser_from_source,resolve

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def scope_pids(unit):
    cg=command(['systemctl','--user','show',unit+'.scope','-p','ControlGroup','--value'],check=True).stdout.strip()
    if not cg:return [],cg
    base=Path('/sys/fs/cgroup')/cg.lstrip('/')
    return sorted({int(v) for p in base.rglob('cgroup.procs') for v in p.read_text().split()}),cg

def monitor(unit):
    info=gpu_info(7);pids,cg=scope_pids(unit)
    raw=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],check=True).stdout
    rows=[]
    for line in raw.splitlines():
        gpu,pid,used=[x.strip() for x in line.split(',')]
        if gpu==A['GPU_uuid']:rows.append({'pid':int(pid),'used_MiB':int(used),'owned':int(pid) in pids})
    identities={}
    for pid in pids:
        p=Path('/proc')/str(pid)
        try:identities[str(pid)]={'stat':(p/'stat').read_text(),'uid':p.stat().st_uid}
        except FileNotFoundError:pass
    return dict(info,compute=rows,owned_MiB=sum(x['used_MiB'] for x in rows if x['owned']),scope_pids=pids,cgroup=cg,identities=identities)

A=json.loads((HERE/'recovery-allocation-01.json').read_text());AH=sha(HERE/'recovery-allocation-01.json')
assert A['execution_authorized'] and A['runner_sha256']==sha(Path(__file__))
assert sha(HERE/'commands.json')==A['commands_sha256']
assert sha(HERE/'execution-01/result.json')==A['original_result_sha256']
assert sha(HERE/'failure-01/failure-and-release.json')==A['failure_receipt_sha256']
plan=json.loads((HERE/'commands.json').read_text());old=json.loads((HERE/'execution-01/result.json').read_text())
assert old['status']=='FAILED_STOP_NO_RETRY' and len(old['jobs'])==5
assert all(j['exit_code']==0 for j in old['jobs'][:4]) and old['jobs'][4]['stage']=='H2'
job=plan['jobs'][-1];assert job['name']=='cont_sh256_s4h2_f1'
assert resolve(parser_from_source(),job['argv'][2:])==job['resolved_args']
parent=ROOT/'artifacts/runs/cont_sh256_s4_f1/model_ep3.pt';assert sha(parent)==A['parent_epoch3_sha256']
assert A['GPU_id']==7 and A['allocator_fraction']==.25 and A['stage_seconds']==600
assert A['owned_GPU_MiB_max']==5632 and A['minimum_initial_free_MiB']==5120 and A['minimum_running_free_MiB']==1024
assert A['CPU_ids']==[22,23,24,25,28,32] and A['host_memory_bytes']==20*2**30 and A['cpu_quota_percent']==600
sources(plan)
for name,v in plan['input_stat_bindings'].items():
    st=Path(name).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(v['size'],v['mtime_ns'],v['inode'])
for name in required_files(plan):assert (ROOT/name).is_file()
run=ROOT/'artifacts/runs'/job['name'];assert run.is_dir() and not run.is_symlink() and not list(run.iterdir())
for suffix in ('.npy','.npy.partial','.inputs.json'):
    assert not (HERE.parent/'cache'/('pred_'+job['name']+'_valid'+suffix)).exists()
info=gpu_info(7);assert info['total_MiB']==A['GPU_total_MiB'] and info['free_MiB']>=5120,info
D=HERE/'recovery-01';assert not D.exists();D.mkdir()
run.rename(HERE/'failure-01/original-empty-H2-run-directory')
env=os.environ.copy();env.update(plan['environment']);env.update(CUDA_VISIBLE_DEVICES='7',WNN_GPU_FRACTION='.25',PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic();end=start+600;records=[];status='FAILED_STOP_NO_SECOND_RETRY'
def emit(x):
    with (D/'events.jsonl').open('a') as f:f.write(json.dumps(dict(x,utc=utc()))+'\n');f.flush()
try:
    for stage,argv in [('H2',job['argv']),('OOF',job['OOF_argv'])]:
        sources(plan);assert sha(HERE/'commands.json')==A['commands_sha256'] and sha(HERE/'recovery-allocation-01.json')==AH and sha(parent)==A['parent_epoch3_sha256']
        info=gpu_info(7);assert info['free_MiB']>=5120,info
        assert time.monotonic()<end
        if stage=='OOF':assert (ROOT/job['endpoint']).is_file()
        unit='student-seed4-recovery-'+uuid.uuid4().hex
        assert command(['systemctl','--user','show',unit+'.scope','-p','LoadState','--value']).stdout.strip() in ('','not-found')
        cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','MemoryMax=21474836480','-p','MemorySwapMax=0','-p','CPUQuota=600%','--','taskset','-c','22,23,24,25,28,32','nice','-n','10','/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(max(.1,end-time.monotonic()))+'s','/usr/bin/time','-v',*argv]
        rec={'stage':stage,'unit':unit,'argv':cmd,'started_utc':utc(),'GPU_before':info};t=time.monotonic();failure=None
        with (D/(stage+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);rec['launcher_PID']=proc.pid;emit({'event':'started',**rec})
            try:
                while proc.poll() is None:
                    assert time.monotonic()<end,'600s whole-stage deadline'
                    m=monitor(unit);emit({'event':'sample','unit':unit,**m})
                    assert m['owned_MiB']<=5632,'owned GPU memory cap'
                    assert m['free_MiB']>=1024,'physical GPU headroom below1GiB'
                    time.sleep(min(1,max(0,end-time.monotonic())))
                assert proc.returncode==0,proc.returncode
            except BaseException as e:failure=repr(e);raise
            finally:
                if proc.poll() is None:
                    try:r=command(['systemctl','--user','stop',unit+'.scope']);emit({'event':'owned_scope_stop','returncode':r.returncode,'stderr':r.stderr,'unit':unit});kill=r.returncode!=0
                    except subprocess.TimeoutExpired:kill=True
                    if kill:command(['systemctl','--user','kill','--signal=KILL',unit+'.scope'])
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:command(['systemctl','--user','kill','--signal=KILL',unit+'.scope']);proc.wait(timeout=15)
                rec.update(seconds=time.monotonic()-t,exit_code=proc.poll(),failure=failure,scope_state=command(['systemctl','--user','show',unit+'.scope','-p','ActiveState','--value']).stdout.strip());records.append(rec);emit({'event':'finished',**rec})
        assert rec['scope_state'] in ('','inactive')
    status='PASS_INFRASTRUCTURE_RECOVERY_H2_F1_OOF_F1_NO_METRICS'
finally:
    result={'status':status,'utc':utc(),'seconds':time.monotonic()-start,'allocation_sha256':AH,'runner_sha256':sha(Path(__file__)),'original_result_sha256':A['original_result_sha256'],'parent_epoch3_sha256':sha(parent),'jobs':records,'metrics_computed':0}
    (D/'result.json').write_text(json.dumps(result,indent=2)+'\n')
