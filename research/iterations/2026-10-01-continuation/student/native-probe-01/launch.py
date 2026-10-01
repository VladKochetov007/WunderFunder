"""One export then one prefix execution; separate quota/memory/wall bounded scopes."""
import datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
    plan=json.loads((HERE/'plan.json').read_text());assert plan['authorised_by_root'] is True
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert not (HERE/'launch.json').exists() and not (HERE/'student_fold0.zip').exists()
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','ARROW_NUM_THREADS']:env[k]='1'
    env['CUDA_VISIBLE_DEVICES']='';env['PYTHONUNBUFFERED']='1';env['MALLOC_CONF']='background_thread:false';env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ['WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S','PYTHONHOME','PYTHONPATH']:env.pop(k,None)
    stages=[('export',60,[str(ROOT/'.venv/bin/python'),str(HERE/'export.py'),'--members',str(HERE/'members.json'),'--native-dir',str(ROOT/'research/iterations/2026-10-01-overnight/quant3/native/build'),'--output',str(HERE/'student_fold0.zip')]),('probe',180,['/home/grinchenkoeg/scorer_image/run.sh',str(HERE/'probe.py')])]
    start=time.monotonic();receipt={'status':'STARTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(HERE/'plan.json'),'cpu':18,'sibling':58,'attempts':1,'retry_allowed':False,'stages':[]}
    write(HERE/'launch.json',receipt)
    for stage,seconds,argv in stages:
        unit='wunder-student-native-probe01-'+stage
        command=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c','18','/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',*argv]
        t=time.monotonic()
        write(HERE/(stage+'-launch.json'),{'argv':command,'timeout_seconds':seconds,'kill_grace_seconds':5,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fresh_loadavg':Path('/proc/loadavg').read_text().strip(),'fresh_cpu18_58_stat':[s for s in Path('/proc/stat').read_text().splitlines() if s.startswith(('cpu18 ','cpu58 '))],'sibling_topology':Path('/sys/devices/system/cpu/cpu18/topology/thread_siblings_list').read_text().strip()})
        with (HERE/(stage+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(HERE/(stage+'-process.json'),{'pid':p.pid,'start_ticks':int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),'unit':unit})
            code=p.wait()
        result={'stage':stage,'exit_code':code,'wall_seconds':time.monotonic()-t,'log_sha256':sha(HERE/(stage+'.log'))}
        receipt['stages'].append(result);write(HERE/(stage+'-exit.json'),result)
        if code:receipt['status']='FAILED';break
    else:receipt['status']='COMPLETE'
    receipt.update(whole_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_runtime_qualification=False)
    write(HERE/'launch-result.json',receipt);print(json.dumps(receipt),flush=True)
    if receipt['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
