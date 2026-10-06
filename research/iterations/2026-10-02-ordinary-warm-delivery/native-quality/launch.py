"""Export, two independently capped OOF lanes, then one fixed metric readout."""
import datetime,json,os,subprocess,time,uuid
from pathlib import Path
from common import P,ROOT,sha,write

def main():
    allocation=json.loads((P/'allocation.json').read_text());assert allocation['authorized'] is True
    for path,digest in allocation['source_sha256'].items():assert sha(ROOT/path)==digest
    write(P/'supervisor.json',{'PID':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation_sha256':sha(P/'allocation.json'),'attempts':1,'retries':False})
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','ARROW_NUM_THREADS']:env[k]='1'
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MALLOC_CONF='background_thread:false')
    for k in ['PYTHONHOME','PYTHONPATH','WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S']:env.pop(k,None)
    receipt={'status':'STARTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation_sha256':sha(P/'allocation.json'),'stages':[]};start=time.monotonic();owned=[]
    def spawn(name,cpu,seconds,argv):
        unit='wunder-ordinary-warm-native-'+name+'-'+uuid.uuid4().hex
        cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c',str(cpu),'/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',*argv]
        log=(P/(name+'.log')).open('x');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        owned.append((p,log,unit))
        record={'name':name,'PID':p.pid,'unit':unit,'cpu':cpu,'timeout_seconds':seconds,'kill_grace_seconds':5,'argv':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_monotonic':time.monotonic()}
        try:record['start_ticks']=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        except FileNotFoundError:record['start_ticks']=None
        write(P/(name+'-launch.json'),record);print(json.dumps(record),flush=True);return p,log,record
    def finish(job):
        p,log,record=job;code=p.wait();log.close();record.update(exit_code=code,seconds=time.monotonic()-record['started_monotonic'],ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(P/(record['name']+'.log')))
        write(P/(record['name']+'-exit.json'),record);receipt['stages'].append(record);return code
    try:
        code=finish(spawn('prepare',30,60,[str(ROOT/'.venv/bin/python'),str(P/'prepare.py')]))
        assert code==0,'preparation failed; no predictions started'
        # Both fold recipes and exports are now bound before either prediction pass starts.
        jobs=[spawn('fold'+str(f),(30,35)[f],1800,['/home/grinchenkoeg/scorer_image/run.sh',str(P/'predict.py'),'--fold',str(f)]) for f in (0,1)]
        codes=[finish(job) for job in jobs];assert codes==[0,0],('fixed OOF failed; no retry or score',codes)
        assert sum(p.stat().st_size for p in P.rglob('*') if p.is_file())<1<<30
        code=finish(spawn('evaluate',30,600,[str(ROOT/'.venv/bin/python'),str(P/'evaluate.py')]))
        assert code==0,'fixed readout failed; no retry'
        receipt['status']='COMPLETE_TWO_OOF_ONE_FIXED_READOUT'
    except BaseException as exc:
        receipt['status']='FAIL_NO_RETRY';receipt['error']=repr(exc);raise
    finally:
        # Only scopes created by this invocation may be stopped; timeout remains
        # independently active in both forecast lanes if the supervisor exits.
        for process,stream,unit in owned:
            if process.poll() is None:
                subprocess.run(['systemctl','--user','stop',unit+'.scope'],timeout=15,check=False)
                process.wait(timeout=15)
            if not stream.closed:stream.close()
        receipt.update(seconds=time.monotonic()-start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output_bytes=sum(p.stat().st_size for p in P.rglob('*') if p.is_file()))
        write(P/'supervisor-result.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
