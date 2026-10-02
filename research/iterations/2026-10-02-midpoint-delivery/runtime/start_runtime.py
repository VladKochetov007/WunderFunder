"""Root-allocated once-only runtime scope. Dry-run unless --execute is supplied."""
import argparse,datetime,hashlib,json,os,subprocess,uuid
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--allocation-sha256',required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 allocation=a.allocation.resolve();assert sha(allocation)==a.allocation_sha256
 value=json.loads(allocation.read_text());plan=json.loads((P/'runtime-plan.json').read_text())
 assert value['midpoint_runtime_audit_authorized'] is True and value['attempts']==1 and value['plan_sha256']==sha(P/'runtime-plan.json')
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h
 for name in ['execution','supervisor.json','outer.log']:assert not (P/name).exists() and not (P/name).is_symlink()
 cpu=int(plan['resources']['CPU']);hard=int(plan['resources']['hard_wall_seconds']);assert 3600<=hard<=4200
 unit='midpoint-final-runtime-'+uuid.uuid4().hex
 cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=100%','-p','MemoryMax=16G','-p','MemorySwapMax=0','--','taskset','-c',str(cpu),'/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(hard+5)+'s',str(ROOT/'.venv/bin/python'),str(P/'launch_runtime.py'),'--allocation',str(allocation),'--allocation-sha256',a.allocation_sha256]
 if not a.execute:print(json.dumps(dict(execute=False,argv=cmd)));return
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
 for key in ['WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S','PYTHONHOME','PYTHONPATH']:env.pop(key,None)
 with (P/'outer.log').open('x') as log:process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:start_ticks=None
 receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=process.pid,start_ticks=start_ticks,unit=unit,argv=cmd,allocation_sha256=a.allocation_sha256,state='STARTED; inspect execution/launch-result.json')
 with (P/'supervisor.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
