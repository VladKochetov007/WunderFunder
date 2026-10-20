from pathlib import Path
import datetime,hashlib,json,subprocess

B=Path('/home/grinchenkoeg/computer-vision-homework-20261018');D=B/'research_tmp/fg19_parent_rng_diag_v2'
UNIT='computer-vision-homework-fg19-parent-rng-diag-v2.service';INV='b9dcbfc0b8054e51a49fc3792fd037a0'
CAPTURE_ACCEPTANCE_SHA='655c5f12db5e166cfabf68c343aeb10d709f95db4abffeaf2925b413226837ec'
RAW_CAPTURE_SHA='2f6ced10d3812a3b3ae8354268a193323df6a22ea8523fdd2458060f4b8fcf69'
PINS={'RNG_DIAGNOSTIC_ADMISSION.json':('e0f52c8e25cfd6191822084e61c3eedf0c864604f04d7e980ebc8b2cbf0db847',1_000_000),'RNG_DIAGNOSTIC_LAUNCH.json':('ae0aaf0f14e2b38d3884adb49fb003f0581311366435a66253689600c766be1d',1_000_000),'RNG_DIAGNOSTIC.json':('0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b',256_000),'unit.log':('6986ad88dacb6fe06553d36e296f54119bd7f0696da6bd98beb56acb08c88c9e',1_000_000)}
SOURCE_SHA='75f3b53762f5faf60f785126a8dec4b24e2bf52f620fce1397d7958eb25906b2'
KEYS=('InvocationID','LoadState','ActiveState','SubState','MainPID','Result','ExecMainCode','ExecMainStatus','ControlGroup','CPUUsageNSec','MemoryCurrent','MemoryPeak')
def show(unit):
 r=subprocess.run(['systemctl','--user','show','--all',unit,*sum((['-p',k] for k in KEYS),[])],capture_output=True,text=True,timeout=15);assert r.returncode==0;return dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
def file_bytes(name,cap,sha):
 p=D/name;a=p.lstat();assert p.is_file() and not p.is_symlink() and a.st_size<=cap
 b=p.read_bytes();z=p.stat();assert len(b)==a.st_size and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns) and hashlib.sha256(b).hexdigest()==sha;return b
assert D.is_dir() and not D.is_symlink()
bound={name:file_bytes(name,cap,sha) for name,(sha,cap) in PINS.items()}
a=json.loads(bound['RNG_DIAGNOSTIC_ADMISSION.json']);l=json.loads(bound['RNG_DIAGNOSTIC_LAUNCH.json']);d=json.loads(bound['RNG_DIAGNOSTIC.json'])
assert a['status']=='DIAGNOSTIC_ONLY_NO_ACCEPTANCE' and a['unit']==UNIT and l['unit']==UNIT and l['invocation_id']==INV
assert d['schema']=='fg19.parent-fit-r2-rng-diagnostic.v2' and d['status']=='DIAGNOSTIC_ONLY_NO_ACCEPTANCE'
s=file_bytes('RNG_DIAGNOSTIC.py',12_000,SOURCE_SHA);stage=a['staged_metadata_and_source']['RNG_DIAGNOSTIC.py'];assert (len(s),SOURCE_SHA)==(stage['bytes'],stage['sha256'])
def parent_state():
 p=show('wnn-campaign.slice');assert p.get('ControlGroup') not in ('','/')
 g=Path('/sys/fs/cgroup')/p['ControlGroup'].lstrip('/');paths=[g/'cgroup.procs',*g.rglob('cgroup.procs')];assert len(paths)<128
 pop={str(x):v for x in paths if x.is_file() and (v:=x.read_text().strip())};events={k:(g/k).read_text() for k in ('memory.current','memory.events','memory.events.local') if (g/k).is_file()}
 return p,pop,events
before=show(UNIT);assert (before.get('InvocationID'),before.get('ActiveState'),before.get('SubState'),before.get('MainPID'),before.get('ExecMainCode'),before.get('ExecMainStatus'),before.get('Result'))==(INV,'active','exited','0','1','0','success')
parent_before,pop_before,events_before=parent_state()
stopped=subprocess.run(['systemctl','--user','stop',UNIT],capture_output=True,text=True,timeout=30,check=False)
after=show(UNIT);parent_after,pop_after,events_after=parent_state()
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':UNIT,'invocation_id':INV,'capture_acceptance_sha256':CAPTURE_ACCEPTANCE_SHA,'raw_capture_sha256':RAW_CAPTURE_SHA,'diagnostic_report_sha256':PINS['RNG_DIAGNOSTIC.json'][0],'before':before,'stop':{'returncode':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr},'after':after,'parent_before':parent_before,'populated_before':pop_before,'events_before':events_before,'parent_after':parent_after,'populated_after':pop_after,'events_after':events_after,'scope':'One exact stop of the successful V2 RNG diagnostic only. No native parent or failed verifier V1 unit is queried or stopped.'},sort_keys=True))
