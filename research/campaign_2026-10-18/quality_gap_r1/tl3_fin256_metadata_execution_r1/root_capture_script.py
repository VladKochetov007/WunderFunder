import subprocess,json,pathlib,hashlib,base64
D=pathlib.Path('/Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-18/quality_gap_r1/tl3_fin256_metadata_execution_r1')
script=r'''
import subprocess,pathlib,json,hashlib,base64,datetime
B=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018');unit='computer-vision-homework-tl3-fin256-metadata-v1.service'
props=['ActiveState','SubState','ExecMainStatus','Result','ExecMainStartTimestamp','ExecMainExitTimestamp','ExecMainStartTimestampMonotonic','ExecMainExitTimestampMonotonic','MemoryPeak','MemoryCurrent','CPUUsageNSec','ControlGroup','CPUQuotaPerSecUSec','MemoryMax','MemorySwapMax','RuntimeMaxUSec']
p=subprocess.run(['systemctl','--user','show',unit]+sum((['-p',x] for x in props),[]),capture_output=True,text=True,timeout=10);assert p.returncode==0
s=dict(x.split('=',1) for x in p.stdout.splitlines())
r={'schema':'quality_gap.tl3_fin256_metadata_capture.v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'unit_properties':s}
if s['SubState'] in ['exited','failed']:
 r['journal']=subprocess.run(['journalctl','--user','-u',unit,'-n','18','--no-pager'],capture_output=True,text=True,timeout=10).stdout
 path=B/'receipts/TL3_FIN256_METADATA_R1.json'
 if path.exists():
  assert path.stat().st_size<2100000;raw=path.read_bytes();r['receipt_sha256']=hashlib.sha256(raw).hexdigest();r['receipt_bytes']=len(raw);r['receipt_base64']=base64.b64encode(raw).decode()
 parent=subprocess.check_output(['systemctl','--user','show','wnn-campaign.slice','-p','ControlGroup','--value'],text=True,timeout=10).strip()
 r['parent_memory_events']=(pathlib.Path('/sys/fs/cgroup')/parent.lstrip('/')/'memory.events').read_text()
print(json.dumps(r,indent=2,sort_keys=True))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','python3 -B -'],input=script.encode(),capture_output=True,timeout=35)
assert p.returncode==0,p.stderr.decode();r=json.loads(p.stdout)
state=r['unit_properties']['SubState']
if state not in ['exited','failed']:
 print(json.dumps({'state':state,'properties':r['unit_properties']}));raise SystemExit(0)
encoded=r.pop('receipt_base64',None);raw=base64.b64decode(encoded) if encoded else None
receipt_valid=None;problems=[]
if raw is not None:
 if hashlib.sha256(raw).hexdigest()!=r['receipt_sha256']:problems.append('receipt_transport_sha_mismatch')
 else:
  with (D/'TL3_FIN256_METADATA_R1.json').open('xb') as f:f.write(raw)
  try:
   j=json.loads(raw)
   receipt_valid=(j['schema']=='quality_gap.tl3_fin256_parent_metadata.v1' and j['status']=='METADATA_ONLY_NO_WEIGHT_VALUES_EXPORTED' and not j['cuda_initialized'] and len(j['inputs'])==2 and j['torch_version']=='2.14.0+cu130')
   if not receipt_valid:problems.append('receipt_acceptance_failed')
  except (ValueError,KeyError,TypeError):problems.append('invalid_receipt_schema')
else:problems.append('no_output_receipt')
if state!='exited' or r['unit_properties']['Result']!='success' or r['unit_properties']['ExecMainStatus']!='0':problems.append('unit_unsuccessful')
r['acceptance_errors']=problems;r['capture_status']='PASS_METADATA_ONLY' if not problems else 'TERMINAL_FAILURE_CAPTURED'
with (D/'EXECUTION_V1.json').open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
# The owned process is terminal and its evidence is durable before cleanup.
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand',"systemctl --user show computer-vision-homework-tl3-fin256-metadata-v1.service -p SubState --value"],capture_output=True,timeout=25)
assert p.returncode==0 and p.stdout.decode().strip() in ['exited','failed'],p.stderr.decode()
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','beleriand','systemctl --user stop computer-vision-homework-tl3-fin256-metadata-v1.service'],capture_output=True,timeout=25)
with (D/'STOP_V1.json').open('x') as f:json.dump({'unit':r['unit'],'after_capture':True,'terminal_substate':state,'exit_status':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()},f,indent=2);f.write('\n')
assert p.returncode==0
print(json.dumps({'state':r['capture_status'],'receipt_sha256':r.get('receipt_sha256'),'errors':problems,'properties':r['unit_properties']}))
if problems:raise SystemExit(1)
