from pathlib import Path
import os,sys,subprocess,time,json
r=Path('/data/epanov_s/tabdata-causal-stem-b300-20261110-r1');sys.path.insert(0,str(r/'control/receiver-r1/receiving'));sys.path.insert(0,str(r/'control/receiver-r2'))
from b300_owned_runner_r2 import ResourceGuard,run_owned,OwnedRunError
from b300_driver_core_r1 import proc_entry
control=r/'control/native-tree-probe-r3';control.mkdir(exist_ok=False)
info=control/'rank.json'
parent="import subprocess,sys,time,json,os;from pathlib import Path;q=subprocess.Popen([sys.executable,'-c','import time;time.sleep(45)'],start_new_session=True);p=Path('/proc/'+str(q.pid)+'/stat').read_text().rsplit(')',1)[1].split();Path("+repr(str(info))+").write_text(json.dumps({'pid':q.pid,'birth':int(p[19]),'pgid':int(p[2]),'parent':os.getpid()}));time.sleep(45)"
sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(45)'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
sent_identity=proc_entry(sentinel.pid)
(control/'sentinel-owner.json').write_text(json.dumps({'pid':sentinel.pid,'birth_ticks':sent_identity.start_ticks,'purpose':'own untracked sentinel for exclusion proof'}))
try:
 guard=ResourceGuard(r,control,{'owned_rss_bytes':274877906944,'output_bytes':100000000000,'minimum_free_bytes':100000000000})
 failed=False
 try:run_owned([sys.executable,'-c',parent],root=r,control=control,cell='native-proof',stage='stop-descendants',gpu_uuids=[],owner={'pid':os.getpid()},environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''),deadline=time.time()+5,stage_timeout=5,guard=guard,progress=lambda:None)
 except OwnedRunError:failed=True
 assert failed and info.is_file();v=json.loads(info.read_bytes());rank=proc_entry(v['pid']);assert rank is None or rank.start_ticks!=v['birth'] or rank.state=='Z'
 assert sentinel.poll() is None and proc_entry(sentinel.pid).start_ticks==sent_identity.start_ticks
 receipt=json.loads((control/'owned/native-proof-stop-descendants-r2.json').read_bytes());tracked={v['pid'] for v in receipt['owned_pid_births']};assert v['pid'] in tracked and sentinel.pid not in tracked and v['pgid']==v['pid'] and v['parent']!=v['pgid']
 result={'status':'qualified','rank_new_session_counted_and_stopped':True,'unrelated_own_sentinel_preserved':True,'rank_pid':v['pid'],'sentinel_pid':sentinel.pid,'timeout_expected':True,'new_scientific_fits':0,'at':time.time()};(control/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:
 if sentinel.poll() is None:sentinel.terminate()
 sentinel.wait(timeout=10)
