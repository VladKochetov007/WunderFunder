from __future__ import annotations
import fcntl,hashlib,json,os,subprocess
from datetime import UTC,datetime
from pathlib import Path
MAIN=Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
ACTOR=Path('/home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile')
OPS=Path('research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011')
PACKET=MAIN/OPS/'clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2'
R21=MAIN/OPS/'monitor_20261012_1044_r21'
CONTRACT=PACKET/'CONTROL_P1_FIT_CONTRACT_R1.json'; MANIFEST=PACKET/'CONTROL_P1_LAUNCH_MANIFEST_R1.json'
RELEASE=R21/'ROOT_CONTROL_P1_FIT_RELEASE_R1.json'; BINDING=R21/'ROOT_CONTROL_P1_FINAL_BINDING_R1.json'
REVIEW=MAIN/'research/campaign_2026-10-12/independent_reviews/clean-t3-control-p1-fit-r21/INDEPENDENT_REVIEW_R1.json'
RECEIPT=ACTOR/OPS/'clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/PRELAUNCH_RESOURCE_RECLAIM_R2.json'
RUNTIME=MAIN/'research/campaign_2026-10-06/bandit_continuation/runtime'
RESERVE=5_500_000_000; ALLOWANCE=18_000_000_000; MAX_RECLAIM=3_000_000_000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def run(argv): return subprocess.run(argv,check=True,capture_output=True,text=True).stdout.strip()
def snap(c):
 st=dict(x.split() for x in (c/'memory.stat').read_text().splitlines()); ev=dict(x.split() for x in (c/'memory.events').read_text().splitlines())
 return {'memory_current_bytes':int((c/'memory.current').read_text()),'memory_file_bytes':int(st['file']),'memory_anon_bytes':int(st['anon']),'memory_swap_current_bytes':int((c/'memory.swap.current').read_text()),'memory_events_counts':{k:int(v) for k,v in ev.items()}}
if RECEIPT.exists(): raise FileExistsError(RECEIPT)
contract_sha=sha(CONTRACT); manifest_sha=sha(MANIFEST); release_sha=sha(RELEASE); review_sha=sha(REVIEW); binding_sha=sha(BINDING)
assert contract_sha=='17350c0616debacd3507ad25bc627ba2f7087ddb17e0398cedf7674d696aaf79'
assert manifest_sha=='178e5edc2a60054f0a5291d181359157b655f5ec7aef7e95364a5f8b77fa7c89'
c=json.loads(CONTRACT.read_text()); m=json.loads(MANIFEST.read_text()); rel=json.loads(RELEASE.read_text()); bind=json.loads(BINDING.read_text()); review=json.loads(REVIEW.read_text())
assert bind['fit_authorized'] is True and bind['contract_sha256']==contract_sha and bind['launch_manifest_sha256']==manifest_sha
assert bind['release_sha256']==release_sha and bind['independent_review_sha256']==review_sha and bind['independent_review_status']=='PASS_EXACT_PREPARED_P1'
assert rel['fit_authorized'] is True and rel['contract_sha256']==contract_sha and rel['launch_manifest_sha256']==manifest_sha
assert review['result']=='PASS_FOR_ROOT_CONSIDERATION'
assert review['reviewed_files']['fit_contract']['sha256']==contract_sha and review['reviewed_files']['launch_manifest']['sha256']==manifest_sha
assert m['attempt_id']==rel['attempt_id'] and m['output_absent_at_creation'] is True
out=MAIN/'artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p1'
if out.exists(): raise FileExistsError(out)
source_root=MAIN/OPS/'clean_t3_fold0_h390_r3_teacher_cohort'
small={
 Path(c['source']['trainer_path']):c['source']['trainer_sha256'],Path(c['source']['wrapper_path']):c['source']['wrapper_sha256'],
 source_root/'source_snapshot/SOURCE_MANIFEST_H390.json':'41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa',
 source_root/'SOURCE_DELTA_R3.json':'912205eba101c8539c372206e5ae0e56f2d0a39f8997b271189b3d2f7d601406',
 MAIN/'artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt':'d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e',
 MAIN/'artifacts/runs/c20261003_combo_s3_f0/provenance.json':'9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f',
 MAIN/'artifacts/runs/c20260929_prop_f0/model.pt':'b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7',
 MAIN/'research/campaign_2026-09-29/artifacts/splits/fold0.json':'09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d'}
actual={str(p):sha(p) for p in small}
for p,expected in small.items():
 if actual[str(p)]!=expected: raise ValueError(f'source/checkpoint hash mismatch: {p}')
cache=MAIN/'cache/clean_t3_f0_h390_t3r5_a9_r2'; meta=cache/'metadata.json'
if sha(meta)!='1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8':
 # Preserve exact pin failure rather than proceeding with a changed metadata source.
 raise ValueError('cache metadata hash mismatch')
cache_stats={}
for name,size in [('main_homogeneous_train.npy',1685600128),('main_homogeneous_valid.npy',224800128)]:
 p=cache/name; st=p.stat()
 if st.st_size!=size: raise ValueError(f'cache array size mismatch: {p}')
 cache_stats[str(p)]={'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
props=run(['systemctl','--user','show','wnn-campaign.slice','--no-pager','-p','ControlGroup','-p','MemoryCurrent','-p','MemoryHigh','-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec'])
parent=dict(x.split('=',1) for x in props.splitlines() if '=' in x); cg=Path('/sys/fs/cgroup')/parent['ControlGroup'].lstrip('/'); page=os.sysconf('SC_PAGE_SIZE')
high=int((cg/'memory.high').read_text()); maximum=int((cg/'memory.max').read_text()); q,period=map(int,(cg/'cpu.max').read_text().split())
if high!=(18_000_000_000//page)*page or maximum!=(20_000_000_000//page)*page or (cg/'memory.swap.max').read_text().strip()!='0' or q>5*period: raise RuntimeError('campaign cgroup caps differ from authorized limits')
with (RUNTIME/'gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 apps=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
 gpu=run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'])
 services=run(['systemctl','--user','list-units','--type=service','--state=running','--no-legend','--no-pager'])
 if apps.strip(): raise RuntimeError(f'GPU compute processes active: {apps}')
 before=snap(cg)
 if before['memory_swap_current_bytes']!=0: raise RuntimeError('campaign slice has nonzero swap use')
 needed=max(0,before['memory_current_bytes']+RESERVE-ALLOWANCE)
 requested=min(MAX_RECLAIM,((needed+500_000_000+page-1)//page)*page) if needed else 0
 if requested>before['memory_file_bytes']: raise RuntimeError('charged file cache below bounded reclaim request')
 if requested: (cg/'memory.reclaim').write_text(f'{requested} swappiness=0\n')
 after=snap(cg); host_swap=run(['swapon','--show','--bytes','--noheadings'])
 status='PASS_FRESH_RESOURCE_HEADROOM' if after['memory_current_bytes']+RESERVE<=ALLOWANCE else 'FAILED_RESOURCE_HEADROOM'
 rec={'schema':'wnn.clean-t3-control-p1-prelaunch-resource-reclaim.r2','status':status,'observed_utc':datetime.now(UTC).isoformat(),'attempt_id':m['attempt_id'],'contract_sha256':contract_sha,'launch_manifest_sha256':manifest_sha,'root_release_path':str(RELEASE),'root_release_sha256':release_sha,'final_binding_path':str(BINDING),'final_binding_sha256':binding_sha,'critic_receipt_path':str(REVIEW),'critic_receipt_sha256':review_sha,'critic_result':review['result'],'cgroup':str(cg),'page_size_bytes':page,'parent_systemd_properties':parent,'effective_limits':{'memory_high_bytes':high,'memory_max_bytes':maximum,'memory_swap_max_bytes':0,'cpu_quota':q,'cpu_period':period},'shared_gpu_lock_acquired':True,'gpu_compute_apps':apps,'gpu_state':gpu,'active_services_snapshot':services,'before':before,'after':after,'reserve_bytes':RESERVE,'working_allowance_bytes':ALLOWANCE,'requested_reclaim_bytes':requested,'maximum_authorized_reclaim_bytes':MAX_RECLAIM,'observed_memory_drop_bytes':before['memory_current_bytes']-after['memory_current_bytes'],'gate_current_plus_reserve_bytes':after['memory_current_bytes']+RESERVE,'small_source_checkpoint_hashes':actual,'cache_metadata_sha256':sha(meta),'large_cache_array_stat_after_prior_full_consumer_hash_validation':cache_stats,'prior_consumer_receipt_sha256':c['inputs']['consumer_validation']['sha256'],'profile_terminal_receipt_sha256':c['inputs']['profile_terminal']['sha256'],'host_swap_snapshot':host_swap,'campaign_parent_swap_bytes':after['memory_swap_current_bytes'],'memory_events_are_counter_counts_not_bytes':True,'memory_events_before_after':{'before':before['memory_events_counts'],'after':after['memory_events_counts']},'limits_changed':False,'reclaim_swappiness':0,'output_directory_absent':True,'fit_started':False,'execution_source_path':str(Path(__file__)),'execution_source_sha256':sha(Path(__file__))}
RECEIPT.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
print(json.dumps(rec,indent=2,sort_keys=True))
if status!='PASS_FRESH_RESOURCE_HEADROOM': raise RuntimeError('bounded reclaim did not restore fresh p1 headroom')
