"""Independent receipt-only full trace audit; no model/data/prediction execution."""
from pathlib import Path
import datetime,hashlib,json,math
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2];D=P/'final-runtime-audit';E=D/'execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8),(a,b)
def main():
 plan=read(D/'plan.json');alloc=read(D/'allocation.json');launch=read(E/'launch-result.json');w=read(E/'worker-result.json');process=read(E/'worker-process.json');start=read(E/'launch.json')
 assert sha(D/'plan.json')=='1337643db2d5b3d07932ddc47c1a2f808d57c31a41a80c85eee04e622bda1275'
 assert sha(D/'allocation.json')=='bf83294836a822387acdac74803b6715c87290b96c66c462483f2d97ddf0e63d'
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 assert alloc['source_sha256']==plan['source_sha256']
 assert launch['worker_result_sha256']==sha(E/'worker-result.json') and launch['log_sha256']==sha(E/'worker.log')
 assert launch['plan_sha256']==w['plan_sha256']==start['plan_sha256']==sha(D/'plan.json')
 assert launch['allocation_sha256']==w['allocation_sha256']==start['allocation_sha256']==sha(D/'allocation.json')
 assert launch['status']==w['status']=='COMPLETE' and launch['exit_code']==0 and not launch['hard_timeout'] and launch['attempts']==1 and not launch['retry_allowed']
 assert process['pid']==process['process_group']==w['pid'] and process['start_ticks']>0 and process['argv']==launch['argv']==start['argv']
 assert launch['affinity']==[18] and launch['RLIMIT_AS']==16*(1<<30)
 caps={'cpu.max':'100000 100000','memory.max':str(16*(1<<30)),'memory.swap.max':'0'}
 assert launch['cgroup_limits']==caps and all(w['cgroup_limits'][k]==v for k,v in caps.items())
 assert w['native_sha256']=='dd6e59b2d75518ee06fcbe3ba49f8450b3ee83695fa5692e4b8edaf86d7c8bad'
 assert w['library_versions']=={'numpy':'2.2.6','pyarrow':'19.0.1'}
 assert w['input_identity_before']==w['input_identity_after'] and w['input_identity_before']['sha256']==alloc['input_sha256']=='2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782'
 assert sha(ROOT/'submissions/overnight_baseline3/baseline3_fixed.zip')==alloc['artifact_sha256']=='9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098'
 rows=[json.loads(line) for line in (E/'sequences.jsonl').read_text().splitlines()]
 assert len(rows)==1873 and [r['row_group'] for r in rows]==list(range(1873)) and len({r['seq_ix'] for r in rows})==1873
 assert len(w['raw_read_calls'])==1873
 for r,raw in zip(rows,w['raw_read_calls']):
  assert r['callbacks']==20000 and r['emitted']==19901 and r['wrapper_seq_ix_after']==r['seq_ix']
  assert raw=={'row_group':r['row_group'],'seq_ix':r['seq_ix'],'rows':20000}
  assert len(r['emitted_f32_sha256'])==64 and all(math.isfinite(r[k]) and r[k]>0 for k in ['callback_loop_wall_seconds','callback_loop_cpu_seconds','sequence_total_seconds'])
 assert sum(r['callbacks'] for r in rows)==w['callback_count']==37460000
 assert sum(r['emitted'] for r in rows)==w['emitted_count']==37274573
 assert w['completed_sequences']==w['unique_sequence_ids']==w['sequence_resets_observed']==1873
 assert not w['targets_decoded'] and not w['scoring_masks_read'] and w['WP_queries']==0
 for key in ['callback_loop_wall_seconds','callback_loop_cpu_seconds','io_decode_conversion_seconds','validation_checksum_seconds']:close(sum(r[key] for r in rows),w[key])
 for cpu in ['cpu18','cpu58']:
  ticks=sum(r['callback_activity'][cpu]['total_ticks'] for r in rows);busy=sum(r['callback_activity'][cpu]['busy_ticks'] for r in rows)
  assert w['callback_activity'][cpu]['total_ticks']==ticks and w['callback_activity'][cpu]['busy_ticks']==busy
  close(busy/ticks,w['callback_activity'][cpu]['busy_fraction'])
  whole=w['whole_trace_activity'][cpu];close(whole['busy_ticks']/whole['total_ticks'],whole['busy_fraction'])
 statuses=[w['process_status_after_dependencies']]+[r[k] for r in rows for k in ['status_before','status_after']]+[w['final_process_status']]
 tasks={};hwm=0
 for status in statuses:
  assert status['threads']==len(status['tasks']);hwm=max(hwm,status['vm_hwm_KiB'])
  for tid,t in status['tasks'].items():
   assert t['affinity']==[18]
   if tid in tasks:assert t['user_ticks']>=tasks[tid]['user_ticks'] and t['system_ticks']>=tasks[tid]['system_ticks']
   tasks[tid]=t
 assert max(s['threads'] for s in statuses)==w['max_observed_os_threads']==2
 assert max(hwm,w['max_rss_KiB'])<16*1024*1024
 close(w['callback_average_us'],w['callback_loop_wall_seconds']/37460000*1e6)
 close(w['callback_projection_minutes_39_4M'],w['callback_loop_wall_seconds']/37460000*39400000/60)
 close(launch['full_process_projection_minutes_39_4M'],launch['process_wall_seconds']/37460000*39400000/60)
 assert w['worker_total_seconds']<=launch['process_wall_seconds'] and launch['process_wall_seconds']<3900
 quietpre=w['quiet_precheck']['cpu58']['busy_fraction']<.05;quietcallback=w['callback_activity']['cpu58']['busy_fraction']<.05;quietwhole=w['whole_trace_activity']['cpu58']['busy_fraction']<.05
 assert quietcallback==w['callback_smt_clean'] and quietwhole==w['whole_trace_smt_clean']
 gates={'functional_complete':launch['functional_complete'],'precheck_quiet':quietpre,'callback_quiet':quietcallback,'whole_trace_quiet':quietwhole,'callback_projection_le50':w['callback_projection_minutes_39_4M']<=50,'full_process_projection_le60':launch['full_process_projection_minutes_39_4M']<=60,'observed_process_le65':launch['process_wall_seconds']<=3900}
 assert all(gates.values()) and launch['callback_runtime_qualified'] and w['callback_runtime_qualified'] and launch['process_deadline_observed']
 result={'status':'PASS_INDEPENDENT_FULL_VALID_RUNTIME_RECEIPT_AUDIT','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checker_sha256':sha(__file__),'gates':gates,'callbacks':37460000,'emitted_pairs':37274573,'sequences':1873,'callback_observed_minutes':w['callback_loop_wall_seconds']/60,'callback_projected_39_4M_minutes':w['callback_projection_minutes_39_4M'],'complete_process_observed_minutes':launch['process_wall_seconds']/60,'complete_process_projected_39_4M_minutes':launch['full_process_projection_minutes_39_4M'],'max_observed_hwm_KiB':hwm,'rusage_maxrss_KiB':w['max_rss_KiB'],'max_observed_OS_threads':2,'final_task_ticks':tasks,'SMT58_callback_busy_fraction':w['callback_activity']['cpu58']['busy_fraction'],'SMT58_whole_busy_fraction':w['whole_trace_activity']['cpu58']['busy_fraction'],'artifact_sha256':alloc['artifact_sha256'],'input_identity_same_before_after':True,'source_files_checked':len(plan['source_sha256']),'receipts':{str(f.relative_to(P)):sha(f) for f in [D/'plan.json',D/'allocation.json',E/'launch.json',E/'worker-process.json',E/'worker-result.json',E/'launch-result.json',E/'sequences.jsonl',E/'worker.log']},'targets_read':False,'model_or_empirical_inputs_executed':False,'WP_queries':0,'limits':['One target-free v3 trace on beleriand; unknown public CPU or other ISA runtime is not covered.','Complete prehash warms page cache; not a cold-cache guarantee.','All37.46M callbacks checked for finite output and wrapper/reset semantics by bound worker source; no second full-output oracle or independent inference replay.','Two observed OS threads both pinned18; helper recorded zero CPU ticks, not literal single-OS-thread execution.']}
 out=P/'critic/baseline-full-runtime-actual-review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','final_task_ticks','limits']}))
if __name__=='__main__':main()
