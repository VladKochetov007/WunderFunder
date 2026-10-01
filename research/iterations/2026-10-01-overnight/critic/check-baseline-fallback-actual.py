"""Independent metadata-only review of the two allocated fallback timing traces."""
from pathlib import Path
import datetime,hashlib,json,math
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2];D=P/'final-runtime-fallback-isa';E=D/'execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
def main():
 plan=read(D/'plan.json');allocation=read(D/'allocation.json');release=read(E/'release-check.json');results={};prior_end=None
 assert sha(D/'plan.json')=='cdfce4b8c8e923086f6aeb074b94e16cd506371385603f4bafadd500ff88274c'
 assert sha(D/'allocation.json')=='b27ed7010e370132dbe05ed1f366840d31d70cc62ed92a14588c9cf80a39032b'
 assert allocation['sources']==plan['source_sha256'] and allocation['levels_fixed_order']==['avx2','v2'] and allocation['retries']==0
 for name,h in allocation['sources'].items():assert sha(ROOT/name)==h
 assert sha(P/'final-runtime-audit/execution/launch-result.json')==allocation['full_audit_result_sha256']
 for level in ['avx2','v2']:
  r=read(E/f'{level}.json');outer=read(E/f'{level}-outer-result.json');start=read(E/f'{level}-outer-start.json');attempt=read(E/f'{level}-attempt.json')
  assert r['status']=='COMPLETE' and outer['exit_code']==0 and outer['attempt']==r['attempt']==attempt['attempt']==1 and not outer['retry_allowed']
  assert r['plan_sha256']==outer['plan_sha256']==sha(D/'plan.json') and r['allocation_sha256']==outer['allocation_sha256']==sha(D/'allocation.json')
  assert outer['worker_result_sha256']==sha(E/f'{level}.json') and outer['log_sha256']==sha(E/f'{level}.log') and outer['argv']==start['argv']
  if prior_end:assert outer['utc']>=prior_end
  prior_end=outer['ended_utc'];assert outer['elapsed_seconds']<=180
  args=outer['argv'];assert args[0]=='systemd-run' and 'CPUQuota=100%' in args and 'MemoryMax=4G' in args and 'MemorySwapMax=0' in args and args[args.index('taskset')+1:args.index('taskset')+3]==['-c','18'] and '--kill-after=2s' in args and '178s' in args
  assert r['cgroup_limits']=={'cpu.max':'100000 100000','memory.max':'4294967296','memory.swap.max':'0'}
  assert r['max_rss_KiB']<4*1024*1024 and r['max_observed_threads']==1
  assert r['zip_sha256']==plan['zip_sha256'] and r['raw_sha256']==plan['raw_sha256'] and r['reference_sha256']==plan['reference'][level]['sha256']
  assert r['native_sha256']==plan['zip_files'][f'{level}/wnn_cpp.cpython-311-x86_64-linux-gnu.so']['sha256'] and r['forced_isa_harness_only']
  assert r['rows_completed']==r['callback_rows']==100000 and r['emitted_rows']==99505 and r['sequence_count']==5
  assert r['archived_prefix_callback_slots_bitexact']==2560 and r['archived_emitted_prediction_rows_bitexact']==2065 and r['warmup_zero_placeholder_slots']==495 and not r['full100k_archived_parity_claimed'] and r['all100k_finite']
  assert not r['targets_read'] and r['WP_queries']==0
  rows=r['per_sequence'];assert len(rows)==5 and [s['sequence'] for s in rows]==list(range(5)) and all(s['rows']==20000 for s in rows)
  close(sum(s['wall_seconds'] for s in rows),r['wall_seconds']);close(sum(s['cpu_seconds'] for s in rows),r['cpu_seconds'])
  for cpu in ['cpu18','cpu58']:
   ticks=sum(s['activity'][cpu]['ticks'] for s in rows);busy=sum(s['activity'][cpu]['busy_ticks'] for s in rows)
   assert r['callback_activity'][cpu]['ticks']==ticks and r['callback_activity'][cpu]['busy_ticks']==busy
   close(busy/ticks,r['callback_activity'][cpu]['busy_fraction'])
  assert r['precheck']['cpu58']['busy_fraction']<.05 and r['callback_activity']['cpu58']['busy_fraction']<.05 and r['whole_trace_activity']['cpu58']['busy_fraction']<.05 and r['clean']
  close(r['wall_seconds']/100000*39400000/60,r['projected_callback_minutes_39_4M'])
  assert r['projection_at_most_50min']==(r['projected_callback_minutes_39_4M']<=50) and r['projection_at_most_60min']==(r['projected_callback_minutes_39_4M']<=60)
  rel=release['levels'][level];assert rel['worker_PID']==r['PID'] and not rel['worker_proc_exists'] and 'ActiveState=inactive' in rel['scope_show_stdout'] and 'SubState=dead' in rel['scope_show_stdout']
  results[level]={'integrity':'PASS','quiet':True,'projected_callback_minutes_39_4M':r['projected_callback_minutes_39_4M'],'at_most_50min':r['projection_at_most_50min'],'at_most_60min':r['projection_at_most_60min'],'result_sha256':sha(E/f'{level}.json'),'outer_receipt_sha256':sha(E/f'{level}-outer-result.json')}
 result={'status':'PASS_RECEIPT_INTEGRITY_BOTH_FALLBACKS_EXCEED_RUNTIME_THRESHOLDS','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checker_sha256':sha(__file__),'plan_sha256':sha(D/'plan.json'),'allocation_sha256':sha(D/'allocation.json'),'results':results,'scope_release_receipt_sha256':sha(E/'release-check.json'),'interpretation':'Numerically checked fallback dispatch works, but these quiet100k samples project beyond both50 and60minutes. Existing baseline remains runtime-qualified only for the measuredv3path. No new artifact/composition/ISA search or qualityreadout.','limits':['Single warm100k sample perforcedISA; no confidence interval or fullprocess extrapolation.','Archived byteparity covers2560prefixcallbackslots, notall100k outputs.','This reviewer reads receipts/source hashes only; no rawdata/model replay.'],'targets_read':False,'WP_queries':0}
 out=P/'critic/baseline-fallback-isa-actual-review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
