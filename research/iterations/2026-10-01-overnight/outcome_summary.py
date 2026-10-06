"""Summarize only preserved runtime receipts; never import models or read empirical arrays."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(a,b)
def main():
 e=HERE/'execution';launch=json.loads((e/'launch-result.json').read_text());worker=json.loads((e/'worker-result.json').read_text())
 rows=[json.loads(line) for line in (e/'sequences.jsonl').read_text().splitlines() if line]
 assert launch['worker_result_sha256']==sha(e/'worker-result.json')
 assert launch['log_sha256']==sha(e/'worker.log')
 assert launch['plan_sha256']==worker['plan_sha256']==sha(HERE/'plan.json')
 assert launch['allocation_sha256']==worker['allocation_sha256']==sha(HERE/'allocation.json')
 assert launch['status']=='COMPLETE' and worker['status']=='COMPLETE' and launch['exit_code']==0
 assert len(rows)==1873 and [r['row_group'] for r in rows]==list(range(1873))
 assert len({r['seq_ix'] for r in rows})==1873
 assert all(r['callbacks']==20000 and r['emitted']==19901 and r['wrapper_seq_ix_after']==r['seq_ix'] for r in rows)
 assert sum(r['callbacks'] for r in rows)==worker['callback_count']==37460000
 assert sum(r['emitted'] for r in rows)==worker['emitted_count']==37274573
 callbacks=sum(r['callback_loop_wall_seconds'] for r in rows)
 close(callbacks,worker['callback_loop_wall_seconds'])
 close(callbacks/37460000*39400000/60,worker['callback_projection_minutes_39_4M'])
 close(launch['process_wall_seconds']/37460000*39400000/60,launch['full_process_projection_minutes_39_4M'])
 for field in ['io_decode_conversion_seconds','validation_checksum_seconds']:
  close(sum(r[field] for r in rows),worker[field])
 for cpu in ['cpu18','cpu58']:
  ticks=sum(r['callback_activity'][cpu]['total_ticks'] for r in rows);busy=sum(r['callback_activity'][cpu]['busy_ticks'] for r in rows)
  assert worker['callback_activity'][cpu]['total_ticks']==ticks and worker['callback_activity'][cpu]['busy_ticks']==busy
  close(busy/ticks,worker['callback_activity'][cpu]['busy_fraction'])
 all_tasks={}
 for r in rows:
  for which in ['status_before','status_after']:
   for tid,t in r[which]['tasks'].items():
    assert t['affinity']==[18];all_tasks[tid]=t
 assert worker['input_identity_before']['sha256']==worker['input_identity_after']['sha256']
 gates={'functional_complete':launch['functional_complete'], 'callback_39_4M_at_most_50min':worker['callback_projection_minutes_39_4M']<=50,
        'callback_SMT58_below_5pct':worker['callback_smt_clean'], 'whole_trace_SMT58_below_5pct':worker['whole_trace_smt_clean'],
        'full_process_projection_at_most_60min':launch['full_process_projection_at_most_60min'], 'actual_process_at_most_65min':launch['process_deadline_observed']}
 result={'status':'PASS' if all(gates.values()) else 'FAIL','qualification_conjunction':gates,
         'scope':'Single allocated target-free full-valid existing-baseline3 runtime audit; no quality query, model change, or fallback-ISA coverage.',
         'callbacks':worker['callback_count'],'emitted_prediction_pairs':worker['emitted_count'],'sequences':len(rows),
         'callback_minutes_observed':callbacks/60,'callback_minutes_projected_39_4M':worker['callback_projection_minutes_39_4M'],
         'full_process_minutes_observed':launch['process_wall_seconds']/60,'full_process_minutes_projected_39_4M':launch['full_process_projection_minutes_39_4M'],
         'io_decode_conversion_seconds':worker['io_decode_conversion_seconds'],'validation_checksum_seconds':worker['validation_checksum_seconds'],
         'input_hash_before_seconds':worker['input_hash_before_seconds'],'input_hash_after_seconds':worker['input_hash_after_seconds'],
         'max_rss_KiB':worker['max_rss_KiB'],'max_observed_os_threads':worker['max_observed_os_threads'],
         'all_observed_TIDs_final_CPU_ticks':all_tasks,'callback_activity':worker['callback_activity'],'whole_trace_activity':worker['whole_trace_activity'],
         'targets_decoded':worker['targets_decoded'],'scoring_masks_read':worker['scoring_masks_read'],'WP_queries':worker['WP_queries'],
         'cache_limit':'Complete input prehash warmed the page cache; this is not a cold-cache process guarantee.',
         'source_sha256':sha(__file__),'inputs':{str(p.relative_to(HERE)):sha(p) for p in [e/'launch-result.json',e/'worker-result.json',e/'sequences.jsonl',HERE/'plan.json',HERE/'allocation.json']}}
 (HERE/'outcome-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
